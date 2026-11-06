# 11 — Laptop Performance and Resource Budget Specification

## 1. Hardware profiles and release targets

**Minimum target:** 8 GiB RAM, modern dual/quad-core CPU, SSD, integrated graphics only, macOS/Windows/Linux.  
**Recommended:** 16 GiB RAM, 4+ cores, SSD, browser with Canvas2D.  
**Storage:** Checkpoint itself remains on existing local disk; application needs hundreds of MB for runtime, plus configurable cache; large exact scans require I/O but not a copy of full model.  
**Optional inference:** May require substantially more RAM/CPU and is never included in the R1 guarantees.

Memory targets to **verify**, not guaranteed present-day benchmarks:

| Component | Default cap / target | Failure response |
|---|---:|---|
| Backend steady-state working RSS | <= 300 MiB target idle | investigate leaks |
| Backend peak during R1 bounded tasks | <= 1 GiB target | reject/queue if estimate exceeds cap |
| Backend tile cache | 128 MiB LRU | evict |
| Frontend decoded tile cache | 64 MiB target | evict |
| Exact ROI table (`roi_max_cells`) | <= 4096 cells initial | prompt to narrow/export as stream |
| Tile payload (`max_tile_cells`) | <= 256x256 = 65,536 cells (320 KiB binary incl. status) | no oversized tile responses |
| Exact pyramid (`pyramid_exact_max_mib`) | auto-built on demand per source/view up to 1024 MiB on disk; all levels cost 0.323 B per cell (56 B base + 80 B coarse records): ~16.1% of BF16, 8.1% of F32, 4.0% of F64 tensor bytes | larger sources: row-sampled preview + explicit job; pyramids that do not fit the disk cache stay in memory for the session or are refused |
| Direct tile read (`direct_tile_max_mib`) | 64 MiB covered bytes | above it: pyramid |
| LoRA scan (`max_lora_scan_gmac`) | 50 GMAC per module (rows x cols x rank) | reject with estimate; report module as not analyzed |
| Active tile jobs | 2 | queue/cancel stale |
| Heavy full-scan jobs | 1 | queue |
| CPU workers | max 2 by default | config controls |
| Disk tile cache | 1 GiB cap default | LRU/age eviction |
| Concurrent model handles | max 16 | LRU close |
| UI in-flight requests | <= 6 | abort obsolete requests |

RSS alone cannot certify absence of memory pressure from mmap/page cache or browser allocations. Measure system pressure/swap and frontend process usage.

## 2. Request-class budget

| Operation | Expected complexity | R1 mode | Work limit |
|---|---|---|---|
| Metadata catalog | O(number of tensor headers/shards) | immediate | cap header size and count |
| Exact ROI | O(selected rows/columns + page accesses) | immediate | 4096 cells, I/O estimate cap |
| Tile, covered bytes <= `direct_tile_max_mib` (all L0-L3) | O(covered cells), sequential row stripes | interactive | max bytes read |
| Tile from ready pyramid | O(output cells) | interactive | 202 + deduplicated build job when not ready |
| Exact pyramid build | O(N) sequential, once per tensor version | background, automatic up to limit | cancel, progress |
| Row-sampled tile (above limit) | O(sampled rows x cols), sequential | interactive | fixed row count, max bytes read |
| Exact aggregate | O(underlying elements) | background | explicit approval; bounded chunking |
| Exact tensor statistics | O(N elements) | background | cancel, progress, timeout |
| Pairwise row cosine all rows | O(n²d) | unavailable R1 | selected/sample only |
| Full SVD | superlinear, high RAM | unavailable R1 | only bounded optional R2 |
| LoRA ROI | O((rows+cols)r + rows*cols*r) | interactive bounded | r and dimensions capped |
| LoRA full-module change report (FR-023) | O(out*in*r) multiply-adds per module; Frobenius norm via `s^2 * tr((B^T B)(A A^T))` in O((in+out)r^2) only if independently tested against the direct product | background | `max_lora_scan_gmac`, cancel, completeness flag |
| Full model evaluation | model load + decoding | optional R3 | separate subprocess and preflight |

## 3. I/O estimates

To compute a mean/RMS over every cell in a 2D tensor, the system must read every relevant numeric value at least once absent cached summaries. A 7B F16 model is around 14GB of raw parameter bytes before metadata/extra tensors. Bounded reads avoid copying that entire array into application heap, but scans still read every byte. Scattered access is worse than it looks: sampling k cells per block for a 256x256 tile is ~4M random accesses, each potentially a separate 4 KiB page, which exceeds the cost of a sequential exact scan for most single tensors. Hence exact pyramid first, row sampling only for very large tensors (doc 03 section 5). Display estimated bytes/latency class and let user explicitly request the operation.

## 4. Benchmark suite

- **B01 Catalog:** 2/10/20 GB synthetic SafeTensors fixture, with thousands of tensor keys; measure import seconds and peak backend RSS.
- **B02 Heatmap:** 8192x8192 BF16 matrix: cold pyramid build time and RSS, then L0/L3/L5 tile p50/p95 warm/cold, bytes read, cache hit ratio. Plus a 128256x8192 BF16 (~2 GiB) embedding: row-sampled preview latency and bytes read, then queued exact pyramid.
- **B03 Exact ROI:** rectangular 32x32 and 64x64 at distant offsets; verify exact values, p95 and RSS.
- **B04 Stats:** full 1GB synthetic tensor scan; cancellation at 10% and complete; RSS, CPU load, bytes scanned and time.
- **B05 Compare:** same-shape 1GB pair delta sampled/exact; correctness and resource caps.
- **B06 LoRA:** 8192x8192 base and rank 8/32 adapters; 128x128 effective ROI without full `B@A` materialization.
- **B07 Browser:** 20 minutes repeated zoom/pan, multiple tensors, heap/DOM usage; no monotonic leak.
- **B08 Adversarial:** malformed header, huge declared dimensions, zip/path traversal, cancel/restart, changed file.
- **B09 Slow disk:** rate limited reads, UI remains interactive; task cancel works.
- **B10 Windows/Linux/macOS:** same numeric fixture hashes and no OS-only file-mapping failures.
- **B11 Mutation:** truncate, append, same-size rewrite, atomic replace/rename and symlink swap during a running job on each OS; backend alive, `FILE_CHANGED` or verified-snapshot result, never a stale result.
- **B12 Pyramid footprint:** actual pyramid bytes vs the 0.323 B/cell estimate for BF16/F32/F64 tensors; cache admission and eviction under `tile_cache_disk_mib`.

## 5. Performance definition of done

For R1, the above benchmark suite must publish actual results including machine RAM/CPU/OS, cold vs warm cache, benchmark file size, measured RSS, and reason for deviations. First-release target: backend peak RSS <= 1 GiB in approved interactive and streaming workloads, idle <= 300 MiB; UI responsive during job; zero unbounded allocations. If target fails, narrow limits or design, do not claim achievement.

## 6. Guardrail implementation

- Enforce configurable `memory_budget_mib`, `scan_read_budget_mib`, `max_tile_size`, `max_roi_cells`, `max_threads`, `max_jobs`, `cache_mib`, `max_wall_seconds`, `max_file_handles` centrally.
- Set NumPy/BLAS thread counts before importing computational packages, e.g. `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` if applicable.
- Use semaphores for concurrent tasks and bounded queues (reject with 429 when full).
- Convert only chunks; limit temporary arrays to 16-64 MiB defaults. Treat cost of dtype promotion and paired A/B chunks as x2-3 budget.
- Background process has enforced timeout and kill on cancel. Files opened read-only; no checkpoint modification.
- Use browsers' request abort and tile cache eviction when view changes.

## 7. Metrics

`api_latency_ms`, `scan_bytes_estimated`, `scan_bytes_actual`, `tile_cache_hit`, `backend_rss_mib`, `system_memory_available_mib`, `open_file_count`, `job_queue_depth`, `job_cancel_latency_ms`, `frontend_heap_mib` where measurable. Telemetry stored locally only; no network export.

## 8. Nonnegotiable proof of memory efficiency

Use a synthetic file *larger than available RAM* (sparse/allocated test where supported) and inspect a small rectangular ROI while observing process RSS; then compare against independent exact reference. Include real SSD fixture smoke where possible because sparse files hide physical I/O cost. Test on at least one 8GB machine before claiming the laptop promise; CI runners are not an adequate substitute for real device validation.
