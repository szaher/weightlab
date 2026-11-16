# P1.1 delta tiles on real checkpoints

`measure_delta_tiles.py` runs a real `weightlab start` (fresh data directory, loopback, token) on the pinned Qwen2.5 checkpoints from [phase1-real-models](../phase1-real-models/) (SHA-256 checked first). For each base/Instruct pair (lineage class S, `descriptive_only` comparisons), it measures the embedding (the largest matched matrix) and `layers.0.mlp.down_proj`:

- **Direct L3 tile:** a binary 256×256-cell request, covering 2048×2048 cells of both inputs.
- **Cold pyramid build:** a tile that needs the exact delta pyramid, i.e. the 202 `pyramid_build` job, polled to completion. `direct_tile_max_mib` is set to 8, so the MLP projection also takes this path.
- **Exact tile from that pyramid,** checked cell by cell against B − A block means computed by the script from the raw file bytes (its own BF16 decoder, float64).

Raw results: [`results/delta_tiles_darwin.json`](results/delta_tiles_darwin.json). Machine: macOS 26.0, Apple arm64, 12 cores, 36 GiB RAM, Python 3.12.

| Pair | Tensor (BF16) | Bytes read (A + B) | Build wall time | Peak server RSS | Exact tile vs NumPy |
| --- | --- | ---: | ---: | ---: | --- |
| 0.5B | `embed_tokens` 151936×896 | 519 MiB | 1.41 s | 104.8 MiB peak | 75 cells, max rel. error 5.0e-8 |
| 0.5B | `layers.0.mlp.down_proj` 896×4864 | 16.6 MiB | 0.12 s | 100.3 MiB peak | 1,064 cells, 5.9e-8 |
| Math-1.5B | `embed_tokens` 151936×1536 | 890 MiB | 2.40 s | 100.9 MiB peak | 75 cells, 5.0e-8 |
| Math-1.5B | `layers.0.mlp.down_proj` 1536×8960 | 52.5 MiB | 0.23 s | 98.5 MiB peak | 840 cells, 5.9e-8 |

- Idle server RSS was 62 MiB. Across all builds and tiles in the recorded run, the peak server RSS was **at most 104.8 MiB**. This is consistent with the bounded design (a paired build holds one band of each input, not the tensors): the 890 MiB pair did not raise the peak above the 519 MiB pair. Two data points on one machine are not a proof of the bound.
- Direct L3 tiles took 53–88 ms, with peak RSS at most 101.2 MiB.
- No exact tile differed from the independent NumPy computation by more than float32 rounding.

## Limits of this evidence

- **Warm page cache.** The script hashes every checkpoint just before the run, so the files were in the OS page cache. The wall times are not disk throughput; a cold read of two checkpoints will be slower.
- **RSS is the server process only,** sampled every 50 ms with psutil. Short peaks between samples can be missed, and the OS page cache is not counted.
- **One machine with 36 GiB RAM.** This says nothing about the **8 GiB laptop target, which remains unverified**, especially for larger pairs (7B+ checkpoints, multi-shard, or tensors above `pyramid_exact_max_mib`, whose row-sampled path was not exercised here).
- Pyramid files go to the disk cache (`tile_cache_disk_mib`) and are not included in RSS.
