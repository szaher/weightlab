"""Mutation and replacement during a read or job (AC-014, B11, SEC-06).

Each case runs in two places: between open and the next read, and in the middle of a running
flat scan (after the first chunk). Expected: FILE_CHANGED, never a stale result, process alive.
The only documented best-effort gap is a same-size rewrite that also preserves every recorded
timestamp; `consistency=verified` catches it. Outcomes are attached to the JUnit report
(`record_property`) for the G0 evidence.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import run_python, write_safetensors
from weightlab import Consistency, WeightLabError, open_checkpoint

N = 64 * 1024  # elements; 256 KiB of F32 so a scan has many chunks
CHUNK = 4096
WINDOWS = sys.platform == "win32"
Mutation = Callable[[Path, Path], None]


def make(path: Path, fill: float = 1.0) -> Path:
    return write_safetensors(path, {"w": ("F32", (N,), np.full(N, fill, "<f4").tobytes())})


def truncate(p: Path, tmp: Path) -> None:
    os.truncate(p, p.stat().st_size // 2)


def append(p: Path, tmp: Path) -> None:
    with p.open("ab") as f:
        f.write(b"\0" * 16)


def rewrite_same_size(p: Path, tmp: Path) -> None:
    st = p.stat()
    make(tmp / "other.safetensors", 2.0)
    data = (tmp / "other.safetensors").read_bytes()
    assert len(data) == st.st_size
    with p.open("r+b") as f:
        f.write(data)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def rewrite_same_size_keep_mtime(p: Path, tmp: Path) -> None:
    st = p.stat()
    make(tmp / "other.safetensors", 2.0)
    with p.open("r+b") as f:
        f.write((tmp / "other.safetensors").read_bytes())
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def atomic_replace(p: Path, tmp: Path) -> None:
    make(tmp / "new.safetensors", 1.0)  # identical bytes: only identity reveals the swap
    os.replace(tmp / "new.safetensors", p)


def rename_away(p: Path, tmp: Path) -> None:
    os.rename(p, tmp / "moved.safetensors")


def rename_and_put_back_copy(p: Path, tmp: Path) -> None:
    os.rename(p, tmp / "moved.safetensors")
    make(p, 1.0)


def symlink_swap(p: Path, tmp: Path) -> None:
    target = make(tmp / "target2.safetensors", 1.0)
    link_tmp = tmp / "link.tmp"
    os.symlink(target, link_tmp)
    os.replace(link_tmp, p)


DETECTED: dict[str, Mutation] = {
    "truncate": truncate,
    "append": append,
    "same_size_rewrite_new_mtime": rewrite_same_size,
    "atomic_replace": atomic_replace,
    "rename_away": rename_away,
    "rename_and_recreate": rename_and_put_back_copy,
}


def setup_file(tmp_path: Path, symlinked: bool) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    if not symlinked:
        return make(data_dir / "model.safetensors")
    target = make(data_dir / "target1.safetensors")
    link = data_dir / "model.safetensors"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable: {e}")
    return link


def apply(mutate: Mutation, p: Path, tmp: Path) -> None:
    try:
        mutate(p, tmp)
    except PermissionError as e:
        if WINDOWS:  # an open handle without FILE_SHARE_DELETE blocks rename/replace on Windows
            pytest.skip(f"OS refused the mutation while the file is open: {e}")
        raise


def code(fn: Callable[[], object]) -> str | None:
    try:
        fn()
        return None
    except WeightLabError as e:
        return e.code.value


def scan(ck: object, consistency: Consistency = Consistency.BEST_EFFORT) -> float:
    total = 0.0
    for c in ck.iter_flat_chunks("w", CHUNK, consistency=consistency):  # type: ignore[attr-defined]
        total += float(c.sum())
    return total


@pytest.mark.parametrize("name", [*DETECTED, "symlink_swap"])
@pytest.mark.parametrize("when", ["before_read", "mid_scan"])
def test_mutation_yields_file_changed(
    tmp_path: Path, name: str, when: str, record_property: Callable[[str, object], None]
) -> None:
    mutate = symlink_swap if name == "symlink_swap" else DETECTED[name]
    p = setup_file(tmp_path, symlinked=name == "symlink_swap")
    with open_checkpoint(p) as ck:
        if when == "before_read":
            apply(mutate, p, tmp_path)
            got = code(lambda: ck.read_roi_2d("w", 0, 1, 0, 16))
        else:
            it = ck.iter_flat_chunks("w", CHUNK)
            first = next(it)
            assert first.size == CHUNK // 4
            apply(mutate, p, tmp_path)
            got = code(lambda: [float(c.sum()) for c in it])
    record_property("outcome", got)
    assert got == "FILE_CHANGED"


@pytest.mark.parametrize("when", ["before_read", "mid_scan"])
def test_same_size_rewrite_keeping_mtime(
    tmp_path: Path, when: str, record_property: Callable[[str, object], None]
) -> None:
    """Best-effort may miss it where ctime is not an inode change time (Windows: creation time);
    verified consistency must always detect it."""
    p = setup_file(tmp_path, symlinked=False)
    with open_checkpoint(p, hash_files=True) as ck:
        if when == "before_read":
            apply(rewrite_same_size_keep_mtime, p, tmp_path)
            best = code(lambda: ck.read_roi_2d("w", 0, 1, 0, 16))
            verified = code(lambda: ck.read_roi_2d("w", 0, 1, 0, 16, consistency=Consistency.VERIFIED))
        else:
            it = ck.iter_flat_chunks("w", CHUNK, consistency=Consistency.VERIFIED)
            next(it)
            apply(rewrite_same_size_keep_mtime, p, tmp_path)
            verified = code(lambda: list(it))
            best = "n/a"
    record_property("best_effort_outcome", best)
    record_property("verified_outcome", verified)
    assert verified == "FILE_CHANGED"
    if when == "before_read" and not WINDOWS:
        assert best == "FILE_CHANGED"  # POSIX ctime changes on every write and utime


def test_verified_detects_rewrite_during_job_without_recorded_digest(tmp_path: Path) -> None:
    p = setup_file(tmp_path, symlinked=False)
    with open_checkpoint(p) as ck:  # no digest at open
        it = ck.iter_flat_chunks("w", CHUNK, consistency=Consistency.VERIFIED)
        next(it)
        apply(rewrite_same_size_keep_mtime, p, tmp_path)
        assert code(lambda: list(it)) == "FILE_CHANGED"
        # Once changed, the checkpoint stays failed closed until it is reopened (no stale reads).
        assert code(lambda: ck.read_roi_2d("w", 0, 1, 0, 4)) == "FILE_CHANGED"


def test_unchanged_file_verified_scan_succeeds(tmp_path: Path) -> None:
    p = setup_file(tmp_path, symlinked=False)
    with open_checkpoint(p, hash_files=True) as ck:
        assert scan(ck, Consistency.VERIFIED) == float(N)
        assert ck.read_roi_2d("w", 0, 1, 0, 4, consistency=Consistency.VERIFIED).tolist() == [[1.0] * 4]


def test_concurrent_truncation_never_crashes_and_never_returns_stale(tmp_path: Path) -> None:
    """Readers on several threads while another thread truncates: every result is either the
    correct pre-mutation value set or FILE_CHANGED."""
    p = setup_file(tmp_path, symlinked=False)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    with open_checkpoint(p) as ck:
        start = threading.Barrier(5)

        def reader() -> None:
            start.wait()
            for i in range(200):
                try:
                    v = ck.read_roi_2d("w", 0, 1, (i * 97) % (N - 64), (i * 97) % (N - 64) + 64)
                    assert (v == 1.0).all()
                    outcomes.append("ok")
                except WeightLabError as e:
                    outcomes.append(e.code.value)
                except BaseException as e:  # pragma: no cover - would be a defect
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        start.wait()
        os.truncate(p, 100)
        for t in threads:
            t.join()
    assert not errors
    assert set(outcomes) <= {"ok", "FILE_CHANGED"}
    assert outcomes[-1] == "FILE_CHANGED" or "FILE_CHANGED" in outcomes


def test_concurrent_readers_get_correct_values(tmp_path: Path) -> None:
    """Shared handle under concurrency (Windows path uses seek+read under a lock)."""
    vals = np.arange(N, dtype="<f4")
    p = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", (256, 256), vals.tobytes())})
    ref = vals.reshape(256, 256).astype(np.float64)
    errors: list[str] = []
    with open_checkpoint(p) as ck:

        def reader(seed: int) -> None:
            rng = np.random.default_rng(seed)
            for _ in range(200):
                r0, c0 = (int(x) for x in rng.integers(0, 192, 2))
                got = ck.read_roi_2d("w", r0, r0 + 64, c0, c0 + 64)
                if not np.array_equal(got, ref[r0 : r0 + 64, c0 : c0 + 64]):
                    errors.append(f"mismatch at {r0},{c0}")

        threads = [threading.Thread(target=reader, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors


SURVIVE = """
import os, sys
from weightlab import open_checkpoint, WeightLabError
p = sys.argv[1]
with open_checkpoint(p) as ck:
    it = ck.iter_flat_chunks("w", 4096)
    next(it)
    os.truncate(p, 64)
    try:
        list(it)
        print("STALE")
    except WeightLabError as e:
        print(e.code.value)
print("alive")
"""


def test_backend_process_survives_truncation(tmp_path: Path) -> None:
    """With mmap this would be SIGBUS (exit -7 / 135); positioned reads turn it into FILE_CHANGED."""
    p = setup_file(tmp_path, symlinked=False)
    proc = run_python(SURVIVE, str(p), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["FILE_CHANGED", "alive"]
