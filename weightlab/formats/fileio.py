"""Read-only positioned file access with change detection (ADR-011, doc 03 section 6).

No mmap: a file truncated during a read becomes a short read (reported as FILE_CHANGED) instead
of SIGBUS. Every check compares both the open handle (fstat) and a fresh stat of the path, so an
atomic replace, rename or symlink swap of the path is detected even though the old inode stays
readable through the open handle.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from weightlab.core.errors import ErrorCode, WeightLabError, fail


@dataclass(frozen=True)
class FileIdentity:
    size: int
    mtime_ns: int
    ctime_ns: int  # POSIX: inode change time; Windows: creation time
    dev: int  # POSIX device; Windows volume serial number
    ino: int  # POSIX inode; Windows file index

    @classmethod
    def of(cls, st: os.stat_result) -> FileIdentity:
        return cls(st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino)


def change_time_ns(ident: FileIdentity) -> int | None:
    """The inode change time recorded with a fingerprint, or None where there is none to compare.

    POSIX: ctime moves on every content or metadata change and cannot be set from user space, so a same-size
    rewrite with mtime restored is still visible (doc 03). Windows: st_ctime is the creation time, which an in-place
    rewrite does not move (doc 03's documented limit), and path stat and handle fstat do not report it identically,
    so it is neither recorded nor compared there."""
    return None if sys.platform == "win32" else ident.ctime_ns


def path_identity(path: Path) -> FileIdentity:
    """Identity of whatever the path resolves to now (symlinks followed).

    Windows: `os.stat` by path may return directory-entry metadata that lags the file's own
    (observed in CI right after a write), so the path is opened briefly and the handle is
    queried, making both sides of the comparison come from GetFileInformationByHandle.
    """
    if sys.platform == "win32":
        fd = os.open(path, os.O_RDONLY | os.O_BINARY)
        try:
            return FileIdentity.of(os.fstat(fd))
        finally:
            os.close(fd)
    return FileIdentity.of(os.stat(path))


class PositionedFile:
    """One read-only handle. POSIX reads use os.pread (position-independent, thread-safe);
    Windows uses seek + read under a lock because the file position is shared."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # O_NONBLOCK: a FIFO or device swapped in after any earlier check cannot block open(); it is then
        # refused by the S_ISREG test on the handle below (doc 12 section 3: regular files only).
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            self._fd: int | None = os.open(path, flags)
        except FileNotFoundError:
            raise fail(ErrorCode.NOT_FOUND, f"file not found: {path.name}") from None
        except IsADirectoryError:
            raise fail(ErrorCode.INVALID_FORMAT, f"not a file: {path.name}") from None
        except PermissionError as e:
            if os.path.isdir(path):  # Windows reports opening a directory as EACCES
                raise fail(ErrorCode.INVALID_FORMAT, f"not a file: {path.name}") from None
            raise fail(ErrorCode.VALIDATION_ERROR, f"permission denied: {path.name} ({e.strerror})") from None
        try:
            st = os.fstat(self._fd)
            if not stat.S_ISREG(st.st_mode):
                raise fail(ErrorCode.INVALID_FORMAT, f"not a regular file: {path.name}")
            handle = FileIdentity.of(st)
            if handle != path_identity(path):
                raise fail(ErrorCode.FILE_CHANGED, f"{path.name} changed while being opened")
        except BaseException:
            self.close()
            raise
        self.identity = handle
        self.bytes_read = 0
        self._lock = threading.Lock()
        self._changed: str | None = None  # latched FILE_CHANGED reason; see invalidate()

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise fail(ErrorCode.VALIDATION_ERROR, "file is closed")
        return self._fd

    def invalidate(self, reason: str) -> WeightLabError:
        """Latch FILE_CHANGED: once any change was detected through this handle (identity, short read, or a
        digest mismatch found by the caller), every later check() and read() fails too, until the file is
        reopened. Without the latch a same-size rewrite with the mtime restored passed the next quick check on
        Windows, where ctime is the creation time and does not move (PR #3 review). Returns the error to raise."""
        if self._changed is None:
            self._changed = reason
        return fail(ErrorCode.FILE_CHANGED, reason)

    def _assert_unchanged(self) -> None:
        if self._changed is not None:
            raise fail(ErrorCode.FILE_CHANGED, f"{self._changed} (detected earlier; reopen the file)")

    def check(self) -> None:
        """Best-effort identity check of handle and path against the identity recorded at open."""
        self._assert_unchanged()
        try:
            handle = FileIdentity.of(os.fstat(self.fd))
            current = path_identity(self.path)
        except OSError:
            raise self.invalidate(f"{self.path.name} is no longer accessible") from None
        if handle != self.identity:
            raise self.invalidate(f"{self.path.name} was modified (size, time or identity)")
        if current != self.identity:
            raise self.invalidate(f"{self.path.name} was replaced, renamed or relinked")

    def read(self, offset: int, length: int) -> bytes:
        """Exactly `length` bytes at `offset`, inside the size recorded at open; short read -> FILE_CHANGED."""
        self._assert_unchanged()
        if offset < 0 or length < 0 or offset + length > self.identity.size:
            raise fail(ErrorCode.INVALID_FORMAT, "read outside the validated file range")
        if length == 0:
            return b""
        parts: list[bytes] = []
        done = 0
        if sys.platform == "win32":  # literal check so mypy narrows per platform (no os.pread on Windows)
            with self._lock:
                os.lseek(self.fd, offset, os.SEEK_SET)
                while done < length:
                    chunk = os.read(self.fd, length - done)
                    if not chunk:
                        break
                    parts.append(chunk)
                    done += len(chunk)
        else:
            while done < length:
                chunk = os.pread(self.fd, length - done, offset + done)
                if not chunk:
                    break
                parts.append(chunk)
                done += len(chunk)
        self.bytes_read += done
        if done != length:
            raise self.invalidate(f"{self.path.name} is shorter than when it was opened")
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def sha256(self, chunk_bytes: int, tick: Callable[[int], None] | None = None) -> str:
        """Digest of the recorded extent, streamed in bounded chunks (verified consistency). `tick(n)` runs after
        each chunk of n bytes; it may raise (cancel, deadline) and then no digest is returned."""
        h = hashlib.sha256()
        for off in range(0, self.identity.size, chunk_bytes):
            n = min(chunk_bytes, self.identity.size - off)
            h.update(self.read(off, n))
            if tick is not None:
                tick(n)
        self.check()
        return h.hexdigest()

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
