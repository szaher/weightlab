-- WeightLab project database, migration 0002 (PR #9, real-checkpoint validation).
--
-- ADDED: model_files.ctime_ns, the shard's inode change time at import (POSIX). NULL on Windows, where st_ctime is
-- the creation time (an in-place rewrite does not move it: doc 03's documented limit) and is not compared.
-- docs/03_SYSTEM_ARCHITECTURE.md section 3 compares size, mtime_ns, ctime_ns and device + inode against the
-- identity recorded at import, and accepts only a same-size rewrite that preserves BOTH mtime_ns and ctime_ns as
-- undetectable. Without this column a same-size in-place rewrite with mtime_ns restored was served from caches.
-- NULL for rows imported before this migration: those keep the previous (size, mtime_ns, file ID) check until the
-- model is re-imported or relinked.

ALTER TABLE model_files ADD COLUMN ctime_ns INTEGER;
