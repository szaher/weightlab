"""Peak process memory for benchmarks and budget checks (doc 11: `backend_rss_mib`).

One number per OS that means "memory this process made the system commit to it", plus the
resident-set peak for reference:

* macOS: `ledger_phys_footprint_peak` from `task_info(TASK_VM_INFO)`, the figure Activity
  Monitor and the memory-pressure killer use. The resident peak (`ru_maxrss`) also counts freed
  allocations the allocator keeps cached and the kernel has not reclaimed, so it varies with the
  host rather than with WeightLab.
* Linux: `VmHWM` from /proc/self/status. Unlike `ru_maxrss`, it is per address space and starts
  fresh at `exec`; `ru_maxrss` keeps the high-water mark of the process that forked the child.
* Windows: `PeakWorkingSetSize` from `GetProcessMemoryInfo` (handles typed as pointers; untyped
  ctypes calls truncate the pseudo-handle and fail, which reads as 0).

Each function raises OSError instead of returning 0 when the OS refuses the query, so a broken
probe can never make a memory bound pass.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PeakMemory:
    metric: str  # which OS counter `peak` is
    peak: int  # bytes
    resident_peak: int  # bytes, peak resident set (ru_maxrss or equivalent)


def _ru_maxrss() -> int:
    if sys.platform == "win32":
        return 0
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _darwin() -> PeakMemory:
    class TaskVmInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("region_count", ctypes.c_int32),
            ("page_size", ctypes.c_int32),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_peak", ctypes.c_uint64),
            *[(f"_rev0_{i}", ctypes.c_uint64) for i in range(14)],  # device .. compressed_lifetime
            ("phys_footprint", ctypes.c_uint64),  # rev1
            ("min_address", ctypes.c_uint64),  # rev2
            ("max_address", ctypes.c_uint64),
            ("ledger_phys_footprint_peak", ctypes.c_uint64),  # rev3 (macOS 10.15+)
            *[(f"_tail_{i}", ctypes.c_uint64) for i in range(32)],  # later revisions, room to spare
        ]

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libc.mach_task_self.restype = ctypes.c_uint32
    libc.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    libc.task_info.restype = ctypes.c_int
    info = TaskVmInfo()
    count = ctypes.c_uint32(ctypes.sizeof(TaskVmInfo) // 4)
    task_vm_info = 22
    if libc.task_info(libc.mach_task_self(), task_vm_info, ctypes.byref(info), ctypes.byref(count)) != 0:
        raise OSError("task_info(TASK_VM_INFO) failed")
    rev3_words = (TaskVmInfo.ledger_phys_footprint_peak.offset + 8) // 4
    if count.value < rev3_words or info.ledger_phys_footprint_peak == 0:
        raise OSError("task_info returned no phys_footprint peak (macOS older than 10.15?)")
    return PeakMemory("macos_phys_footprint_peak", int(info.ledger_phys_footprint_peak), _ru_maxrss())


def _linux() -> PeakMemory:
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return PeakMemory("linux_vmhwm", int(line.split()[1]) * 1024, _ru_maxrss())
    raise OSError("VmHWM not found in /proc/self/status")


def _windows() -> PeakMemory:
    if sys.platform != "win32":
        raise OSError("not Windows")
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    if c.PeakWorkingSetSize == 0:
        raise OSError("GetProcessMemoryInfo returned a zero working set")
    return PeakMemory("windows_peak_working_set", int(c.PeakWorkingSetSize), int(c.PeakWorkingSetSize))


def peak_memory() -> PeakMemory:
    if sys.platform == "darwin":
        return _darwin()
    if sys.platform == "win32":
        return _windows()
    if sys.platform.startswith("linux"):
        return _linux()
    return PeakMemory("ru_maxrss", _ru_maxrss(), _ru_maxrss())
