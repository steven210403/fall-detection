"""Process resource measurements that work on macOS, Linux and Windows.

The ``resource`` module and ``os.getloadavg`` exist only on POSIX, and
``ru_maxrss`` is bytes on macOS but kilobytes on Linux. Windows exposes the same
figures through ``GetProcessMemoryInfo``. Callers get one shape everywhere; a
figure the platform does not provide is ``None``, never zero.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

__all__ = ["ProcessUsage", "load_average", "process_usage"]


@dataclass(frozen=True, slots=True)
class ProcessUsage:
    """Memory and CPU time of the current process."""

    rss_bytes: int | None
    """Current resident set (working set on Windows)."""

    peak_rss_bytes: int | None
    """Peak resident set (peak working set on Windows)."""

    cpu_user_s: float
    cpu_system_s: float


def process_usage() -> ProcessUsage:
    """Return this process's memory and CPU time on the running platform."""
    times = os.times()  # user/system CPU time on every platform, Windows included
    if sys.platform == "win32":
        rss, peak = _windows_memory()
    else:
        rss, peak = _current_rss_posix(), _peak_rss_posix()
    return ProcessUsage(
        rss_bytes=rss,
        peak_rss_bytes=peak,
        cpu_user_s=times.user,
        cpu_system_s=times.system,
    )


def load_average() -> tuple[float, float, float] | None:
    """The 1/5/15-minute load average, or ``None`` where the OS has none (Windows)."""
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return getloadavg()
    except OSError:
        return None


def _peak_rss_posix() -> int | None:
    try:
        import resource  # noqa: PLC0415 - POSIX only
    except ImportError:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ``ru_maxrss`` is bytes on macOS and kilobytes on Linux; reporting the raw Linux
    # value as bytes would understate it by 1024x.
    return peak if sys.platform == "darwin" else peak * 1024


def _current_rss_posix() -> int | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except OSError, ValueError, IndexError:
        pass
    try:
        import psutil  # noqa: PLC0415 - optional, only for a live figure  # pyright: ignore[reportMissingModuleSource]

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - a missing optional reader is not an error
        return None


def _windows_memory() -> tuple[int | None, int | None]:
    """Working set and peak working set from ``GetProcessMemoryInfo``."""
    if sys.platform != "win32":  # pragma: no cover - narrows the platform for type checkers
        return None, None
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class _Counters(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - ctypes layout
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

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except OSError:
        return None, None
    if not ok:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
