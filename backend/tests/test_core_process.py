"""Portable process measurements (``core.process``).

The figures themselves are host-dependent; these tests assert the contract: one
shape on every OS, and ``None`` (never a fake zero) where the OS has no figure.
"""

from __future__ import annotations

import sys

import pytest

from fall_detection.core import process
from fall_detection.core.process import load_average, process_usage


def test_usage_reports_cpu_time_and_a_positive_peak() -> None:
    usage = process_usage()
    assert usage.cpu_user_s >= 0.0
    assert usage.cpu_system_s >= 0.0
    assert usage.peak_rss_bytes is not None
    assert usage.peak_rss_bytes > 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ru_maxrss units")
def test_linux_peak_is_converted_from_kilobytes(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = process._peak_rss_posix()  # noqa: SLF001
    assert raw is not None
    darwin_reading = raw if sys.platform == "darwin" else raw // 1024
    monkeypatch.setattr(process.sys, "platform", "linux")
    linux = process._peak_rss_posix()  # noqa: SLF001
    assert linux is not None
    assert linux >= darwin_reading * 1024 - 1024


def test_load_average_is_none_where_the_os_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(process.os, "getloadavg", raising=False)
    assert load_average() is None


@pytest.mark.skipif(sys.platform != "win32", reason="GetProcessMemoryInfo is Windows only")
def test_windows_reads_the_working_set() -> None:
    rss, peak = process._windows_memory()  # noqa: SLF001
    assert rss is not None
    assert peak is not None
    assert peak >= rss > 0
