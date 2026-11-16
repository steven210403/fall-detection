"""Offline evaluation tooling (THESIS_PLAN §6–§7).

* ``acquisition``: resumable, checksum-verified downloads that never re-fetch a valid file.
* ``metrics``: the confusion matrix and accuracy / precision / recall / specificity / F1.
* ``bootstrap``: percentile bootstrap confidence intervals resampled by sequence.
* ``pipeline_benchmark``: stage timing, percentiles, throughput, and memory.
* ``urfd``: the UR Fall Detection Dataset adapter (the held-out test set).
* ``caucafall``: the CAUCAFall adapter (the calibration set).
* ``replay``: plays a labelled sequence through the real fall pipeline on its
  acquisition timestamps.
* ``fall_eval``: event-level and frame-level scoring, alarm response time.
* ``calibration``: the threshold grid search, run on the calibration set only.

Nothing here is imported by the application at runtime.
"""

from __future__ import annotations

__all__: list[str] = []
