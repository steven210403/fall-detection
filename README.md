# Pemantauan Jatuh Lansia — YOLOv8-pose Fall Detection

**Rekognisi Gerakan Manusia Berbasis Computer Vision untuk Pemantauan Jatuh (Fall
Detection) pada Lansia Menggunakan YOLOv8** — Tugas Akhir, Program Studi Informatika,
Universitas Pembangunan Jaya.

A camera feed goes through YOLOv8n-pose, which finds each person and their 17 body
keypoints. ByteTrack keeps each person's identity. Three parameters are then computed for
every person on every frame:

- the bounding-box **aspect ratio** (W/H);
- the **torso angle** (shoulder→hip vs horizontal);
- the **hip vertical velocity** (body lengths per second).

They drive a four-state machine: **Normal → Falling → Impact → Fallen**. When a person stays
down for 3 s after a sudden fall, an **alarm** is stored and pushed to the dashboard. The
dashboard shows a red banner, plays a sound, and offers an acknowledge button.

The method, protocol and evaluation are in [`THESIS_PLAN.md`](THESIS_PLAN.md), and the
black-box test plan is in [`docs/BLACKBOX_TESTING.md`](docs/BLACKBOX_TESTING.md). The
previous office-activity project is preserved at the git tag `office-activity-final`.

## Quick start

Requirements: macOS, Linux or Windows, [uv](https://docs.astral.sh/uv/), Bun 1.4.3,
FFmpeg on `PATH`, and CPython 3.14.8 (uv installs it). On Windows, use
`scripts\dev.ps1` instead of `make` (see [Windows](#windows)).

```bash
make install                                   # backend + frontend from lockfiles
cp .env.example backend/.env                   # then edit backend/.env (see below)
uv run --project backend python scripts/phase0_ml_smoke.py --device mps   # downloads the 3 pose models into models/
make run                                       # migrate, API on :8000, dashboard on :5173
```

In `backend/.env`:

- Point `FALL_VIDEO__SOURCE_REF` at a video inside `FALL_MEDIA_ROOTS__ROOTS`, or use a
  webcam with `FALL_VIDEO__MODE=webcam` (or `FALL_VIDEO__MODE=rtsp` and `make camera`).
- Set `FALL_DEVICE` to `mps` (Apple Silicon), `cuda` (NVIDIA), `intel:gpu` (Intel Iris
  Xe/Arc graphics, through OpenVINO), `intel:cpu` (OpenVINO on the CPU), or `cpu`.
- A requested device that is unavailable falls back to PyTorch on `cpu` with a warning.
  `/health/ready` reports the device and runtime (`pytorch` or `openvino`) actually used.

#### Intel graphics (OpenVINO)

`intel:gpu` and `intel:cpu` run an OpenVINO export of the configured weights. The first
start exports `models/<weights>_<imgsz>_openvino_model/` (a few seconds). Later starts
reuse it, and changing `FALL_DETECTION__IMGSZ` creates a new one. Intel graphics need a
current Intel GPU driver. Check that OpenVINO sees the GPU:

```bash
uv run --project backend python -c "import openvino as ov; print(ov.Core().available_devices)"
```

The list must contain `GPU`. The OpenVINO export is a different configuration from the
`.pt` weights. The evaluation scripts (`fall_calibrate.py`, `urfd_evaluate.py`,
`pose_benchmark.py`, `local_evaluate.py`) accept only `cpu`, `mps` and `cuda`, so their
reported figures always come from PyTorch.

The webcam is read through the platform's FFmpeg capture device: AVFoundation on macOS,
V4L2 on Linux (`FALL_VIDEO__WEBCAM_DEVICE=0` is `/dev/video0`; the user must be in the
`video` group), DirectShow on Windows (the device *name*).

### Windows

Run from PowerShell in the repository root. `make` needs a POSIX shell, so
`scripts\dev.ps1` provides the same targets:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 install
Copy-Item .env.example backend\.env              # then edit backend\.env
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 weights
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 run      # API :8000 + dashboard :5173
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 devices  # DirectShow camera names
```

- Separate `FALL_MEDIA_ROOTS__ROOTS` entries with `;` (`C:\videos;D:\datasets`).
- For `FALL_VIDEO__MODE=webcam`, set `FALL_VIDEO__WEBCAM_DEVICE` to the camera name from
  `devices`, e.g. `Integrated Camera`, and allow desktop apps to use the camera
  (Settings > Privacy & security > Camera).
- `dev.ps1 check` runs every gate.
- NVIDIA GPU: set `FALL_DEVICE=cuda` (and `-Device cuda` for `weights`). The lockfile
  takes Windows PyTorch from PyTorch's CUDA 13.0 index (PyPI's Windows wheel is
  CPU-only), so a recent NVIDIA driver is needed; CUDA 13 does not support Maxwell,
  Pascal or Volta GPUs. Without a usable GPU the detector falls back to CPU.
- Intel Iris/Arc graphics (no NVIDIA GPU): CUDA cannot run on them. Set
  `FALL_DEVICE=intel:gpu` instead (see [Intel graphics](#intel-graphics-openvino)).
- `ta-docx` (the Word build of the report) is a shell script and is not ported.

Then open http://localhost:5173 and press **Mulai kamera**.

To switch to a comparison model, set `FALL_DETECTION__MODEL_PATH` to `models/yolo11n-pose.pt` or
`models/yolo26n-pose.pt`. To use the thresholds frozen on CAUCAFall, set
`FALL_FALL__CALIBRATION_PATH=datasets/manifests/fall_calibration.json`.

If an older database from the office project exists, delete it before migrating:
`rm backend/data/*.db`.

## Evaluation (THESIS_PLAN §6–§7)

All scripts run from the repository root:

```bash
# 1. Data (raw files are gitignored)
uv run --project backend python scripts/caucafall_download.py          # ~170 MB, calibration set
uv run --project backend python scripts/urfd_download.py               # ~4 GB, test set (RGB 640x480)

# 2. Calibrate on CAUCAFall only, then freeze
uv run --project backend python scripts/fall_calibrate.py --model yolov8n-pose --device mps

# 3. Score URFD once (frozen thresholds + literature baseline)
uv run --project backend python scripts/urfd_evaluate.py --model yolov8n-pose --device mps

# 4. Person-detection / pose accuracy (RQ1) on COCO-Pose val2017
uv run --project backend python scripts/pose_benchmark.py --device mps

# 5. Your own recordings (2/3/4 m, lighting, fall-like activities)
uv run --project backend python scripts/local_evaluate.py --root datasets/raw/local --device mps
```

Reports are written to `experiments/reports/`. Pose-and-tracking output is cached under
`datasets/cache/fall_observations/`, so calibration and re-scoring do not re-run the model.

To compare models, repeat steps 2–3 with `--model yolo11n-pose` and `--model yolo26n-pose`,
passing `--out datasets/manifests/fall_calibration_<model>.json` to the calibration and
`--calibration` to the evaluation.

## Layout

| Path | What |
|---|---|
| `backend/src/fall_detection/` | Python package. `video/` FFmpeg ingestion; `detection/` YOLO pose; `tracking/` ByteTrack/BoT-SORT; `fall/` geometry, velocity, state machine, consumer; `events/` the alarm writer; `api/` FastAPI; `streaming/` MJPEG + WebSocket; `database/` SQLite + Alembic; `research/` dataset adapters and evaluation |
| `frontend/` | React 19 dashboard (Bahasa Indonesia) |
| `scripts/` | Download, calibration, evaluation, and smoke-test entrypoints |
| `datasets/manifests/` | Committed CAUCAFall video manifest and frozen calibrations |
| `experiments/reports/` | JSON evidence written by the scripts |
| `docs/` | Black-box test plan, compatibility report, TA draft |

## Quality gates

```bash
make check     # ruff, basedpyright, pytest, alembic check, tsc, eslint, vitest
```

## Data licences

- **UR Fall Detection Dataset**: CC BY-NC-SA 4.0. Kwolek, B., & Kępski, M. (2014). *Human
  fall detection on embedded platform using depth maps and wireless accelerometer.*
  Computer Methods and Programs in Biomedicine, 117(3), 489–501.
- **CAUCAFall**: CC BY 4.0. Eraso, J. C., et al. (2022). *Dataset CAUCAFall: Recognition of
  human falls in an uncontrolled environment.* Data in Brief, 45, 108610.
  DOI 10.17632/7w7fccy7ky.4.
