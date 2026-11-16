"""FastAPI health smoke tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from fall_detection.api.app import create_app
from fall_detection.config import Settings


def test_liveness_probe(app_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(app_settings())) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "fall-detection"


def test_readiness_probe_reports_every_dependency(
    app_settings: Callable[..., Settings],
) -> None:
    with TestClient(create_app(app_settings())) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    names = {check["name"] for check in body["checks"]}
    assert {"numpy", "cv2", "torch", "torchvision", "ultralytics", "database"} <= names
    # The fall layer and its event storage are reported separately.
    assert {"fall", "events"} <= names
    assert not [check for check in body["checks"] if check["status"] != "ok"], body


def test_readiness_is_independent_of_camera_state(
    app_settings: Callable[..., Settings],
) -> None:
    # FR-09 / §17.2: readiness describes models and local dependencies. No camera
    # is started here, and that must not change the readiness verdict.
    with TestClient(create_app(app_settings())) as client:
        body = client.get("/health/ready").json()
    assert body["status"] == "ok", body
    assert not any(check["status"] == "degraded" for check in body["checks"])


def test_openapi_schema_is_served(app_settings: Callable[..., Settings]) -> None:
    with TestClient(create_app(app_settings())) as client:
        schema = client.get("/openapi.json").json()

    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_readiness_omits_detection_when_disabled(
    app_settings: Callable[..., Settings],
) -> None:
    # A deliberately disabled detector is not a degradation, so readiness stays
    # "ok" and does not report a detection check at all.
    with TestClient(create_app(app_settings())) as client:
        body = client.get("/health/ready").json()
    assert body["status"] == "ok"
    assert all(check["name"] != "detection" for check in body["checks"])


def test_readiness_reports_detection_when_enabled(
    app_settings: Callable[..., Settings],
    detector_weights: Path,
) -> None:
    # §17.2: present model readiness separately from camera state. Enabling the
    # detector must surface a "detection" check, ready or not.
    settings = app_settings(
        detection={"enabled": True, "model_path": str(detector_weights), "imgsz": 320}
    )
    with TestClient(create_app(settings)) as client:
        body = client.get("/health/ready").json()
    detection = [check for check in body["checks"] if check["name"] == "detection"]
    assert len(detection) == 1
    assert detection[0]["status"] == "ok"
    assert "yolov8n-pose.pt" in detection[0]["detail"]


def test_readiness_degrades_on_missing_weights(
    app_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # A missing weight file must not stop the process: readiness reports it as
    # degraded so the operator can reach the endpoint and read the reason.
    settings = app_settings(detection={"enabled": True, "model_path": str(tmp_path / "absent.pt")})
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
    assert body["status"] == "degraded"
    detection = next(check for check in body["checks"] if check["name"] == "detection")
    assert detection["status"] == "degraded"


def test_readiness_reports_the_tracker_separately(
    app_settings: Callable[..., Settings],
) -> None:
    # §17.2: model readiness and camera state are reported separately. The tracker
    # is a third thing again, and it must not be inferred from the detector.
    with TestClient(create_app(app_settings())) as client:
        body = client.get("/health/ready").json()
    tracking = [check for check in body["checks"] if check["name"] == "tracking"]
    assert len(tracking) == 1
    assert tracking[0]["status"] == "ok"
    assert "bytetrack" in tracking[0]["detail"]
    assert body["status"] == "ok"


def test_readiness_omits_tracking_when_disabled(
    app_settings: Callable[..., Settings],
) -> None:
    # A deliberately disabled tracker is not a degradation.
    settings = app_settings(tracking={"enabled": False})
    with TestClient(create_app(settings)) as client:
        body = client.get("/health/ready").json()
    assert all(check["name"] != "tracking" for check in body["checks"])
    assert body["status"] == "ok"


def test_readiness_is_unaffected_by_an_unstarted_camera(
    app_settings: Callable[..., Settings],
) -> None:
    # A camera that was never started is a camera state, not a tracker problem, so
    # the tracker check must stay "ok" with no live session.
    with TestClient(create_app(app_settings())) as client:
        body = client.get("/health/ready").json()
    tracking = next(check for check in body["checks"] if check["name"] == "tracking")
    assert tracking["status"] == "ok"


def test_readiness_reports_the_fall_thresholds(
    app_settings: Callable[..., Settings],
) -> None:
    # The fall check names the thresholds and whether they are calibrated, so a run on
    # literature defaults can never be mistaken for a calibrated one.
    with TestClient(create_app(app_settings())) as client:
        body = client.get("/health/ready").json()
    fall = next(check for check in body["checks"] if check["name"] == "fall")
    assert fall["status"] == "ok"
    assert "literature defaults" in fall["detail"]
    assert "long-lie 3.0s" in fall["detail"]


def test_readiness_omits_fall_when_disabled(
    app_settings: Callable[..., Settings],
) -> None:
    settings = app_settings(fall={"enabled": False})
    with TestClient(create_app(settings)) as client:
        body = client.get("/health/ready").json()
    assert all(check["name"] not in ("fall", "events") for check in body["checks"])
    assert body["status"] == "ok"
