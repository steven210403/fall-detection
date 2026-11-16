"""Camera and preview endpoints.

The operator's control surface: list the camera, start and stop it, and watch the
annotated preview (boxes, skeletons, and each person's fall state).

* **Start/stop are idempotent**: they delegate to the supervisor's own idempotent
  methods, so a duplicated request from a browser cannot create a second decoder.
* **No endpoint leaks a credential.** Camera metadata is built from the resolved source's
  ``safe_ref`` and ``describe()``, which redact the userinfo section.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Path, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from fall_detection.api.deps import AppStateDependency
from fall_detection.core.errors import NotFoundError
from fall_detection.core.logging import get_logger
from fall_detection.database import CameraRepository
from fall_detection.schemas.api import CameraModel, CameraStateModel

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from fall_detection.api.state import AppState

__all__ = ["router"]

logger = get_logger("api.cameras")

router = APIRouter(prefix="/api/v1", tags=["cameras"])

BOUNDARY = "cctvframe"
"""MIME boundary used to frame the MJPEG stream.

A fixed boundary is safe here because the payload is JPEG, whose segment markers cannot
contain a 9-byte ASCII boundary followed by a CRLF — the same argument every MJPEG
implementation relies on. §18.2 chooses MJPEG for the MVP and records its bandwidth cost,
so the format is a decision rather than a default.
"""


@router.get("/cameras", response_model=list[CameraModel], summary="Configured cameras")
def list_cameras(state: AppStateDependency) -> list[CameraModel]:
    """Return camera metadata without credentials (§17.2).

    Reports the *configured* camera even before it has been started, because an operator
    needs to see what the process is configured to read. A camera row that already exists
    in the database contributes its stored name, so the history and the running process
    agree on what the camera is called.
    """
    camera_id = state.camera_id
    pipeline = state.pipeline
    session = state.current_session()
    stored_name: str | None = None

    def read(session: Session) -> str | None:
        row = CameraRepository(session).get(camera_id)
        return None if row is None else row.name

    try:
        stored_name = state.writer.read(read)
    except Exception as exc:  # noqa: BLE001 - a database problem must not hide the camera
        logger.warning("could not read the stored camera row: %s", exc)

    return [
        CameraModel(
            camera_id=str(camera_id),
            name=stored_name or state.settings.camera_name,
            source_type=state.settings.video.mode,
            source_ref=_safe_ref(state),
            state=state.camera_state.value,
            stream_session_id=None if session is None else str(session.stream_session_id),
            attempt=None if session is None else session.attempt,
            stream=_as_dict(None if pipeline is None else pipeline.snapshot().get("stream")),
            source=_as_dict(None if pipeline is None else pipeline.snapshot().get("source")),
        )
    ]


@router.post(
    "/cameras/{camera_id}/start",
    response_model=CameraStateModel,
    summary="Start the configured source",
)
def start_camera(
    state: AppStateDependency,
    camera_id: Annotated[uuid.UUID, Path()],
) -> CameraStateModel:
    """Start decoding (FR-11).

    Idempotent: a second request while the camera is live returns the current state
    without starting a second decoder.

    Raises:
        404: The camera identifier is not the configured one.
        400: The source is misconfigured or unusable (FR-01).
        503: No pipeline is available because detection and tracking are disabled.
    """
    _require_camera(state, camera_id)
    state.start_camera()
    session = state.current_session()
    return CameraStateModel(
        camera_id=str(camera_id),
        state=state.camera_state.value,
        stream_session_id=None if session is None else str(session.stream_session_id),
        started=True,
    )


@router.post(
    "/cameras/{camera_id}/stop",
    response_model=CameraStateModel,
    summary="Stop the pipeline",
)
def stop_camera(
    state: AppStateDependency,
    camera_id: Annotated[uuid.UUID, Path()],
) -> CameraStateModel:
    """Stop decoding and close the session (FR-11).

    Idempotent: stopping a camera that is already stopped is a no-op, and every open
    fall alarm is closed with reason ``stopped`` rather than left for the next run.
    """
    _require_camera(state, camera_id)
    state.stop_camera(reason="stopped")
    return CameraStateModel(camera_id=str(camera_id), state=state.camera_state.value, stopped=True)


@router.get(
    "/cameras/{camera_id}/snapshot.jpg",
    summary="Current annotated frame as a single JPEG",
    response_class=Response,
)
def snapshot(
    state: AppStateDependency,
    camera_id: Annotated[uuid.UUID, Path()],
) -> Response:
    """Return the newest annotated frame as one JPEG (§18.1).

    A still frame for documentation and for clients that cannot read MJPEG.

    The response carries the *source* frame's dimensions in headers, so the browser can
    compute the letterboxing scale without a second request.

    Raises:
        404: No frame has been encoded yet.
    """
    _require_camera(state, camera_id)
    preview = state.preview
    frame = None if preview is None else preview.latest()
    if frame is None:
        raise NotFoundError("no preview frame is available yet; start the camera first")
    headers = {
        "Cache-Control": "no-store",
        "X-Frame-Seq": str(frame.frame_seq),
        "X-Preview-Size": f"{frame.width}x{frame.height}",
        "X-Source-Size": f"{frame.source_width}x{frame.source_height}",
    }
    return Response(content=frame.jpeg, media_type="image/jpeg", headers=headers)


@router.get(
    "/cameras/{camera_id}/preview.mjpeg",
    summary="MJPEG preview",
    response_class=StreamingResponse,
)
def preview_mjpeg(
    state: AppStateDependency,
    request: Request,
    camera_id: Annotated[uuid.UUID, Path()],
    max_frames: Annotated[int | None, Query(ge=1, le=100_000)] = None,
) -> StreamingResponse:
    """Stream the annotated preview as MJPEG (§18.2).

    §18.2 chooses MJPEG for the MVP: "High because every frame is JPEG | Adequate for a
    5 FPS preview; limited browser frame control | Low [complexity]". The stream is
    rate-limited by :class:`~fall_detection.streaming.frames.PreviewStream`, not here, so
    the same cap applies to every consumer.

    The generator stops when the client disconnects, so a closed tab does not leak a
    worker. It also stops on a session change: mixing two decoding attempts in one stream
    would be exactly the pairing of mismatched frames §18.1 forbids.

    ``max_frames`` exists for a bounded read in a test or a script; a browser leaves it
    unset and reads until it navigates away.
    """
    _require_camera(state, camera_id)
    preview = state.preview
    if preview is None or not preview.enabled:
        raise NotFoundError("the preview is disabled in configuration")

    def frames() -> Any:
        sent = 0
        last_sequence = 0
        while True:
            if max_frames is not None and sent >= max_frames:
                return
            frame = preview.latest()
            if frame is None or frame.sequence == last_sequence:
                # Poll rather than block: the encoder is rate-limited, and a blocking
                # read would need a condition variable per client for a frame that is
                # produced at 5 Hz. The sleep follows the configured rate so the loop
                # cannot spin.
                import time

                time.sleep(min(0.05, preview.min_interval_s))
                continue
            last_sequence = frame.sequence
            sent += 1
            yield (
                b"--" + BOUNDARY.encode("ascii") + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame.jpeg)).encode("ascii") + b"\r\n"
                b"X-Frame-Seq: "
                + str(frame.frame_seq).encode("ascii")
                + b"\r\n\r\n"
                + frame.jpeg
                + b"\r\n"
            )

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


# -- helpers -----------------------------------------------------------


def _as_dict(value: object) -> dict[str, Any] | None:
    """Narrow a snapshot's untyped value to a dict, or ``None``.

    ``PipelineSupervisor.snapshot`` is a ``TypedDict`` whose nested ``stream`` and ``source``
    entries are ``object`` in the response model, so the shape is checked here instead of
    being asserted at the boundary.
    """
    return dict(value) if isinstance(value, dict) else None  # type: ignore[arg-type]


def _require_camera(state: AppState, camera_id: uuid.UUID) -> None:
    """Reject a request for a camera this process does not run (FR-09).

    The MVP is one fixed camera (§6), so an unknown identifier is a client error rather
    than a request for a second camera. Reported as ``404`` with a typed code so a
    dashboard can tell "wrong camera" from "camera offline".
    """
    if camera_id != state.camera_id:
        raise NotFoundError(
            f"camera {camera_id} is not configured; this process runs {state.camera_id}"
        )


def _safe_ref(state: AppState) -> str:
    """Credential-free source reference (§17.2, §24.2)."""
    pipeline = state.pipeline
    if pipeline is None:
        return state.settings.video.source_ref or ""
    snapshot = pipeline.snapshot()
    source = snapshot.get("source")
    if isinstance(source, dict) and "safe_ref" in source:
        return str(source["safe_ref"])
    return state.settings.video.source_ref or ""
