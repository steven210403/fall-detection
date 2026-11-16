"""Pydantic and settings smoke tests."""

from __future__ import annotations

import pydantic
import pytest
from pydantic import ValidationError

from fall_detection.config import Settings, get_settings
from fall_detection.schemas.health import HealthReady, ReadinessCheck


def test_pydantic_v2_runtime() -> None:
    assert pydantic.VERSION.startswith("2."), pydantic.VERSION


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FALL_DEVICE", raising=False)
    settings = Settings()
    assert settings.device == "cpu"
    assert (settings.preview_max_width, settings.preview_max_height) == (960, 540)


def test_settings_reject_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALL_DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_accepts_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALL_DEVICE", "mps")
    settings = Settings()
    assert settings.device == "mps"


def test_sqlite_path_is_derived() -> None:
    settings = Settings(database_url="sqlite+pysqlite:///./data/x.db")
    assert str(settings.database_path).endswith("data/x.db")


def test_readiness_schema_round_trip() -> None:
    payload = HealthReady(
        status="degraded",
        checks=[ReadinessCheck(name="torch", status="unavailable", detail=None)],
    )
    restored = HealthReady.model_validate_json(payload.model_dump_json())
    assert restored == payload


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
