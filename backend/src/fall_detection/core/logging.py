"""Logging configuration and credential redaction.

Thesis plan §24.2 requires that credentials never reach logs, exceptions, or
FFmpeg stderr output. Two mechanisms enforce that:

* :func:`redact_url` and :func:`redact_text` replace the userinfo section of any
  URL (``rtsp://user:secret@host/...`` → ``rtsp://***@host/...``) and are used
  by the video layer before a URL is logged or embedded in an exception.
* :class:`RedactionFilter` applies :func:`redact_text` to every record as a last
  line of defence, so a message that accidentally interpolates a raw URL is
  still scrubbed before it is emitted.

Configuration is idempotent: :func:`configure_logging` replaces the handlers it
installed on the ``fall_detection`` logger and leaves other libraries alone.
"""

from __future__ import annotations

import logging
import re
from typing import Final

__all__ = [
    "RedactionFilter",
    "configure_logging",
    "get_logger",
    "redact_text",
    "redact_url",
]

LOGGER_NAME: Final = "fall_detection"
REDACTED: Final = "***"

# Matches ``scheme://userinfo@`` where userinfo may contain a password. Only the
# userinfo section is replaced; scheme, host, port, and path stay readable so
# logs remain diagnostically useful.
_URL_USERINFO_RE: Final = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<userinfo>[^/@\s]+)@",
)

# Catches credentials passed as explicit parameters rather than in a URL, e.g.
# ``?password=...`` or ``--password ...``.
_CREDENTIAL_PARAM_RE: Final = re.compile(
    r"(?P<key>(?:password|passwd|pwd|secret|token|api[_-]?key)=)(?P<value>[^&\s'\"]+)",
    re.IGNORECASE,
)

# An RTSP/HTTP URL without userinfo is left untouched; nothing else to do.


def redact_url(url: str) -> str:
    """Return ``url`` with any userinfo section replaced by ``***``.

    Args:
        url: A possibly credential-bearing URL.

    Returns:
        The URL with credentials removed. Non-URL strings are returned as-is
        apart from credential query parameters.
    """
    redacted = _URL_USERINFO_RE.sub(lambda m: f"{m['scheme']}{REDACTED}@", url)
    return _CREDENTIAL_PARAM_RE.sub(lambda m: f"{m['key']}{REDACTED}", redacted)


def redact_text(text: str) -> str:
    """Scrub every credential pattern found in free-form ``text``."""
    return redact_url(text)


class RedactionFilter(logging.Filter):
    """Rewrite log records so credentials never leave the process.

    Handlers only see records that reach them, and a filter attached to one
    logger is not applied to records emitted by its children. This filter is
    therefore a second line of defence; the primary mechanism is the record
    factory installed by :func:`_install_record_factory`, which redacts the
    message where the record is created.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = redact_text(record.getMessage())
        if message != record.getMessage():
            record.msg = message
            record.args = ()
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        return True


def _install_record_factory() -> None:
    """Redact every :class:`logging.LogRecord` at creation time.

    A logger-level filter does not apply to records produced by child loggers,
    and a handler added later (by pytest's ``caplog``, for example) carries no
    filter of its own. Redacting inside the record factory is the only place
    that covers every logger and every handler, so a credential interpolated
    into a message can never reach any sink.
    """
    current = logging.getLogRecordFactory()
    if getattr(current, "_cctv_redacting", False):
        return
    original = current

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = original(*args, **kwargs)  # type: ignore[arg-type]
        message = record.getMessage()
        redacted = redact_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return record

    factory._cctv_redacting = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


def configure_logging(level: str = "INFO", *, stream: bool = True) -> None:
    """Install the project's logging handler on the application logger.

    Args:
        level: A standard logging level name.
        stream: When ``True``, emit to stderr. Tests pass ``False`` to install
            only the redaction filter.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    _install_record_factory()
    # Remove handlers this function previously installed so repeated calls
    # (for example in tests) do not duplicate output.
    for handler in [h for h in logger.handlers if getattr(h, "_cctv_handler", False)]:
        logger.removeHandler(handler)
        handler.close()

    logger.addFilter(RedactionFilter())
    if not stream:
        return

    handler = logging.StreamHandler()
    handler._cctv_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(RedactionFilter())
    logger.addHandler(handler)
    # Do not double-log through the root logger; the app owns its own handler.
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced application logger.

    Args:
        name: Optional sub-logger name, e.g. ``"video.decoder"``. When omitted
            the root application logger is returned.
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


# Redaction must hold even if ``configure_logging`` was never called (a test may
# import a module and log directly), so the record factory is installed as soon
# as this module is imported.
_install_record_factory()
