"""Detection class vocabulary and mapping to raw pretrained labels.

The fall detector uses a YOLO pose model, whose only class is ``person``. Rather than
hard-coding a class index, the mapping resolves at load time from the model's own
``names`` dictionary, and a checkpoint without a ``person`` class is refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from fall_detection.config import DEFAULT_DETECTION_CLASSES, DetectionClass
from fall_detection.core.errors import ModelUnavailableError

__all__ = [
    "PRIMARY_PERSON_CLASS",
    "RAW_LABEL_ALIASES",
    "class_id_map",
    "resolve_classes",
]

PRIMARY_PERSON_CLASS: Final[DetectionClass] = "person"
"""The one class whose absence means "no people are visible"."""

RAW_LABEL_ALIASES: Final[dict[str, DetectionClass]] = {
    # Raw label in a pretrained vocabulary -> detection class. A YOLO pose model's
    # vocabulary is ``{0: "person"}``.
    "person": "person",
}


def resolve_classes(names: Mapping[int, str] | Sequence[str]) -> dict[int, DetectionClass]:
    """Map raw vocabulary indices to the requested detection classes.

    Args:
        names: The model's ``names`` mapping (COCO index -> label) or a plain
            sequence of labels.

    Returns:
        ``{raw_index: detection_class}`` covering every requested class that the
        vocabulary provides. A class with no matching raw index is simply
        absent — a room without chairs is not a configuration error.

    Raises:
        ModelUnavailableError: The vocabulary provides no ``person`` entry, so
            the checkpoint is unusable for this project (thesis plan §10.2 makes
            person detection the mandatory class).
    """
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    # First raw label wins, exactly as the model declares it, rather than
    # letting a later alias overwrite it.
    mapping: dict[int, DetectionClass] = {}
    for index, label in items:
        resolved = RAW_LABEL_ALIASES.get(str(label).strip().lower())
        if resolved is not None and resolved not in mapping.values():
            mapping[int(index)] = resolved

    if PRIMARY_PERSON_CLASS not in mapping.values():
        raise ModelUnavailableError(
            "detector vocabulary has no 'person' class; "
            "the checkpoint cannot be used for this project"
        )
    return mapping


def class_id_map(
    names: Mapping[int, str] | Sequence[str],
    requested: Iterable[str] | None = None,
) -> dict[int, DetectionClass]:
    """Filter :func:`resolve_classes` down to the operator's requested classes.

    Args:
        names: Model vocabulary, as for :func:`resolve_classes`.
        requested: Class names to keep. ``None`` keeps every resolvable class
            (including ones outside the committed default list, which is useful
            for an exploratory run).

    Returns:
        ``{raw_index: detection_class}`` restricted to the requested classes.

    Raises:
        ModelUnavailableError: The vocabulary cannot resolve ``person``, or a
            requested class is unknown to this module's alias table.
    """
    resolved = resolve_classes(names)
    if requested is None:
        return resolved

    wanted: set[DetectionClass] = set()
    for name in requested:
        normalised = str(name).strip().lower()
        if normalised not in RAW_LABEL_ALIASES:
            raise ModelUnavailableError(
                f"unknown detection class {name!r}; "
                f"known classes are {sorted(set(DEFAULT_DETECTION_CLASSES))}"
            )
        wanted.add(RAW_LABEL_ALIASES[normalised])

    filtered: dict[int, DetectionClass] = {
        index: name for index, name in resolved.items() if name in wanted
    }
    if PRIMARY_PERSON_CLASS not in filtered.values():
        raise ModelUnavailableError(
            "detection.classes must include 'person'; without it the pipeline "
            "cannot report whether people are visible"
        )
    return filtered
