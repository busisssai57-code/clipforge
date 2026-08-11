"""Artifact base model: pinned schema version + deterministic serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ValidationError

from clipforge.paths import atomic_write_bytes


class ArtifactModel(BaseModel):
    """Base for every on-disk artifact.

    Subclasses MUST override ``schema_version`` (and bump it on any field
    change) and set ``stage`` to their producing stage's name.
    """

    model_config = {"extra": "forbid"}

    schema_version: int
    stage: str
    cache_key: str

    def to_json_bytes(self) -> bytes:
        """Deterministic bytes: pydantic dump → sorted-key JSON → LF newline.

        This is the Determinism Law at the serialization boundary — the same
        model instance always produces the same bytes, independent of field
        declaration order or dict insertion order.
        """
        data = self.model_dump(mode="json")
        # allow_nan=False: Python's json emits bare NaN/Infinity tokens by
        # default, which are NOT valid JSON (RFC 8259) — a strict parser in
        # any downstream tool rejects the file, and Python only round-trips
        # them because its own loader is lenient. An artifact that cannot be
        # read by a conforming parser is not a durable artifact. Raising here
        # makes the stage that produced the non-finite value fail loudly
        # instead of writing a landmine to disk.
        text = json.dumps(data, sort_keys=True, ensure_ascii=False,
                          allow_nan=False,
                          separators=(",", ": "), indent=2) + "\n"
        return text.encode("utf-8")

    def write(self, dest: Path) -> Path:
        """Atomic write (temp + fsync + os.replace) — spec §5."""
        return atomic_write_bytes(dest, self.to_json_bytes())

    @classmethod
    def read(cls, path: Path) -> Self:
        """Load + validate. Every corrupt-bytes shape — undecodable UTF-8
        (real disk corruption), non-JSON, JSON-but-not-an-object, version
        mismatch — surfaces as a TYPED StateError so the resume path can
        quarantine + recompute instead of dying on a raw ValueError."""
        from clipforge.errors import StateError

        try:
            raw = json.loads(Path(path).read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateError(f"{cls.__name__} at {path}: unreadable/corrupt "
                             f"bytes: {exc}") from exc
        if not isinstance(raw, dict):
            raise StateError(f"{cls.__name__} at {path}: JSON top level is "
                             f"{type(raw).__name__}, expected object")
        expected = cls.model_fields["schema_version"].default
        found = raw.get("schema_version")
        if expected is not None and found != expected:
            raise StateError(
                f"{cls.__name__} at {path}: schema_version {found} != expected "
                f"{expected}. Re-run the producing stage (its cache key changed).")
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:  # → our typed error (docstring contract)
            raise StateError(
                f"{cls.__name__} at {path}: field validation failed: {exc}") from exc
