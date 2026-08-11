"""Versioned artifact models (DAG Law: stages exchange ONLY these).

Every artifact:
  * is a pydantic model with a pinned ``schema_version`` int,
  * serializes deterministically via :meth:`ArtifactModel.to_json_bytes`
    (sorted keys, fixed separators) so identical content ⇒ identical bytes,
  * records the ``cache_key`` that produced it, closing the loop between the
    artifact registry (state DB) and the file on disk.

Bump ``schema_version`` on ANY field change — stage cache keys include the
stage version, so consumers never see a mixed-version artifact silently.
"""

from clipforge.schemas.base import ArtifactModel
from clipforge.schemas.transcript import (DiarizationTurn, TranscriptArtifact,
                                          TranscriptSegment, Word)
from clipforge.schemas.candidates import CandidatesArtifact, CandidateWindow
from clipforge.schemas.ranking import RankedArtifact, RankedItem
from clipforge.schemas.campath import CamPathArtifact, CropFrame
from clipforge.schemas.clip import ClipArtifact
from clipforge.schemas.editor import EditorArtifact
from clipforge.schemas.poster import PostJob, PostResult

__all__ = [
    "ArtifactModel",
    "Word", "TranscriptSegment", "DiarizationTurn", "TranscriptArtifact",
    "CandidateWindow", "CandidatesArtifact",
    "RankedItem", "RankedArtifact",
    "CropFrame", "CamPathArtifact",
    "ClipArtifact",
    "EditorArtifact",
    "PostJob", "PostResult",
]

