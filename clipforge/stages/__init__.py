"""Pipeline stages S1–S6. Each stage subclasses :class:`clipforge.stages.base.Stage`."""

from clipforge.stages.base import Stage
from clipforge.stages.s1_transcribe import S1Transcribe
from clipforge.stages.s2_prefilter import S2Prefilter
from clipforge.stages.s3_semantic import S3SemanticRanker
from clipforge.stages.s3_5_editor import S3_5_EditorAgent
from clipforge.stages.s4_tracking import S4Tracking

__all__ = [
    "Stage",
    "S1Transcribe",
    "S2Prefilter",
    "S3SemanticRanker",
    "S3_5_EditorAgent",
    "S4Tracking",
]
