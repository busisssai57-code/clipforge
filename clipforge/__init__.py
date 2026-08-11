"""ClipForge — local-first autonomous stream→viral-clip pipeline.

Architecture invariants (build spec §3) enforced across this package:

  1. VRAM Law     — one model class resident at a time (clipforge.gpu).
  2. Determinism  — same input bytes + config = same output bytes.
  3. Resumability — every stage is content-addressed (clipforge.stages.base).
  4. DAG Law      — stages communicate only via versioned on-disk artifacts.
  5. Authorization— operator-supplied channels only; produce files and stop.
"""

__version__ = "0.1.0"
