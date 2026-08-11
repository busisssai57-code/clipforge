"""MAR-2: the wiring between the config, the model file, and the sampler.

The panel's finding was that a ``mar=None`` mutant survived the entire
gate — every MAR test exercised the pure math while nothing checked that
the math is ever REACHED. The math is now pinned in test_s4_mar.py; this
file pins the plumbing, which is the part that silently unplugs.

These are source-structure assertions (the house pattern from
``test_one_time_base_across_the_whole_filter_graph``). They cannot prove
MediaPipe returns a landmark — that needs real faces, and it is measured
against real footage in the live run, recorded in VERIFICATION.md. What
they DO catch is the failure mode that actually happened: a path that
stops being passed, so the sampler is constructed dead and every shot
quietly falls back to presence with no error anywhere.
"""

from __future__ import annotations

import inspect

from clipforge import cli
from clipforge.config import S4Config
from clipforge.stages import s4_tracking


def test_the_model_path_is_an_input_and_the_filename_is_a_param():
    """Cache-key discipline: the FILENAME identifies the model in params,
    the absolute PATH travels as an input. Hashing the path would make
    every cache key machine-specific."""
    src = inspect.getsource(cli)
    assert '"face_model": cfg.s4.face_model' in src, (
        "the model filename must hash into the S4 cache key, or swapping "
        "landmarker models silently reuses the old camera path")
    assert "face_model_path=face_model_path" in src, (
        "the absolute model path must be passed as an INPUT, not a param")


def test_s4_builds_the_sampler_from_that_input():
    src = inspect.getsource(s4_tracking.S4Tracking._execute)
    assert 'kwargs.get("face_model_path")' in src, (
        "S4 no longer reads face_model_path; the sampler will be "
        "constructed dead and MAR silently disappears")
    assert "_FaceMarSampler(" in src
    # The construction must USE the input, not a hardcoded None. Asserting
    # merely that the NAME appears near the constructor is not enough: the
    # sweep proved `_FaceMarSampler(None if face_model_path else None)`
    # passes that check while building a dead sampler. Require the actual
    # Path(...) construction. (Eighth accidental-pass in this project.)
    ctor = src[src.index("_FaceMarSampler("):]
    assert "Path(face_model_path)" in ctor[:200], (
        "_FaceMarSampler is not being built from face_model_path; MAR "
        "will be dead and every shot falls back to presence silently")


def test_the_head_anchor_reaches_the_sampler():
    """MAR-3's fix is a kwarg at the call site; dropping it silently
    restores the full-width crop that credited a neighbour's face."""
    src = inspect.getsource(s4_tracking.S4Tracking._execute)
    call = src[src.index(".mar("):]
    assert "head_x" in call[:200], (
        "the head-keypoint anchor is not passed to .mar(); the face crop "
        "reverts to full box width")


def test_a_missing_model_degrades_loudly_not_silently():
    """MAR is an enhancement, so a missing model must not raise — but it
    must not be invisible either, or a broken install looks like footage
    where nobody happens to be talking."""
    sampler = s4_tracking._FaceMarSampler(None)
    assert sampler.mar(None, (0, 0, 10, 10)) is None
    src = inspect.getsource(s4_tracking._FaceMarSampler.__init__)
    assert "s4.mar_unavailable" in src, (
        "a dead sampler must log; otherwise every shot falls back to "
        "presence with nothing in the record saying why")


def test_the_configured_filename_matches_the_shipped_model():
    assert S4Config().face_model == "face_landmarker.task"
