"""Operator-authored camera moves — the director camera.

S4 decides where the camera looks by tracking whoever is speaking. That is
the right default and it is wrong often enough to need an override: the
subject the tracker picked is not always the shot, and there is no way to
say "push in here, hold, then drift left" from the outside.

This turns a handful of KEYFRAMES into the same per-frame
:class:`CropFrame` list S4 emits, so the renderer needs no new concept —
it already follows a path frame by frame via sendcmd. A director camera is
therefore not a new rendering mode; it is a different author for an
existing artifact.

Everything here is a pure function of its inputs. Camera geometry is
exactly the kind of thing that looks right in a preview and is wrong by
two pixels in the render, and this project's §S4 rules — even coordinates,
9:16, inside the frame — are testable arithmetic rather than something to
eyeball.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from clipforge.log import get_logger
from clipforge.schemas.campath import CropFrame

log = get_logger(__name__)

#: Easing between two keyframes. "hold" is not decoration: a locked-off
#: shot is a real directorial choice, and interpolating through it would
#: make every cut a slow drift.
EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out", "hold")

#: The output aspect. Crops are forced to it because the renderer scales
#: the crop to 1080x1920 — a crop of any other shape would be squeezed.
TARGET_W, TARGET_H = 9, 16

#: Smallest crop we will author, as a fraction of source height. Past this
#: the upscale to 1920 tall is visibly soft; S4's own punch-in caps at 0.25
#: depth for the same reason.
MIN_CROP_FRAC = 0.25


class CamPathError(ValueError):
    """An authored path that cannot be rendered, with the reason."""


@dataclass(frozen=True)
class KeyFrame:
    """One authored camera position, in SOURCE pixel coordinates.

    Time is clip-relative seconds — the same clock the editor's playhead
    and the transcript use, so a keyframe dropped at the playhead means
    what it looks like it means.
    """

    t: float
    #: Centre of the crop, not its corner. Authoring from the centre is
    #: what makes a push-in a single changing number instead of four that
    #: have to stay consistent with each other.
    cx: float
    cy: float
    #: Crop height in source pixels; width follows from the 9:16 rule.
    h: float
    easing: str = "ease_in_out"

    def as_dict(self) -> dict:
        return {"t": self.t, "cx": self.cx, "cy": self.cy, "h": self.h,
                "easing": self.easing}


def _ease(name: str, u: float) -> float:
    """Map 0..1 to 0..1. Unknown names fall back to linear, loudly."""
    u = max(0.0, min(1.0, u))
    if name == "linear":
        return u
    if name == "hold":
        return 0.0
    if name == "ease_in":
        return u * u
    if name == "ease_out":
        return 1.0 - (1.0 - u) ** 2
    if name == "ease_in_out":
        return 2 * u * u if u < 0.5 else 1.0 - ((-2 * u + 2) ** 2) / 2
    log.warning("campath.unknown_easing", easing=name, note="using linear")
    return u


def _even(value: float) -> int:
    """Round to an EVEN integer. §S4: yuv420p needs even crop origins and
    sizes, and an odd number here is an ffmpeg error at render time."""
    return int(round(value / 2.0)) * 2


def parse_keyframes(raw: Iterable[dict]) -> list[KeyFrame]:
    """Validate operator-supplied keyframes, sorted by time."""
    out: list[KeyFrame] = []
    for i, item in enumerate(raw):
        try:
            kf = KeyFrame(
                t=float(item["t"]), cx=float(item["cx"]),
                cy=float(item["cy"]), h=float(item["h"]),
                easing=str(item.get("easing", "ease_in_out")))
        except (KeyError, TypeError, ValueError) as exc:
            raise CamPathError(
                f"keyframe {i} is missing or malformed: {exc}") from exc
        if kf.t < 0:
            raise CamPathError(f"keyframe {i} is at a negative time")
        if kf.h <= 0:
            raise CamPathError(f"keyframe {i} has a non-positive height")
        if kf.easing not in EASINGS:
            raise CamPathError(
                f"keyframe {i} has easing {kf.easing!r}; "
                f"known: {', '.join(EASINGS)}")
        out.append(kf)
    if not out:
        raise CamPathError("a camera path needs at least one keyframe")
    out.sort(key=lambda k: k.t)
    # Two keyframes at the same instant is an ambiguous instruction, not a
    # jump cut — the renderer would silently take whichever sorted last.
    for a, b in zip(out, out[1:]):
        if abs(a.t - b.t) < 1e-6:
            raise CamPathError(
                f"two keyframes share t={a.t:.3f}s; move one, or delete it")
    return out


def sample_at(keys: Sequence[KeyFrame], t: float) -> tuple[float, float, float]:
    """Camera (cx, cy, h) at time ``t``. Clamped outside the key range.

    Before the first key and after the last, the camera HOLDS rather than
    extrapolating — extrapolating a push-in past its last keyframe is how
    you get a crop bigger than the source at the tail of a clip.
    """
    if t <= keys[0].t:
        return keys[0].cx, keys[0].cy, keys[0].h
    if t >= keys[-1].t:
        return keys[-1].cx, keys[-1].cy, keys[-1].h
    for a, b in zip(keys, keys[1:]):
        if a.t <= t <= b.t:
            span = b.t - a.t
            u = 0.0 if span <= 0 else (t - a.t) / span
            # The easing on the SOURCE key governs the move that leaves it.
            e = _ease(a.easing, u)
            return (a.cx + (b.cx - a.cx) * e,
                    a.cy + (b.cy - a.cy) * e,
                    a.h + (b.h - a.h) * e)
    return keys[-1].cx, keys[-1].cy, keys[-1].h


def crop_at(cx: float, cy: float, h: float, *, src_width: int,
            src_height: int) -> tuple[int, int, int, int]:
    """One clamped, even, 9:16 crop rectangle from a camera position.

    Order matters and is the whole function: size is clamped FIRST (a crop
    taller than the source can never be made to fit by moving it), then
    made even, then the origin is clamped so the rectangle stays inside
    the frame. Clamping the origin before the size lets a too-large crop
    hang off the right edge, which ffmpeg rejects.
    """
    # Never exceed the source, in either axis, at 9:16.
    max_h = min(float(src_height), src_width * TARGET_H / TARGET_W)
    min_h = max(2.0, MIN_CROP_FRAC * src_height)
    h = max(min_h, min(float(h), max_h))
    ch = _even(h)
    cw = _even(ch * TARGET_W / TARGET_H)
    # Rounding both to even can push width past the source by a pixel.
    while cw > src_width and ch >= 4:
        ch -= 2
        cw = _even(ch * TARGET_W / TARGET_H)
    x = _even(cx - cw / 2.0)
    y = _even(cy - ch / 2.0)
    x = max(0, min(x, _even(src_width - cw)))
    y = max(0, min(y, _even(src_height - ch)))
    return x, y, cw, ch


def build_frames(keys: Sequence[KeyFrame], *, duration_s: float,
                 fps_rational: str, src_width: int,
                 src_height: int) -> list[CropFrame]:
    """The per-frame path S6 renders, from the authored keyframes.

    Frame count comes from the EXACT rational fps, not a float: at
    30000/1001 a float multiply drifts by a frame over a minute, and the
    campath is indexed by frame number, so a drift here desyncs the camera
    from the picture.
    """
    if duration_s <= 0:
        raise CamPathError("duration must be positive")
    fps = Fraction(fps_rational)
    if fps <= 0:
        raise CamPathError(f"invalid fps {fps_rational!r}")
    total = int(Fraction(int(round(duration_s * 1000)), 1000) * fps)
    if total <= 0:
        raise CamPathError("that duration is shorter than one frame")

    frames: list[CropFrame] = []
    for i in range(total):
        t = float(Fraction(i, 1) / fps)
        cx, cy, h = sample_at(keys, t)
        x, y, w, hh = crop_at(cx, cy, h, src_width=src_width,
                              src_height=src_height)
        frames.append(CropFrame(frame=i, x=x, y=y, w=w, h=hh))
    log.info("campath.authored", keyframes=len(keys), frames=len(frames),
             fps=fps_rational, src=f"{src_width}x{src_height}")
    return frames


def default_keyframes(*, src_width: int, src_height: int,
                      duration_s: float) -> list[KeyFrame]:
    """A sane starting path: centred, full height, locked off.

    The editor opens on this when a clip has no authored path, so the
    first thing an operator drags is a real camera rather than an empty
    canvas they have to understand before touching.
    """
    h = min(float(src_height), src_width * TARGET_H / TARGET_W)
    return [KeyFrame(t=0.0, cx=src_width / 2.0, cy=src_height / 2.0, h=h,
                     easing="ease_in_out"),
            KeyFrame(t=max(0.1, duration_s), cx=src_width / 2.0,
                     cy=src_height / 2.0, h=h, easing="hold")]


def load_keyframes(path: Path | str) -> list[KeyFrame]:
    """Read an authored path file written by the dashboard or by hand."""
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CamPathError(f"no camera path at {path}") from exc
    except ValueError as exc:
        raise CamPathError(f"camera path is not valid JSON: {exc}") from exc
    keys = blob.get("keyframes") if isinstance(blob, dict) else blob
    if not isinstance(keys, list):
        raise CamPathError(
            "camera path must be a list of keyframes, or an object with a "
            "'keyframes' list")
    return parse_keyframes(keys)


def digest(keys: Sequence[KeyFrame]) -> str:
    """Stable digest of an authored path.

    Feeds S6's params so that changing the camera changes the cache key.
    Without it, a re-render with a new path would resolve to the cached
    clip and the operator would watch their edit do nothing — the same
    shape as the trim that rendered a full-length duplicate.
    """
    import hashlib

    payload = json.dumps([k.as_dict() for k in keys], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
