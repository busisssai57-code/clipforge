"""T1 virtual windows: prev-tail + chunk, absolute-time math, dedup."""

from pathlib import Path

from clipforge.errors import FfmpegError
from clipforge.ffmpeg import MediaInfo
from clipforge.ingest.overlap import (build_virtual_window,
                                      dedup_by_absolute_time)


def info(duration: float) -> MediaInfo:
    return MediaInfo(duration_s=duration, width=1920, height=1080, fps=30.0,
                     fps_rational="30/1", v_codec="h264", a_codec="aac")


def make_prober(durations: dict[str, float]):
    def prober(path: Path) -> MediaInfo:
        return info(durations.get(Path(path).name, 900.0))
    return prober


def make_runner(outputs: dict[str, int] | None = None):
    """Fake ffmpeg: creates whatever output file the command names."""
    calls: list[list[str]] = []

    def runner(cmd, **kw):
        calls.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x47" * 1024)
        return None

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_first_chunk_has_no_overlap(tmp_path: Path):
    chunk = tmp_path / "chunk_00000.ts"
    chunk.write_bytes(b"x")
    w = build_virtual_window(chunk=chunk, chunk_abs_start_s=0.0,
                             prev_chunk=None, overlap_s=60.0,
                             out_dir=tmp_path / "win",
                             prober=make_prober({}), runner=make_runner())
    assert w.path == chunk  # no needless copy
    assert w.overlap_s == 0.0
    assert w.abs_start_s == 0.0


def test_window_prepends_previous_tail(tmp_path: Path):
    """The real T1 case: chunk 1 is processed as tail(chunk0) + chunk1, and
    its abs_start_s moves BACK by the real overlap so absolute stream time
    stays exact."""
    prev = tmp_path / "chunk_00000.ts"
    chunk = tmp_path / "chunk_00001.ts"
    for p in (prev, chunk):
        p.write_bytes(b"\x47" * 2048)
    # Keyframe quantization: asked 60 s, actually got 63.4 s.
    prober = make_prober({"chunk_00000.ts": 900.0, "chunk_00001.ts": 900.0,
                          "chunk_00001.tail.ts": 63.4,
                          "chunk_00001.window.ts": 963.4})
    runner = make_runner()

    w = build_virtual_window(chunk=chunk, chunk_abs_start_s=900.0,
                             prev_chunk=prev, overlap_s=60.0,
                             out_dir=tmp_path / "win",
                             prober=prober, runner=runner)

    assert w.path.name == "chunk_00001.window.ts"
    assert w.overlap_s == 63.4, "must report the REAL overlap, not the request"
    assert w.abs_start_s == 900.0 - 63.4
    assert w.duration_s == 963.4
    # Cut used -ss at (900 - 60); concat used the demuxer with stream copy.
    cut_cmd = runner.calls[0]  # type: ignore[attr-defined]
    assert "-ss" in cut_cmd and "840.000" in cut_cmd
    concat_cmd = runner.calls[1]  # type: ignore[attr-defined]
    assert "concat" in concat_cmd and "copy" in concat_cmd


def test_concat_list_uses_forward_slashes(tmp_path: Path):
    """ffmpeg's concat parser treats backslashes as escapes on Windows."""
    prev = tmp_path / "chunk_00000.ts"
    chunk = tmp_path / "chunk_00001.ts"
    for p in (prev, chunk):
        p.write_bytes(b"\x47" * 2048)
    seen: dict[str, str] = {}

    def runner(cmd, **kw):
        if "concat" in cmd:
            list_file = Path(cmd[cmd.index("-i") + 1])
            seen["content"] = list_file.read_text(encoding="utf-8")
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x47" * 1024)

    build_virtual_window(chunk=chunk, chunk_abs_start_s=900.0, prev_chunk=prev,
                         overlap_s=60.0, out_dir=tmp_path / "win",
                         prober=make_prober({}), runner=runner)
    assert "\\" not in seen["content"]
    assert seen["content"].count("file '") == 2


def test_broken_previous_chunk_degrades_to_bare_chunk(tmp_path: Path):
    """A failed tail cut must not block THIS chunk from being processed."""
    prev = tmp_path / "chunk_00000.ts"
    chunk = tmp_path / "chunk_00001.ts"
    for p in (prev, chunk):
        p.write_bytes(b"\x47" * 2048)

    def runner(cmd, **kw):
        raise FfmpegError("Invalid data found when processing input")

    w = build_virtual_window(chunk=chunk, chunk_abs_start_s=900.0,
                             prev_chunk=prev, overlap_s=60.0,
                             out_dir=tmp_path / "win",
                             prober=make_prober({}), runner=runner)
    assert w.path == chunk and w.overlap_s == 0.0
    assert w.abs_start_s == 900.0


def test_dedup_by_absolute_time():
    """The same moment seen from two adjacent windows collapses to one."""
    cands = [
        (930.0, 975.0),   # from window N
        (931.5, 976.5),   # same moment, window N+1 (overlap region)
        (1200.0, 1245.0),  # distinct
    ]
    kept = dedup_by_absolute_time(cands, iou_threshold=0.5)
    assert kept == [(930.0, 975.0), (1200.0, 1245.0)]


def test_dedup_keeps_adjacent_nonoverlapping():
    cands = [(0.0, 45.0), (45.0, 90.0)]
    assert dedup_by_absolute_time(cands) == cands


def test_dedup_is_order_independent():
    a = dedup_by_absolute_time([(0.0, 45.0), (1.0, 46.0), (100.0, 145.0)])
    b = dedup_by_absolute_time([(100.0, 145.0), (1.0, 46.0), (0.0, 45.0)])
    assert a == b, "Determinism Law: sorted iteration, stable result"
