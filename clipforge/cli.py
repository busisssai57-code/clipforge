"""Typer CLI — entrypoints: watch, process, grab, agent, verify, doctor."""

from __future__ import annotations

import importlib
import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from typer.models import ArgumentInfo, OptionInfo

from clipforge.config import load_config
from clipforge.dllpaths import ensure_nvidia_dll_dirs
from clipforge.errors import ClipForgeError
from clipforge.log import setup_logging
from clipforge.paths import Workspace, discard_partials

app = typer.Typer(name="bta", no_args_is_help=True,
                  help="BTA — Beyond The Average. Local-first autonomous "
                       "stream-to-clip pipeline. Nothing leaves this machine.",
                  pretty_exceptions_show_locals=False)
console = Console()

CONFIG_OPT = typer.Option(Path("config/config.toml"), "--config", "-c",
                          help="Path to config.toml")


def _cli_value(value: Any, fallback: Any) -> Any:
    """Real value, or ``fallback`` when a Typer sentinel leaked through.

    Typer command functions are also plain Python functions, and this
    package calls them that way (`grab` and `agent` both invoke
    `process`). Arguments the caller omits then keep their
    ``typer.models.OptionInfo`` / ``ArgumentInfo`` default instead of the
    parsed value — and those objects are TRUTHY, so every `x is not None`
    and `if x:` check downstream reads them as a deliberate choice.

    This normalises them back to the intended default. The alternative
    (never calling a command function directly) is the cleaner rule, but
    it cannot be enforced by the type system, and the failure is silent.
    """
    return fallback if isinstance(value, (OptionInfo, ArgumentInfo)) else value


def _pacing_enabled(jumpcut: Any, config_default: bool) -> bool:
    """Whether jump-cut silence removal runs for this invocation.

    A function, not an inline expression, so the decision can be tested
    behaviourally. The inline form was pinned by a source-string
    assertion, which a mutant defeated by leaving the string in a
    comment — the ninth accidental pass in this project.
    """
    jumpcut = _cli_value(jumpcut, None)
    return bool(config_default) if jumpcut is None else bool(jumpcut)


def _boot(config_path: Path, *, sweep_partials: bool = True) -> tuple:
    """Shared startup: DLL paths → config → workspace → logging → debris sweep.

    ``sweep_partials=False`` defers the sweep to the caller: ``watch`` must
    not delete ``*.partial`` files until it OWNS the workspace, because a
    second instance's sweep would otherwise destroy the running instance's
    in-flight remux and T1 temp files.
    """
    ensure_nvidia_dll_dirs()  # T6: before anything can import ctranslate2
    # Hugging Face's xet transport STALLS on this machine: S3 sat at
    # "Fetching 5 files: 0%" with the process alive, 18 seconds of CPU
    # and a cache that never grew, which reads as a slow stage rather
    # than a hung one — there is no timeout on it. The classic HTTP
    # path fetched the same 7 GB immediately. Set before any
    # huggingface_hub import, and only when the operator has not made
    # their own choice.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cfg = load_config(config_path)
    ws = Workspace(cfg.workspace.root).ensure()
    setup_logging(ws.logs)
    # Machine-wide GPU serialization. Every entry point boots through here
    # — the CLI, the subprocesses the dashboard spawns, the swarm — so
    # pointing the lock at the workspace is what actually makes "one GPU
    # stage at a time" hold ACROSS processes. The in-process semaphores
    # only ever covered threads within one interpreter.
    from clipforge.gpu import configure_process_gpu_lock
    configure_process_gpu_lock(ws.root / "gpu.lock")
    if sweep_partials:
        removed = discard_partials(ws.root)
        if removed:
            console.print(f"[yellow]swept {len(removed)} crash-partial file(s)[/]")
    return cfg, ws


@app.command()
def doctor(config: Path = CONFIG_OPT) -> None:
    """Diagnose every external prerequisite (ffmpeg, CUDA, tokens, disk...)."""
    from clipforge import preflight

    # Doctor must run even with a broken/missing config — it diagnoses that too.
    try:
        cfg = load_config(config)
        root, floor = Path(cfg.workspace.root), cfg.disk.free_floor_gb
    except ClipForgeError as exc:
        console.print(f"[yellow]config not loadable ({exc}); using defaults[/]")
        root, floor = Path("workspace"), 50.0
    report, ok = preflight.render(preflight.run_all(root, disk_floor_gb=floor))
    console.print(report)
    raise typer.Exit(0 if ok else 1)


@app.command()
def verify(module: str = typer.Argument(
        ...,
        # The list is what `clipforge/verify/` actually holds. It read
        # "... | compositing | orchestration" for as long as those
        # modules were planned; both exit 2 as unknown modules, and a
        # help string that names a gate nobody can run is the same
        # false green this gate exists to catch.
        help="all | skeleton | ingestion | ai")) -> None:
    """Run a module's deterministic self-checks (spec section 8, gate #1)."""
    try:
        mod = importlib.import_module(f"clipforge.verify.{module}")
    except ImportError as exc:
        console.print(f"[red]unknown verify module {module!r}: {exc}[/]")
        raise typer.Exit(2)
    rc = mod.main()
    raise typer.Exit(rc)


@app.command()
def watch(config: Path = CONFIG_OPT,
          channels: Path = typer.Option(Path("config/channels.toml"),
                                        "--channels",
                                        help="Path to channels.toml")) -> None:
    """Watch configured channels: record live streams, download new VODs.

    CP1 scope: ingestion only - chunks and VODs land in the workspace and
    state DB. DAG processing (S1-S6) attaches to this loop at CP5.
    """
    from clipforge.config import load_watchlist
    from clipforge.ingest.retention import workspace_lock

    # Defer the crash-debris sweep until we hold the lock: another running
    # instance's in-flight remux/window temps are also *.partial files.
    cfg, ws = _boot(config, sweep_partials=False)
    wl = load_watchlist(channels)

    # ONE writer per workspace. Reconciliation below decides a session is
    # dead purely from `ended_at IS NULL`, so a second instance starting
    # while the first is recording would "recover" (and close) a LIVE
    # session. The OS drops this lock even on a hard kill, so it cannot go
    # stale and block a legitimate restart.
    with workspace_lock(ws) as acquired:
        if not acquired:
            console.print("[red]another clipforge is already using this "
                          f"workspace ({ws.root}). Stop it first.[/]")
            raise typer.Exit(3)
        _watch_locked(cfg, ws, wl, channels, config)


def _watch_locked(cfg, ws, wl, channels_path: Path,
                  config_path: Path) -> None:
    """The body of `watch`, run while holding the workspace lock."""
    import asyncio
    import signal

    from clipforge.ingest.monitor import ChannelMonitor
    from clipforge.ingest.retention import reconcile_sessions
    from clipforge.state import StateDB

    # Safe now that the workspace is exclusively ours.
    removed = discard_partials(ws.root)
    if removed:
        console.print(f"[yellow]swept {len(removed)} crash-partial file(s)[/]")

    from clipforge.dispatch import ClipDispatcher

    db = StateDB(ws.state_db)

    # The DAG seam. `on_media` defaulted to None and nothing ever passed
    # one, so `watch` recorded, chunked and windowed forever without
    # producing a single clip — the largest gap between what this tool
    # claimed and what it did.
    #
    # It goes through a queue rather than being called inline because
    # ingestion must never wait for the GPU: a live stream is not
    # replayable, and a four-minute render on the monitor's thread means
    # four minutes of stream lost for good.
    def _clip_window(path: Path, abs_start_s: float) -> None:
        # jumpcut=None explicitly: passing nothing would hand `process` a
        # truthy Typer sentinel and force pacing on, overriding [pacing].
        process(input_path=path, config=config_path,
                abs_offset=abs_start_s, clips=cfg.orchestration.clips_per_window,
                jumpcut=None)

    dispatcher = ClipDispatcher(
        handler=_clip_window,
        maxsize=cfg.orchestration.queue_maxsize).start()
    monitor = ChannelMonitor(cfg=cfg, db=db, ws=ws, watchlist=wl,
                             on_media=dispatcher.submit)

    # Ingestion's analogue of CP0's .partial sweep: sessions a crash (or a
    # hard kill whose orphaned children kept writing) left open are rescanned
    # and their media recovered onto the correct absolute timeline. Recovered
    # media goes through the same T1 window path as live segments.
    recovered = reconcile_sessions(
        db, ws, on_segment=lambda p, t: monitor.emit_recovered(p, t),
        segment_time_s=cfg.ingest.segment_time_s)
    if recovered:
        console.print(f"[yellow]recovered {recovered} segment(s) from an "
                      "interrupted session[/]")

    # The same treatment for JOBS. A status is only ever moved by the
    # process running it, so a kill leaves one 'running' for ever - and a
    # stuck job is indistinguishable from a busy one, so the queue looks
    # occupied by work nobody is doing.
    stale = db.reap_stale_jobs()
    if stale:
        console.print(f"[yellow]{len(stale)} job(s) were left running by an "
                      "earlier crash; marked failed[/]")

    async def _serve() -> None:
        # Install the handler INSIDE the loop: waiting for asyncio.run() to
        # raise KeyboardInterrupt is too late — an in-flight download would
        # already be blocking an uncancellable worker thread. Setting the
        # stop event at signal time makes every worker wind down promptly.
        loop = asyncio.get_running_loop()
        state = {"signals": 0}

        def _on_signal(*_args: object) -> None:
            state["signals"] += 1
            if state["signals"] == 1:
                console.print("[yellow]stopping - finalizing tails "
                              "(Ctrl+C again to force)[/]")
                monitor.stop()
            else:
                # Escape hatch: a shutdown wedged on an unresponsive
                # subprocess must still be abortable. os._exit skips
                # interpreter cleanup deliberately — the chunker's job
                # object reaps the children even on an abrupt exit.
                console.print("[red]forced exit[/]")
                os._exit(130)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_signal)
            except (NotImplementedError, AttributeError, ValueError):
                # Windows: add_signal_handler is unsupported; the plain
                # signal module still delivers SIGINT to the main thread.
                try:
                    signal.signal(sig, _on_signal)
                except (OSError, ValueError):
                    pass
        await monitor.run()

    console.print(f"[green]watching {len(wl.enabled_sorted())} channel(s); "
                  "Ctrl+C to stop (chunker finalizes tails on exit)[/]")
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:  # backstop if a signal slipped past the handler
        monitor.stop()
    finally:
        # Abandon the clip backlog rather than draining it: Ctrl-C should
        # return the terminal, and every queued window is still a file on
        # disk that `bta process` can pick up.
        dispatcher.stop()
        stats = dispatcher.stats.snapshot()
        if stats["submitted"] or stats["dropped"]:
            console.print(
                f"[yellow]clips: {stats['processed']} rendered, "
                f"{stats['failed']} failed, {stats['dropped']} not "
                f"queued/abandoned[/]")
        console.print("[yellow]stopped - tails finalized, sessions closed[/]")
        db.close()


@app.command()
def process(input_path: Path = typer.Argument(..., help="A local video file to clip"),
            config: Path = CONFIG_OPT,
            abs_offset: float = typer.Option(0.0, "--abs-offset",
                                             help="Absolute stream time of this file's t=0"),
            clips: int = typer.Option(1, "--clips", "-n", min=1,
                                      help="How many top-ranked candidates "
                                           "to render as clips"),
            jumpcut: bool = typer.Option(
                None, "--jumpcut/--no-jumpcut",
                help="Cut silences between words for fast-paced pacing "
                     "(default from [pacing] config)"),
            enhance: str = typer.Option(
                None, "--enhance",
                help="Speech cleanup: off | gentle | strong. Default from "
                     "[s6] config. Leave off for musical sources — "
                     "denoise and de-essing damage a beat"),
            niche: str = typer.Option(
                None, "--niche",
                help="Apply a niche's caption style, pacing and grade "
                     "(see: bta swarm niches)"),
            broll: bool = typer.Option(
                False, "--broll/--no-broll",
                help="Generate B-roll locally from the transcript and cut "
                     "it over the pauses between sentences"),
            campath_file: Path = typer.Option(
                None, "--campath-file",
                help="Operator-authored camera keyframes (JSON). Overrides "
                     "S4's tracked path for this render."),
            cut_file: Path = typer.Option(
                None, "--cut-file",
                help="JSON [[start, end], ...] of WINDOW-relative seconds to "
                     "remove — what the editor's marked words and pauses "
                     "become. Applies with or without --jumpcut"),
            manifest: Path = typer.Option(
                None, "--manifest",
                help="Write a machine-readable JSON result here. Automation "
                     "should read this instead of parsing console output")) -> None:
    """Run the clip DAG on one local file (no ingestion)."""
    import os

    # `grab` and `agent` call this as a plain Python function, where the
    # unpassed options keep their Typer *sentinel* objects rather than the
    # values Typer would have parsed. OptionInfo is TRUTHY (measured), so
    # the old is-not-None ternary read the sentinel as a deliberate choice
    # and silently force-enabled silence removal on exactly the two
    # unattended paths — while [pacing] is off by default precisely
    # because altering rhythm is editorial, not corrective. The decision
    # now lives in _pacing_enabled so it can be tested behaviourally.
    jumpcut = _cli_value(jumpcut, None)
    abs_offset = _cli_value(abs_offset, 0.0)
    clips = _cli_value(clips, 1)
    enhance = _cli_value(enhance, None)
    niche_name = _cli_value(niche, None)
    manifest = _cli_value(manifest, None)
    broll = bool(_cli_value(broll, False))
    campath_file = _cli_value(campath_file, None)
    authored_keys = None
    if campath_file is not None:
        from clipforge.campath_edit import CamPathError, load_keyframes
        try:
            authored_keys = load_keyframes(campath_file)
        except CamPathError as exc:
            console.print(f"[red]--campath-file is not usable:[/] {exc}")
            raise typer.Exit(2) from exc
        console.print(f"  director camera: {len(authored_keys)} keyframe(s) "
                      f"from {Path(campath_file).name}")

    cut_file = _cli_value(cut_file, None)

    # Read once, here, so a malformed cut file fails before any GPU work
    # rather than after S1-S4 have run.
    cut_spans: list[tuple[float, float]] = []
    if cut_file is not None:
        import json as _json

        try:
            raw = _json.loads(Path(cut_file).read_text(encoding="utf-8"))
            cut_spans = [(float(a), float(b)) for a, b in raw
                         if float(b) > float(a)]
        except (OSError, ValueError, TypeError) as exc:
            console.print(f"[red]--cut-file is not usable:[/] {exc}")
            raise typer.Exit(2) from exc
        if not cut_spans:
            console.print("[red]--cut-file contained no usable spans[/]")
            raise typer.Exit(2)

    from clipforge.config import Secrets
    from clipforge.stages.base import digest_file
    from clipforge.stages.s1_transcribe import S1Transcribe
    from clipforge.stages.s2_prefilter import S2Prefilter
    from clipforge.state import StateDB

    # sweep_partials=False: this command does not hold the workspace lock,
    # and a workspace-wide *.partial sweep would destroy a concurrently
    # running `watch`'s in-flight remux and window temp files.
    cfg, ws = _boot(config, sweep_partials=False)
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[red]no such file: {input_path}[/]")
        raise typer.Exit(1)

    db = StateDB(ws.state_db)
    # One job row per (source, offset), and every stage below records its
    # attempt against it. Stage.run has taken a job_id since CP0 and this
    # command never passed one, so the jobs and stage_runs tables were
    # written only by tests — which is why the control API had nothing to
    # report even once its queries were correct. Idempotent by key: the
    # same source re-processed resumes the same job.
    source_digest = digest_file(input_path)
    job_id = db.upsert_job(
        "clip", key=f"process:{source_digest}:{abs_offset:.3f}",
        payload={"source": str(input_path), "abs_offset_s": abs_offset,
                 "clips_requested": clips})
    db.set_job_status(job_id, "running")
    try:
        console.print(f"[green]S1: transcribing {input_path.name} "
                      "(first run downloads models)[/]")
        s1 = S1Transcribe(
            db=db, artifacts_dir=ws.artifacts,
            hf_token=Secrets().hf_token or os.environ.get("HF_TOKEN"))
        s1_params = {
            "model": cfg.s1.model, "compute_type": cfg.s1.compute_type,
            "batch_size": cfg.s1.batch_size, "language": cfg.s1.language,
            "abs_offset_s": abs_offset,
        }
        transcript = s1.run(input_digest=source_digest, job_id=job_id,
                            params=s1_params, media_path=input_path)
        n_words = sum(len(s.words) for s in transcript.segments)
        console.print(f"  {len(transcript.segments)} segments, {n_words} words, "
                      f"language={transcript.language}, "
                      f"diarization={'ok' if transcript.diarization_ok else 'UNAVAILABLE'}")

        # Probed ONCE: the repair loop needs the source duration to know
        # whether a window can legally grow, and _keeps_for needs the exact
        # fps rational. Re-probing per clip cost a subprocess per attempt.
        from clipforge.ffmpeg import probe as _probe_source
        from clipforge.repair import MAX_ATTEMPTS, plan_repair
        source_info = _probe_source(input_path)

        console.print("[green]S2: scoring candidate windows[/]")
        s2 = S2Prefilter(db, ws.artifacts)
        s2_params = {
            "window_min_s": cfg.s2.window_min_s,
            "window_max_s": cfg.s2.window_max_s,
            "top_k": cfg.s2.top_k, "nms_iou": cfg.s2.nms_iou,
        }
        cands = s2.run(input_digest=transcript.cache_key, job_id=job_id,
                       params=s2_params,
                       transcript=transcript)
        console.print("[green]S3: ranking candidate windows (multimodal VL)[/]")
        from clipforge.stages.s3_semantic import S3SemanticRanker
        from clipforge.stages.s3_5_editor import S3_5_EditorAgent
        from clipforge.stages.s4_tracking import S4Tracking

        s3 = S3SemanticRanker(db, ws.artifacts)
        s3_params = {
            "use_cloud": cfg.s3.use_cloud,
            "cloud_model": cfg.s3.cloud_model,
            "cloud_models": list(cfg.s3.cloud_models),
            "cloud_timeout_s": cfg.s3.cloud_timeout_s,
            "cloud_max_attempts": cfg.s3.cloud_max_attempts,
            "model_id": cfg.s3.model_id,
            "fallback_model_id": cfg.s3.fallback_model_id,
            "frames_per_candidate": cfg.s3.frames_per_candidate,
            "max_pixels": cfg.s3.max_pixels,
            "seed": cfg.s3.seed,
        }
        ranked = s3.run(input_digest=cands.cache_key, job_id=job_id,
                        params=s3_params,
                        candidates_artifact=cands, video_path=input_path)
        console.print(f"  {len(ranked.items)} candidates ranked (source={ranked.ranking_source})")

        from clipforge.stages.s5_subtitles import S5Subtitles
        from clipforge.stages.s6_render import S6Render
        from clipforge.stages.s7_qa import S7QualityGate

        s3_5 = S3_5_EditorAgent(db, ws.artifacts)
        s4 = S4Tracking(db, ws.artifacts)
        s5 = S5Subtitles(db, ws.artifacts)
        s6 = S6Render(db, ws.artifacts)
        s7 = S7QualityGate(db, ws.artifacts)
        # Model files live under workspace/models — never the CWD, where
        # ultralytics would otherwise download a stray 40 MB .pt into
        # whatever directory the operator happened to run from. The FILENAME
        # stays in params (model identity hashes into the cache key); the
        # absolute path rides as an input so a moved workspace does not
        # invalidate every campath ever computed.
        pose_model_abs = ws.root / "models" / cfg.s4.pose_model
        s4_params = {
            "pose_model": cfg.s4.pose_model,
            "asd_confidence_tau": cfg.s4.asd_confidence_tau,
            "deadzone_frac": cfg.s4.deadzone_frac,
            "max_pan_px_per_frame": cfg.s4.max_pan_px_per_frame,
            "one_euro_min_cutoff": cfg.s4.one_euro_min_cutoff,
            "one_euro_beta": cfg.s4.one_euro_beta,
            "punch_scale": cfg.s4.punch_scale,
            "punch_hold_s": cfg.s4.punch_hold_s,
            "face_model": cfg.s4.face_model,
            "mar_activity_tau": cfg.s4.mar_activity_tau,
        }
        face_model_path = str(ws.root / "models" / cfg.s4.face_model)
        s5_params = {
            "font": cfg.s5.font, "font_size": cfg.s5.font_size,
            "highlight_color": cfg.s5.highlight_color,
            "base_color": cfg.s5.base_color,
            "outline": cfg.s5.outline, "shadow": cfg.s5.shadow,
            "margin_v": cfg.s5.margin_v,
            "max_words_per_line": cfg.s5.max_words_per_line,
            "max_lines": cfg.s5.max_lines,
            "animation": cfg.s5.animation,
            "width": cfg.s6.width, "height": cfg.s6.height,
        }

        # A niche carries the WHOLE look, so its caption styling, pacing and
        # speech settings have to reach the stages that render them. They
        # were defined and never plumbed: `niche_s5_params` had zero call
        # sites, so a dark_mindset clip came out with the default 96pt
        # uppercase karaoke — the exact look that niche exists to avoid.
        # These ride in params, so they hash into cache keys like every
        # other setting and two niches cannot share one cached artifact.
        active_niche = None
        if niche_name:
            from clipforge.niches import get_niche, niche_s5_params
            try:
                active_niche = get_niche(niche_name)
            except ValueError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(2)
            s5_params.update(niche_s5_params(active_niche))
            console.print(f"[green]niche:[/] {active_niche.label} — "
                          f"{active_niche.caption.animation} captions"
                          + (", graded" if active_niche.grade else ""))
        s6_params = {
            "width": cfg.s6.width, "height": cfg.s6.height,
            "encoder": cfg.s6.encoder, "nvenc_preset": cfg.s6.nvenc_preset,
            "x264_preset": cfg.s6.x264_preset,
            "cq": cfg.s6.cq, "audio_bitrate": cfg.s6.audio_bitrate,
            "loudness_i": cfg.s6.loudness_i, "loudness_tp": cfg.s6.loudness_tp,
            "loudness_lra": cfg.s6.loudness_lra,
            # Precedence: explicit flag > niche > config. A niche is a
            # considered default, not an override of what the operator
            # typed on this specific run.
            "enhance_speech": (enhance
                               or (active_niche.enhance_speech
                                   if active_niche else None)
                               or cfg.s6.enhance_speech),
            # The niche grade joins S6's OWN filter chain rather than being
            # re-encoded afterwards: a second pass costs a generation of
            # quality and, worse, lands after S7's gate so the file that
            # actually ships was never measured.
            "grade": (active_niche.grade if active_niche else ""),
        }

        def _snap_edge(t: float, radius: float = 0.35) -> float:
            """Nudge a cut point into the nearest audio silence trough.

            Blueprint §4.2: word timestamps put an edge BETWEEN words, but
            alignment error of even 60 ms can clip a word's attack or
            release. A silencedetect pass over ±radius finds actual quiet;
            the edge moves to the nearest silence midpoint, or stays put if
            the speaker never pauses — the edit only ever moves TOWARD
            absolute quiet, never into speech.
            """
            import re as _re
            import subprocess as _sp

            from clipforge.ffmpeg import require_binary

            span_start = max(0.0, t - radius)
            try:
                proc = _sp.run(
                    [str(require_binary("ffmpeg")), "-nostdin",
                     "-hide_banner",
                     "-ss", f"{span_start:.3f}", "-t", f"{2 * radius:.3f}",
                     "-i", str(input_path), "-vn",
                     "-af", "silencedetect=noise=-35dB:d=0.05",
                     "-f", "null", "-"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=60)
            except _sp.TimeoutExpired:
                return t          # snapping is a refinement, never a gate
            err = proc.stderr or ""
            starts = [float(m) for m in
                      _re.findall(r"silence_start: ([\d.]+)", err)]
            ends = [float(m) for m in
                    _re.findall(r"silence_end: ([\d.]+)", err)]
            candidates = []
            for i, s in enumerate(starts):
                e = ends[i] if i < len(ends) else 2 * radius
                candidates.append(span_start + (s + e) / 2.0)
            if not candidates:
                return t
            snapped = min(candidates, key=lambda c: abs(c - t))
            return max(t - radius, min(t + radius, snapped))

        rendered = 0
        #: Paths of clips that PASSED QA. Automation reads these from the
        #: manifest instead of regexing them out of console output — rich
        #: wraps long paths at 80 columns when stdout is piped, which
        #: silently defeated the scrape the swarm relied on.
        shipped_clips: list[str] = []
        for pos, item in enumerate(ranked.items[:max(1, clips)], start=1):
            idx = item.candidate_index
            if not (0 <= idx < len(cands.candidates)):
                continue
            cand = cands.candidates[idx]
            win_start = _snap_edge(cand.start - abs_offset)
            win_end = _snap_edge(cand.end - abs_offset)
            cand_id = f"cand_{idx:03d}"

            def _keeps_for(w_start: float, w_end: float,
                           enabled: bool) -> list[list[float]]:
                """Jump-cut keep-intervals (blueprint §9.2), window-relative.

                A function rather than a straight-line block because the
                repair loop re-derives them for a moved window: keeps that
                still described the ORIGINAL window would splice the wrong
                seconds out of the retry.
                """
                # Explicit cuts from the editor are independent of pacing:
                # marking a word for removal must work whether or not
                # jump-cutting is on, so `enabled` no longer short-circuits
                # when there are spans to remove.
                if not enabled and not cut_spans:
                    return []
                from clipforge.pacing import compute_keep_intervals
                spans = sorted(
                    (float(w.start) - abs_offset - w_start,
                     float(w.end) - abs_offset - w_start)
                    for seg in transcript.segments for w in seg.words
                    if w.start is not None and w.end is not None
                    and float(w.end) > w_start + abs_offset
                    and float(w.start) < w_end + abs_offset)
                if enabled:
                    intervals = compute_keep_intervals(
                        spans, w_end - w_start,
                        gap_threshold=cfg.pacing.gap_threshold_s,
                        pad=cfg.pacing.pad_s)
                else:
                    # No pacing: start from the whole window and remove only
                    # what was explicitly marked.
                    intervals = [(0.0, w_end - w_start)]
                if cut_spans:
                    from clipforge.pacing import subtract_spans

                    # Editor spans are WINDOW-relative, the same clock
                    # `compute_keep_intervals` works in and the same one
                    # `clipmeta.transcript_for` reports, so no rebasing is
                    # needed here — and rebasing "just in case" is how the
                    # campath/word time-base mismatch was introduced once.
                    before = sum(b - a for a, b in intervals)
                    intervals = subtract_spans(intervals, cut_spans)
                    removed = before - sum(b - a for a, b in intervals)
                    console.print(f"  editor cuts: {len(cut_spans)} span(s), "
                                  f"{removed:.2f}s removed")
                # EXACT frame quantization via Fraction, then the duration
                # floor re-checked on the quantized result. The first version
                # of this block rounded through decimal seconds — the panel
                # measured it making every seam metric 6-60x worse at
                # 30000/1001 fps (decimal seconds are not the pts grid), and
                # separately measured the pre-quantization floor being
                # violated by post-quantization rounding.
                from clipforge.pacing import (enforce_floor_on_frames,
                                              keeps_seconds_from_frames,
                                              quantize_keeps_to_frames)
                from fractions import Fraction as _Fr
                fps_rat = source_info.fps_rational
                frame_pairs = quantize_keeps_to_frames(intervals, fps_rat)
                window_frames = int(round((w_end - w_start) * _Fr(fps_rat)))
                frame_pairs = enforce_floor_on_frames(
                    frame_pairs, window_frames, fps_rat)
                out = [[a, b] for a, b in
                       keeps_seconds_from_frames(frame_pairs, fps_rat)]
                cut_s = (w_end - w_start) - sum(b - a for a, b in out)
                if cut_s > 0.05:
                    console.print(f"  jump-cuts: {len(out)} segments, "
                                  f"{cut_s:.1f}s of silence removed")
                return out

            pacing_on = _pacing_enabled(jumpcut, cfg.pacing.enabled)
            keeps = _keeps_for(win_start, win_end, pacing_on)
            console.print(f"[green]clip {pos}: window "
                          f"{win_start:.1f}-{win_end:.1f}s ({cand_id})[/]")

            # candidate_id and the window ride in PARAMS, not only in inputs:
            # params feed the cache key, inputs do not. Passed only as inputs,
            # every candidate of one ranking shared one cache key, so clip 2
            # silently resolved to clip 1's cached artifacts.
            # start_s/end_s in ABSOLUTE time (the editor compares against
            # transcript word times). Left unset, the editor's kwargs
            # defaulted to [0, 30) and every clip got a title summarizing
            # the video's first half-minute — three clips, one title.
            def _attempt(w_start: float, w_end: float,
                         w_keeps: list[list[float]],
                         overrides: dict[str, float]):
                """One full S3.5->S7 pass over a window. Returns (clip, qa).

                NOTE the asymmetry in the loudness params: S6 renders with
                the (possibly repaired) target, but S7 always judges
                against the ORIGINAL one. A repair that nudges the target
                to +3 LU is compensating for a source that lands quiet —
                if the gate moved with it, the repair would grade its own
                homework and a clip could "pass" 3 LU off spec.
                """
                editor_art = s3_5.run(
                    input_digest=ranked.cache_key, job_id=job_id,
                    params={"style_profile": cfg.editor.style_profile,
                            "candidate_id": cand_id},
                    ranked_artifact=ranked,
                    transcript_artifact=transcript,
                    candidate_id=cand_id,
                    start_s=w_start + abs_offset,
                    end_s=w_end + abs_offset,
                )
                console.print(f"  Title: {editor_art.title} "
                              f"(hook_score={editor_art.hook_score:.2f})")

                campath = s4.run(
                    input_digest=ranked.cache_key, job_id=job_id,
                    params={**s4_params, "window_start": round(w_start, 3),
                            "window_end": round(w_end, 3)},
                    ranked_artifact=ranked,
                    transcript_artifact=transcript,
                    video_path=input_path,
                    start_s=w_start,
                    end_s=w_end,
                    face_model_path=face_model_path,
                    pose_model_path=(str(pose_model_abs)
                                     if pose_model_abs.exists() else None),
                )
                # The director camera REPLACES the tracked path. S4 still
                # runs: it establishes the geometry (source dims, exact
                # fps) the authored keyframes are interpolated against,
                # and re-deriving that here would be a second source of
                # truth for the one number the camera is indexed by.
                cam_digest = None
                if authored_keys is not None:
                    from clipforge.campath_edit import build_frames, digest
                    cam_digest = digest(authored_keys)
                    campath = campath.model_copy(update={"frames": build_frames(
                        authored_keys,
                        duration_s=w_end - w_start,
                        fps_rational=campath.src_fps_rational,
                        src_width=campath.src_width,
                        src_height=campath.src_height)})
                    console.print(
                        f"  [bold]director camera:[/] {len(campath.frames)} "
                        f"frames authored, replacing the tracked path")

                if authored_keys is None and campath.framing_mode == "center":
                    # The centre crop is the documented DEGRADED mode; a
                    # neutral status line here is how fake tracking went
                    # unnoticed for two checkpoints.
                    console.print(
                        f"  [yellow]Camera path: {len(campath.frames)} "
                        "frames (mode=center — tracking DEGRADED, see "
                        "logs for the recorded reason)[/]")
                else:
                    console.print(
                        f"  Camera path: {len(campath.frames)} frames "
                        f"(mode={campath.framing_mode}, "
                        f"mar_shots={campath.mar_shots})")

                # hook_text rides in params (it is per-clip DATA that changes
                # the output bytes, so it must be in the cache key);
                # viral_fast style implies uppercase karaoke.
                subs = s5.run(input_digest=campath.cache_key, job_id=job_id,
                              params={**s5_params,
                                      "hook_text": editor_art.hook_text,
                                      "keep_intervals": w_keeps,
                                      "uppercase": cfg.editor.style_profile
                                      == "viral_fast"},
                              transcript_artifact=transcript,
                              campath_artifact=campath)
                console.print(f"  {subs.line_count} subtitle events, "
                              f"{subs.word_count} words")

                clip = s6.run(input_digest=subs.cache_key, job_id=job_id,
                              params={**s6_params, **overrides,
                                      "keep_intervals": w_keeps,
                                      **({"campath_digest": cam_digest}
                                         if cam_digest else {})},
                              campath_artifact=campath, subtitle_artifact=subs,
                              video_path=input_path, clips_dir=ws.clips)
                console.print(
                    f"  [bold]{clip.clip_path}[/] "
                    f"({clip.duration_s:.1f}s, {clip.width}x{clip.height}, "
                    f"{clip.encoder}, {clip.loudness_i:.1f} LUFS)")

                # S7: the mechanical quality gate. Every check remeasured
                # from the file, none trusted from S6's own report.
                qa = s7.run(input_digest=clip.cache_key, job_id=job_id,
                            params=s6_params,
                            clip_artifact=clip, subtitle_artifact=subs,
                            campath_artifact=campath)
                # editor_art rides back out: the export pack needs the
                # title and hook, and they are written in here.
                return clip, qa, editor_art

            # Bounded self-repair: a rejection with a mechanical cause is
            # re-rendered with a reasoned edit rather than abandoned. A
            # rejection with a STRUCTURAL cause is never retried — see
            # clipforge/repair.py for why that distinction is the whole
            # point. A clip that ends up failing is QUARANTINED, not
            # deleted: an operator can inspect what the gate saw, and
            # nothing downstream picks it up from clips/.
            cur_start, cur_end, cur_keeps = win_start, win_end, keeps
            cur_pacing = pacing_on
            overrides: dict[str, float] = {}
            attempt = 0
            repairs: list[str] = []
            while True:
                clip, qa, editor_art = _attempt(cur_start, cur_end,
                                                cur_keeps, overrides)
                if qa.passed:
                    break
                failed = [c for c in qa.checks
                          if c.severity == "fail" and not c.passed]
                plan = plan_repair(
                    failed, window_start=cur_start, window_end=cur_end,
                    source_duration=float(source_info.duration_s),
                    attempt=attempt, target_i=float(cfg.s6.loudness_i),
                    target_tp=float(cfg.s6.loudness_tp),
                    original_start=win_start, original_end=win_end)
                if not plan.repairable:
                    if attempt:
                        console.print(f"  [yellow]repair stopped[/]: "
                                      f"{plan.reason}")
                    break
                console.print(f"  [yellow]QA rejected - repairing "
                              f"(attempt {attempt + 1}/{MAX_ATTEMPTS})[/]")
                for r in plan.remedies:
                    console.print(f"    {r.action}: {r.detail}")
                    repairs.append(r.action)
                cur_start, cur_end = plan.window_start, plan.window_end
                if plan.drop_jumpcuts:
                    cur_pacing = False
                cur_keeps = _keeps_for(cur_start, cur_end, cur_pacing)
                overrides.update(plan.param_overrides)
                attempt += 1

            if qa.passed:
                rendered += 1
                shipped_clips.append(str(Path(clip.clip_path).resolve()))
                # Everything needed to post, written beside the clip. This
                # is what exists INSTEAD of auto-publishing: the payload is
                # assembled here where the transcript and title are still
                # in hand, and a person decides whether it goes out.
                # Reporting must never sink a clip that already passed QA.
                try:
                    from clipforge.export_pack import build_pack

                    words = " ".join(
                        w.text for s in transcript.segments for w in s.words
                        if w.start is not None
                        and cur_start + abs_offset <= float(w.start)
                        < cur_end + abs_offset)
                    build_pack(
                        Path(clip.clip_path), title=editor_art.title,
                        transcript_text=words, hook=editor_art.hook_text,
                        segments=[s for s in transcript.segments
                                  if cur_start + abs_offset <= float(s.start or 0)
                                  < cur_end + abs_offset],
                        niche_keywords=(list(active_niche.keywords)
                                        if active_niche else None))
                    console.print("  export pack written "
                                  "(caption, tags, thumbnail, chapters)")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]export pack skipped: "
                                  f"{type(exc).__name__}: {exc}[/]")

                # B-roll runs AFTER QA, on a clip already proven good, and
                # returns the original untouched if anything goes wrong.
                # A garnish must never be able to lose the dish.
                if broll:
                    try:
                        from clipforge.broll import plan_broll, render_broll
                        from clipforge.genvideo import build_router

                        window_segs = [
                            s for s in transcript.segments
                            if cur_start + abs_offset <= float(s.start or 0)
                            < cur_end + abs_offset]
                        cues = plan_broll(
                            window_segs, clip_duration_s=float(clip.duration_s),
                            style=(active_niche.gen_style if active_niche
                                   else ""))
                        if not cues:
                            console.print("  [dim]b-roll: no pause long "
                                          "enough to cut away into[/]")
                        else:
                            console.print(f"  b-roll: {len(cues)} insert(s) "
                                          f"— generating locally")
                            router = build_router(
                                cfg, ws,
                                needs=(set(active_niche.keywords)
                                       if active_niche else set()))
                            final = render_broll(
                                Path(clip.clip_path), cues, router=router,
                                width=cfg.s6.width, height=cfg.s6.height,
                                work_dir=ws.tmp / f"broll_{cand_id}")
                            if str(final) != clip.clip_path:
                                console.print(f"  [bold]{final}[/]")
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"  [yellow]b-roll skipped: "
                                      f"{type(exc).__name__}: {exc}[/]")
                note = (f", {qa.warned_count} warning(s) recorded"
                        if qa.warned_count else "")
                fixed = (f" after {len(repairs)} repair(s): "
                         f"{', '.join(repairs)}" if repairs else "")
                console.print(f"  QA: [green]PASSED[/] "
                              f"({len(qa.checks)} checks{note}){fixed}")
            else:
                rejected_dir = ws.clips / "rejected"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                dest = rejected_dir / Path(clip.clip_path).name
                try:
                    Path(clip.clip_path).replace(dest)
                except OSError:
                    dest = Path(clip.clip_path)  # quarantine failed: report in place
                bad = [c for c in qa.checks
                       if c.severity == "fail" and not c.passed]
                console.print(f"  QA: [red]REJECTED[/] -> {dest}")
                for c in bad:
                    console.print(f"    [red]{c.name}[/]: {c.measured} "
                                  f"(expected {c.expected})")

        # "done" means the run completed, not that every clip shipped:
        # a run whose clips were all quarantined finished correctly and
        # said so. Conflating the two would make the dashboard report a
        # crash whenever QA did its job.
        db.set_job_status(job_id, "done" if rendered else "empty")
        _write_manifest(manifest, kind="process", outputs=shipped_clips,
                        extra={"job_id": job_id, "niche": niche_name,
                               "requested": clips, "passed": rendered})
        console.print(f"[green]DONE - {rendered} clip(s) passed QA in "
                      f"{ws.clips}[/]")
        try:
            from clipforge.report import build_dashboard
            dest = build_dashboard(ws)
            console.print(f"[green]mission control: {dest}[/]")
        except Exception as exc:  # dashboard is reporting, never a gate
            console.print(f"[yellow]dashboard generation failed: {exc}[/]")
    except BaseException:
        # A crashed run must not be left reading "running" forever — that
        # is indistinguishable from a live job in the control API. Catches
        # BaseException so a Ctrl-C is recorded too, and re-raises.
        try:
            db.set_job_status(job_id, "failed")
        except Exception:  # noqa: BLE001 - bookkeeping never masks the cause
            pass
        raise
    finally:
        db.close()



@app.command()
def generate(
        brief: str = typer.Argument(
            None, help="What the piece is about. Optional when --script "
                       "points at a screenplay file."),
        config: Path = CONFIG_OPT,
        preset: str = typer.Option(None, "--preset", "-p",
                                   help="documentary | storytelling | "
                                        "explainer | motion_graphics"),
        shots: int = typer.Option(None, "--shots", "-s", min=1, max=64),
        screenplay: bool = typer.Option(
            False, "--screenplay",
            help="Read the brief as a Fountain screenplay: one shot per "
                 "SCENE, and dialogue routed to the voice instead of into "
                 "the picture prompt."),
        script: Path = typer.Option(
            None, "--script",
            help="Read the screenplay from a .fountain file instead of the "
                 "argument. Implies --screenplay."),
        hook: str = typer.Option(
            None, "--hook",
            help="Hook card burned over the opening seconds, e.g. "
                 "\"INTAAN MIDKEE KAA QOSLIYAY\"."),
        handle: str = typer.Option(
            None, "--handle",
            help="Your own handle, stamped bottom-centre on every frame. "
                 "Defaults to [genvideo] handle."),
        clip_it: bool = typer.Option(
            False, "--clip/--no-clip",
            help="Run the generated piece through the clip DAG"),
        aspect: str = typer.Option(
            None, "--aspect",
            help="9:16, 16:9, 3:4, 4:5 or 1:1. Overrides the niche's own "
                 "declared frame."),
        model: str = typer.Option(
            None, "--model",
            help="Force a generation model by key (see: bta models). "
                 "Without it the registry picks by the niche's needs, and "
                 "unverified models are never picked."),
        manifest: Path = typer.Option(
            None, "--manifest",
            help="Write a machine-readable JSON result here. Automation "
                 "should read this instead of parsing console output")) -> None:
    """Generate a video from a text brief, then optionally clip it.

    Shots are generated individually and concatenated: every current
    text-to-video model loses coherence past a few seconds, so length comes
    from CUTTING, which is also how the job is actually done.

    Providers fail over automatically — the metered cloud model first (only
    if you enabled it), the local open-source model when its quota is gone,
    and back again once the window resets.
    """
    from clipforge.config import Secrets
    from clipforge.genvideo import ProviderError, build_router
    from clipforge.niches import resolve_preset

    preset_name = _cli_value(preset, None)
    shot_count = _cli_value(shots, None)
    script = _cli_value(script, None)
    hook = _cli_value(hook, None)
    handle = _cli_value(handle, None)
    screenplay = bool(_cli_value(screenplay, False))
    clip_it = bool(_cli_value(clip_it, False))
    aspect_ratio = _cli_value(aspect, None)
    manifest = _cli_value(manifest, None)

    brief = _cli_value(brief, None)
    if script is None and not (brief or "").strip():
        console.print("[red]give a brief, or --script pointing at a "
                      "screenplay file[/]")
        raise typer.Exit(2)
    if script is not None:
        script_path = Path(script)
        if not script_path.is_file():
            console.print(f"[red]no screenplay at[/] {script_path}")
            raise typer.Exit(2)
        # utf-8 explicitly: a Somali script is Latin-1-safe but an emoji
        # punchline mark is not, and Windows would otherwise decode this
        # file as cp1252 and lose the marks that drive the post layer.
        brief = script_path.read_text(encoding="utf-8")
        if not brief.strip():
            console.print(f"[red]{script_path} is empty[/]")
            raise typer.Exit(2)
        screenplay = True

    if screenplay and hook is None:
        # The hook belongs to the piece, so a script carries its own in
        # the title page (`Hook: ...`). Read for any screenplay, not just
        # one that arrived as a file: the dashboard sends the same script
        # as text, and a hook that only worked from disk would be a
        # feature of the file path rather than of the format. --hook
        # still wins - that is the operator answering directly.
        from clipforge.screenplay import title_page

        hook = title_page(brief).get("hook") or None

    cfg, ws = _boot(config, sweep_partials=False)
    if not cfg.genvideo.enabled:
        console.print("[red]generation is disabled. Set [genvideo] "
                      "enabled = true in config.toml.[/]")
        raise typer.Exit(2)

    # A niche name (dark_mindset, viral_clips, ...) resolves here too, not
    # just the four base creative presets — a niche's --preset value was
    # rejected before this, which broke both the dashboard's Generate
    # button and the swarm's Generator role the moment either used one.
    try:
        chosen = resolve_preset(preset_name or cfg.genvideo.preset)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2)

    # §2 chokepoint: the credential is handed out only by clipforge.cloud,
    # and only when [genvideo] use_cloud authorizes it. The old direct
    # Secrets()/os.environ read here was one of the leaf checks the
    # chokepoint replaced.
    from clipforge.cloud import gemini_key

    key = gemini_key(cfg, feature="genvideo")
    # The niche's keywords are what selection scores models against, so a
    # photoreal format routes to the stronger model and an atmospheric one
    # to the faster. Without this the registry could never take effect.
    try:
        from clipforge.niches import get_niche
        _needs = set(get_niche(chosen.name).keywords)
    except Exception:  # noqa: BLE001 - a base preset has no niche
        _needs = set(getattr(chosen, "keywords", ()) or ())
    # A niche DECLARES its frame, and until now nothing read it: the
    # config value won and a 3:4 format rendered 9:16 with only the
    # dashboard label saying otherwise. Explicit --aspect still wins over
    # both - it is the operator answering the question directly.
    from clipforge.niches import resolve_aspect

    aspect_ratio = resolve_aspect(aspect_ratio, chosen.name,
                                  cfg.genvideo.aspect_ratio)
    # `prefer` had been reachable from nowhere: `build_router` took it,
    # `select_model` implemented it — including the one path that lets an
    # UNVERIFIED model run, which is how a model gets verified — and no
    # caller in the product supplied it. `swarm plan --model` carried a
    # key through two task payloads to a subprocess call that never
    # passed it on. The registry's own note said "select it explicitly
    # with --model ltx25"; there was no such flag.
    router = build_router(cfg, ws, api_key=key, needs=_needs,
                          aspect_ratio=aspect_ratio,
                          prefer=_cli_value(model, None))

    if cfg.genvideo.use_cloud:
        console.print("[yellow]cloud generation is ON: prompts will be sent "
                      "to Google. Everything else in BTA stays local.[/]")
    for row in router.status():
        state = ("ready" if row["configured"] and row["quota_ok"]
                 else "not configured" if not row["configured"]
                 else f"metered out for {row['available_in_s']:.0f}s")
        console.print(f"  provider {row['name']:6s}: {state}")

    # A screenplay's folder is named for its TITLE, not for the first 48
    # characters of the file - which, with a title page at the top, is
    # "title-geel-suuqa-tegey-credit-bta-draft-date-202". The dashboard
    # lists these directories by name, so the name is the piece's label.
    label = brief
    if screenplay:
        from clipforge.screenplay import title_page, to_beats

        page = title_page(brief)
        first = (to_beats(brief, 1) or [""])[0]
        label = page.get("title") or first or brief
    out_dir = ws.root / "generated" / _slug(label)
    console.print(f"[green]generating {shot_count or chosen.default_shots} "
                  f"shot(s) - {chosen.name}: {chosen.summary}[/]")
    result = router.generate_sequence(
        brief=brief, preset=chosen, out_dir=out_dir, shots=shot_count,
        aspect_ratio=aspect_ratio, screenplay=screenplay)

    if not result.paths:
        console.print("[red]no shots were generated[/]")
        for shot in result.shots:
            console.print(f"  [red]shot {shot.index}[/]: {shot.error}")
        raise typer.Exit(1)

    for shot in result.shots:
        mark = "[green]ok[/]" if shot.ok else "[red]FAILED[/]"
        console.print(f"  shot {shot.index}: {mark} "
                      f"({shot.provider}) {shot.error}")
    if result.degraded:
        console.print("[yellow]this piece is not uniform: shots came from "
                      f"{', '.join(result.providers_used) or 'no'} provider(s)"
                      " or a shot is missing[/]")

    stitched = out_dir / "sequence.mp4"
    try:
        _concat_shots(result.paths, stitched)
    except ClipForgeError as exc:
        console.print(f"[red]assembly failed: {exc}[/]")
        raise typer.Exit(1)

    stitched = _apply_post_layer(
        stitched, result=result, niche_name=chosen.name, hook=hook,
        handle=handle if handle is not None else cfg.genvideo.handle)
    console.print(f"[bold green]{stitched}[/] "
                  f"({result.ok_count} shot(s), "
                  f"~{result.ok_count * chosen.shot_seconds:.0f}s)")
    _write_manifest(manifest, kind="generate",
                    outputs=[str(stitched.resolve())],
                    extra={"shots": result.ok_count,
                           "preset": chosen.name,
                           "providers": list(result.providers_used),
                           "degraded": result.degraded,
                           "shot_files": [str(p.resolve())
                                          for p in result.paths]})

    try:
        from clipforge.report import build_dashboard
        build_dashboard(ws)
    except Exception as exc:  # noqa: BLE001 - reporting is never a gate
        console.print(f"[yellow]dashboard refresh failed: {exc}[/]")

    if clip_it:
        # The generated piece is just a video file, so it goes through the
        # SAME DAG as anything else — no special path, no second pipeline.
        console.print("[green]running the generated piece through the clip "
                      "DAG[/]")
        process(input_path=stitched, config=config, abs_offset=0.0,
                clips=1, jumpcut=None)


@app.command()
def genquota(config: Path = CONFIG_OPT,
             reset: bool = typer.Option(
                 False, "--reset",
                 help="Clear recorded exhaustion so the premium provider is "
                      "tried again immediately"),
             provider: str = typer.Option(
                 None, "--provider",
                 help="Only reset this provider (default: all)")) -> None:
    """Show — or clear — recorded generation quota state.

    Needed because not every 429 is a window that reopens on its own. A
    rate limit resets; a "check your plan and billing details" refusal does
    not, and the ledger cannot tell them apart from the response alone. It
    therefore records the conservative default window, and this is the
    lever for when you KNOW the situation changed (billing enabled, plan
    upgraded, new key) and do not want to wait the day out.
    """
    import json as _json

    reset = bool(_cli_value(reset, False))
    provider = _cli_value(provider, None)
    _cfg, ws = _boot(config, sweep_partials=False)
    path = ws.root / "genvideo_quota.json"
    if not path.exists():
        console.print("no generation has run yet (no quota ledger)")
        return

    from clipforge.genvideo.quota import QuotaLedger

    ledger = QuotaLedger.load(path)
    if reset:
        targets = ([provider] if provider
                   else sorted(ledger.providers.keys()))
        for name in targets:
            st = ledger.state(name)
            st.exhausted_until = 0.0
            st.consecutive_errors = 0
            st.last_reason = ""
        ledger.save()
        console.print(f"[green]cleared: {', '.join(targets) or 'nothing'}[/]")

    for name, row in ledger.snapshot().items():
        if row["available"]:
            state = "[green]ready[/]"
        else:
            state = (f"[yellow]metered out for "
                     f"{float(row['available_in_s']) / 60:.0f} min[/]")
        console.print(f"  {name:8s} {state}  "
                      f"{row['calls']} call(s), "
                      f"{row['seconds_generated']:.0f}s generated")
        if row["last_reason"]:
            console.print(f"           [dim]{str(row['last_reason'])[:160]}[/]")
    _json  # noqa: B018 - imported for future --json output


def _write_manifest(dest: Path | None, *, kind: str, outputs: list[str],
                    extra: dict | None = None) -> None:
    """Write a machine-readable result file, atomically.

    This exists because automation was reading output paths by regexing
    them out of the console. That is unreliable in two ways that both bit
    in production: rich wraps long paths at 80 columns whenever stdout is
    a pipe rather than a terminal, splitting a path across lines so no
    regex can match it; and any path containing a space never matched at
    all. Both failures look identical to "the render produced nothing",
    which is exactly what the swarm reported after two shots rendered
    fine.

    A no-op when no path was requested, so interactive runs are unchanged.
    """
    if dest is None:
        return
    import json

    payload = {"kind": kind, "outputs": outputs, "count": len(outputs)}
    payload.update(extra or {})
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True),
                           encoding="utf-8")
        partial.replace(dest)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        # Reporting must never sink a run that actually produced clips.
        console.print(f"[yellow]could not write manifest {dest}: {exc}[/]")


def _slug(text: str, limit: int = 48) -> str:
    import re
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (out[:limit] or "piece").rstrip("-")


def _apply_post_layer(stitched: Path, *, result, niche_name: str,
                      hook: str | None, handle: str | None) -> Path:
    """Stamp hook card, punchline stickers and handle — or don't.

    Returns the path to hand on: the stamped file when there was anything
    to stamp, and the untouched sequence otherwise. Never raises. This is
    the last step of a run that may already have spent GPU-hours, and a
    missing emoji font must not turn a finished piece into a failed
    command.
    """
    from clipforge.niches import NICHES

    niche = NICHES.get(niche_name)
    if niche is None or not niche.post_layer:
        return stitched
    ok = [s for s in result.shots if s.ok]
    marks = [list(s.marks) for s in ok]
    if not (hook or handle or any(marks)):
        return stitched

    from clipforge.socialpost import PostError, apply_post, spec_from_shots

    try:
        # REAL durations, probed, not the preset's nominal shot length. A
        # provider that returns 1.87s for a 1.9s request drifts by half a
        # shot over thirty cuts, and the sticker would land on the wrong
        # face.
        from clipforge.ffmpeg import probe

        seconds = []
        for shot in ok:
            try:
                seconds.append(float(probe(shot.path).duration_s))
            except Exception:  # noqa: BLE001 - fall back to the request
                seconds.append(float(shot.seconds))
        spec = spec_from_shots(
            seconds, marks, hook=hook or "", watermark=handle or "",
            hook_seconds=niche.hook_seconds)
        # `sequence.mp4` is the FINISHED piece, always. The dashboard
        # lists `*/sequence.mp4`, so writing the stamped version beside
        # it under another name would have shown the operator the cut
        # without its hook or stickers and called that the output. The
        # un-stamped concat is kept so a different hook can be burned
        # without regenerating a single shot.
        dest = stitched.with_name("post.mp4")
        apply_post(stitched, dest, spec)
        raw = stitched.with_name("sequence.raw.mp4")
        raw.unlink(missing_ok=True)
        stitched.rename(raw)
        dest.rename(stitched)
        dest = stitched
    except (PostError, ClipForgeError) as exc:
        console.print(f"[yellow]post layer skipped: {exc}[/]")
        return stitched
    console.print(f"  [bold]post layer:[/] hook={'yes' if hook else 'no'} "
                  f"stickers={sum(len(m) for m in marks)} "
                  f"handle={handle or 'none'}")
    return dest


def _has_audio(path: Path) -> bool:
    """Whether a file carries an audio stream, asked of ffprobe."""
    import subprocess

    from clipforge.ffmpeg import require_binary

    proc = subprocess.run(
        [str(require_binary("ffprobe")), "-v", "error",
         "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    return bool((proc.stdout or "").strip())


def _with_uniform_audio(paths: list[Path], work_dir: Path) -> list[Path]:
    """Give silent shots a silent TRACK, so a mixed sequence still concats.

    The concat demuxer needs every input to have the same streams. One
    sequence can mix providers — that is what the failover is for — and
    LTX-2.5 generates sound while Wan 2.2 does not, so a piece can arrive
    half with audio and half without. Dropping the audio would be the easy
    fix and the wrong one: the model's own sound is the thing being
    protected here, so the silent shots get silence instead.
    """
    import subprocess

    from clipforge.ffmpeg import require_binary

    flags = [_has_audio(p) for p in paths]
    if not any(flags) or all(flags):
        return paths
    out: list[Path] = []
    for path, has in zip(paths, flags):
        if has:
            out.append(path)
            continue
        padded = work_dir / f"{path.stem}.silent{path.suffix}"
        proc = subprocess.run(
            [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
             "-i", str(path),
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             str(padded)],
            capture_output=True, text=True, timeout=600)
        # A shot that cannot be padded is still a shot: fall back to the
        # original and let the concat drop audio rather than lose a beat.
        out.append(padded if proc.returncode == 0 and padded.is_file()
                   else path)
    console.print(f"  [dim]audio: {sum(1 for f in flags if not f)} of "
                  f"{len(flags)} shot(s) were silent; padded so the "
                  f"generated sound survives the concat[/]")
    return out


def _concat_shots(paths: list[Path], dest: Path) -> None:
    """Concatenate shots into one piece via the ffmpeg concat demuxer.

    Re-encodes rather than stream-copies: shots can come from DIFFERENT
    providers in one sequence (that is the whole point of the failover),
    and a stream copy across mismatched encoder settings produces a file
    that plays for exactly one shot.
    """
    import subprocess

    from clipforge.errors import ClipForgeError as _Err
    from clipforge.ffmpeg import require_binary

    dest.parent.mkdir(parents=True, exist_ok=True)
    # A TEMP dir, not the piece's own folder: the padded copies are
    # scaffolding for one concat, and left beside the shots they double a
    # shot's bytes and look like shots to anything globbing *.mp4 there.
    stack = contextlib.ExitStack()
    with stack:
        work = Path(stack.enter_context(
            tempfile.TemporaryDirectory(prefix="concat-", dir=dest.parent)))
        paths = _with_uniform_audio(paths, work)
        return _concat_listed(paths, dest)


def _concat_listed(paths: list[Path], dest: Path) -> None:
    """The ffmpeg half, once every input has the same streams."""
    import subprocess

    from clipforge.errors import ClipForgeError as _Err
    from clipforge.ffmpeg import require_binary

    listing = dest.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in paths),
        encoding="utf-8")
    partial = dest.with_suffix(".mp4.partial")
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p",
         # Named rather than left to the muxer's default: shots now
         # arrive with sound, and the file the rest of the DAG consumes
         # should not depend on what ffmpeg happens to pick.
         "-c:a", "aac", "-b:a", "192k",
         "-f", "mp4", str(partial)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800)
    if proc.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise _Err(f"concat failed: {(proc.stderr or '')[-400:]}")
    partial.replace(dest)


def _trending_default_limit() -> int:
    """``trending.DEFAULT_LIMIT`` as a Typer default, without duplicating it.

    Typer evaluates option defaults at decoration time, so this runs on
    import; ``clipforge.trending`` is deliberately torch-free (it shells out to
    yt-dlp), so importing it here costs nothing measurable.
    """
    from clipforge.trending import DEFAULT_LIMIT

    return DEFAULT_LIMIT


def _js_runtime() -> str | None:
    """``"node:C:\\...\\node.exe"`` for the first runtime on this machine.

    yt-dlp names the runtime and optionally its path. Passing the resolved
    path matters on Windows, where the interpreter running this process
    does not necessarily share a PATH with an interactive shell.
    """
    import shutil

    for name in ("node", "deno", "bun"):
        found = shutil.which(name)
        if found:
            return f"{name}:{found}"
    return None


@app.command()
def grab(url: str = typer.Argument(..., help="Video URL (yt-dlp supported)"),
         config: Path = CONFIG_OPT,
         clips: int = typer.Option(3, "--clips", "-n", min=1)) -> None:
    """Download a video, then clip it. One command, URL to clips."""
    import subprocess
    import time

    cfg, ws = _boot(config, sweep_partials=False)
    dest_dir = ws.root / "downloads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Captured before the download so the newest-file fallback only ever
    # considers files this run produced, never a stale prior download.
    started_at = time.time()
    console.print(f"[green]downloading {url}[/]")
    # Height cap, not "best": a 4K source triples every decode in the DAG
    # for pixels that 1080x1920 throws away.
    cmd = [sys.executable, "-m", "yt_dlp",
           "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
           "--merge-output-format", "mp4", "--no-playlist",
           "-o", str(dest_dir / "%(title).60s.%(ext)s"),
           "--print", "after_move:filepath"]
    # YouTube extraction needs a JavaScript runtime to solve the player
    # challenge; without one yt-dlp warns and then 403s on the media URL.
    # Only deno is enabled by default, but node is far more likely to be
    # present — point yt-dlp at whichever this machine actually has
    # rather than making the operator discover the flag from a 403.
    runtime = _js_runtime()
    if runtime:
        cmd.extend(["--js-runtimes", runtime])
    else:
        console.print("[yellow]no JavaScript runtime found (node/deno/bun). "
                      "YouTube downloads will likely fail with 403; install "
                      "Node and retry.[/]")
    cmd.append(url)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    if proc.returncode != 0:
        console.print(f"[red]download failed:[/] {(proc.stderr or '')[-400:]}")
        raise typer.Exit(1)
    # yt-dlp prints the final path via --print after_move:filepath, but it is
    # NOT reliably the last stdout line: post-processing notices, JS-runtime
    # messages, and merge summaries can trail it. Take the last line that is an
    # existing file (same rule as ingest.youtube.download_vod), then fall back
    # to the newest file that landed in dest_dir during this run — which is how
    # a real 465 MB download once got misreported as "no file landed".
    path = ""
    for line in reversed((proc.stdout or "").splitlines()):
        cand = line.strip()
        if cand and Path(cand).exists():
            path = cand
            break
    if not path:
        landed = [p for p in dest_dir.glob("*")
                  if p.is_file() and p.stat().st_mtime >= started_at]
        if landed:
            path = str(max(landed, key=lambda p: p.stat().st_mtime))
    if not path or not Path(path).exists():
        console.print("[red]download reported success but no file landed[/]")
        raise typer.Exit(1)
    console.print(f"[green]downloaded:[/] {path}")
    process(input_path=Path(path), config=config, abs_offset=0.0, clips=clips)


@app.command()
def trending(config: Path = CONFIG_OPT,
             query: str = typer.Option(
                 "podcast", "--query", "-q",
                 help="Search seed for trend discovery. The result is the "
                      "most-viewed video THIS WEEK matching it — 'podcast', "
                      "'interview', 'news', a topic, or a creator's name."),
             region: str = typer.Option("US", "--region",
                                        help="Two-letter region to bias toward (gl)."),
             clips: int = typer.Option(3, "--clips", "-n", min=1,
                                       help="Clips to render from the chosen video."),
             pick: int = typer.Option(1, "--pick", min=1,
                                      help="Use the Nth trending candidate (1 = top)."),
             min_minutes: float = typer.Option(
                 3.0, "--min-minutes",
                 help="Skip sources shorter than this (too short to clip)."),
             max_minutes: float = typer.Option(
                 90.0, "--max-minutes",
                 help="Skip sources longer than this (ties up the GPU)."),
             limit: int = typer.Option(
                 _trending_default_limit(), "--limit", min=1,
                 help="Search results to consider before filtering. Higher "
                      "costs nothing extra (one listing call) and survives "
                      "an aggressive duration window."),
             lang: str = typer.Option(
                 None, "--lang",
                 help="Only accept a video in this ISO language (e.g. 'en'). "
                      "Off by default; enabling it probes candidates in order."),
             dry_run: bool = typer.Option(
                 False, "--dry-run",
                 help="Show the trending candidates and the pick, then stop "
                      "before downloading or clipping.")) -> None:
    """Find a currently-trending video and run the full clip pipeline on it.

    One command, zero input: discover what is trending this week, pick the
    most-viewed clip-ready video, download it, and cut it into shorts. This is
    ``grab`` with the URL chosen for you from what is trending right now.
    """
    from clipforge import trending as _trending
    from clipforge.errors import IngestError
    from clipforge.ingest.youtube import download_vod

    cfg, ws = _boot(config, sweep_partials=False)
    console.print(f"[green]discovering trending videos[/] "
                  f"(query={query!r}, region={region})")
    stats: dict = {}
    try:
        cands = _trending.discover(
            query, region=region, limit=limit, stats=stats,
            min_minutes=min_minutes, max_minutes=max_minutes)
    except IngestError as exc:
        console.print(f"[red]trend discovery failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Always show what the duration window did. A thin candidate pool is the
    # single most common reason this command disappoints, and without these
    # numbers the next failure gets blamed on --lang or on "nothing trending".
    console.print(
        f"  {stats.get('raw', len(cands))} result(s) -> "
        f"{stats.get('usable', len(cands))} clip-ready "
        f"(dropped: {stats.get('too_long', 0)} too long, "
        f"{stats.get('too_short', 0)} too short, {stats.get('live', 0)} live)")

    if not cands:
        console.print(
            f"[red]no clip-ready trending videos found[/] for {query!r} "
            f"between {min_minutes:g} and {max_minutes:g} minutes.")
        if stats.get("too_long"):
            console.print(
                f"  {stats['too_long']} of {stats['raw']} were longer than "
                f"{max_minutes:g} min — raise --max-minutes (long-form queries "
                "like 'podcast' or 'interview' need 60-120).")
        else:
            console.print("  Widen --min-minutes/--max-minutes, or try a "
                          "different --query.")
        raise typer.Exit(1)

    # Optional language gate: probe candidates top-down until enough match, so
    # the pick still respects view-count order within the requested language.
    # Capped, because each probe is its own yt-dlp extraction (~2-4 s) and a
    # wide discovery list would otherwise spend minutes here before any work.
    if lang:
        want = lang.strip().lower()
        probe_budget = min(len(cands), max(pick * 4, 12))
        console.print(f"  probing up to {probe_budget} candidate(s) for "
                      f"language={want!r}…")
        matched = []
        for c in cands[:probe_budget]:
            if _trending.probe_language(c.video_id) == want:
                matched.append(c)
                if len(matched) >= pick:
                    break
        if not matched:
            console.print(
                f"[red]none of the top {probe_budget} trending candidates are "
                f"in language {want!r}.[/]")
            # Do not let --lang take the blame for a duration window that had
            # already emptied the pool: with 58 of 59 dropped as too long,
            # there was only ever one video for the language gate to look at.
            if stats.get("too_long", 0) > stats.get("usable", 0):
                console.print(
                    f"  Note: the language gate only had "
                    f"{stats.get('usable', 0)} candidate(s) to choose from — "
                    f"{stats['too_long']} were dropped for being longer than "
                    f"{max_minutes:g} min. Raise --max-minutes first.")
            else:
                console.print(
                    "  Re-run without --lang to take the most-viewed "
                    "regardless of language, or try --region/--query closer "
                    "to that audience.")
            raise typer.Exit(1)
        cands = matched

    console.print(f"  {len(cands)} candidate(s):")
    for i, c in enumerate(cands[:max(pick, 8)], start=1):
        marker = "->" if i == pick else "  "
        views = f"{c.view_count:,}" if c.view_count is not None else "?"
        console.print(f"  {marker} {i}. [{c.duration_hms}] {views} views  "
                      f"{c.title[:60]}")

    if pick > len(cands):
        console.print(f"[red]--pick {pick} but only {len(cands)} candidate(s)[/]")
        raise typer.Exit(1)
    chosen = cands[pick - 1]
    console.print(f"[green]chosen:[/] {chosen.title}  [{chosen.url}]")

    if dry_run:
        console.print("[yellow]--dry-run: stopping before download.[/]")
        return

    dest_dir = ws.root / "downloads"
    console.print(f"[green]downloading {chosen.url}[/]")
    try:
        path = download_vod(chosen.video_id, dest_dir)
    except IngestError as exc:
        console.print(f"[red]download failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]downloaded:[/] {path}")
    process(input_path=path, config=config, abs_offset=0.0, clips=clips)


@app.command()
def agent(config: Path = CONFIG_OPT,
          clips: int = typer.Option(3, "--clips", "-n", min=1),
          poll_s: float = typer.Option(20.0, "--poll",
                                       help="Seconds between inbox scans"),
          once: bool = typer.Option(False, "--once",
                                    help="Drain the inbox and exit")) -> None:
    """Autonomous mode: drop videos in the inbox, collect clips.

    Watches ``workspace/inbox/`` and runs the full DAG on anything that
    lands there, then moves the source to ``inbox/done/``. Failures move to
    ``inbox/failed/`` — never silently retried forever, and never left in
    place where the next scan would pick them up again.
    """
    import time

    cfg, ws = _boot(config, sweep_partials=False)
    inbox = ws.root / "inbox"
    done, failed = inbox / "done", inbox / "failed"
    for d in (inbox, done, failed):
        d.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]BTA agent watching[/] {inbox}")
    console.print("  drop videos in; clips land in workspace/clips")
    console.print("  Ctrl+C to stop\n")

    media = {".mp4", ".mkv", ".mov", ".webm", ".ts", ".m4v"}
    seen_stable: dict[Path, int] = {}
    try:
        while True:
            found = sorted(p for p in inbox.iterdir()
                           if p.is_file() and p.suffix.lower() in media)
            for src in found:
                # Size-stability check before touching it: a file still
                # being copied in is not ready, and half a video renders
                # into a clip nobody can use.
                size = src.stat().st_size
                if seen_stable.get(src) != size:
                    seen_stable[src] = size
                    continue
                console.print(f"[green]▶ {src.name}[/]")
                try:
                    process(input_path=src, config=config, abs_offset=0.0,
                            clips=clips)
                    src.replace(done / src.name)
                except Exception as exc:  # noqa: BLE001 - reported, not hidden
                    console.print(f"[red]failed: {exc}[/]")
                    try:
                        src.replace(failed / src.name)
                    except OSError:
                        pass
                seen_stable.pop(src, None)
            if once:
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        console.print("\n[green]agent stopped[/]")


@app.command()
def dashboard(config: Path = CONFIG_OPT,
              open_browser: bool = typer.Option(
                  True, "--open/--no-open",
                  help="Open the page after generating it")) -> None:
    """Generate workspace/dashboard.html — local mission control."""
    from clipforge.report import build_dashboard

    cfg, ws = _boot(config, sweep_partials=False)
    dest = build_dashboard(ws)
    console.print(f"[green]dashboard: {dest}[/]")
    if open_browser:
        import os
        os.startfile(str(dest))  # noqa: S606 - local file, user-invoked


@app.command()
def auth(
    platform: str = typer.Argument(..., help="Social platform: youtube | tiktok | instagram | x_twitter"),
    config: Path = CONFIG_OPT,
) -> None:
    """Launch interactive browser to log into a social platform and save session cookies."""
    from clipforge.poster.scheduler import get_poster

    cfg, ws = _boot(config, sweep_partials=False)
    auth_dir = ws.root / "auth"
    poster = get_poster(platform)
    console.print(f"[green]Launching interactive browser session for {platform}...[/]")
    poster.login_interactive(auth_dir)
    console.print(f"[green]Session saved under {auth_dir}/{platform}_session.json[/]")


@app.command()
def post(
    clip: Path = typer.Option(..., "--clip", "-f", help="Path to rendered 9:16 clip MP4"),
    platform: str = typer.Option("youtube", "--platform", "-p", help="youtube | tiktok | instagram | x_twitter"),
    title: str = typer.Option("Viral Stream Clip", "--title", "-t", help="Clip post title"),
    caption: str = typer.Option("Check out this stream moment!", "--caption", help="Clip description / caption"),
    headed: bool = typer.Option(False, "--headed", help="Show browser UI (non-headless)"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt (still draft-only)"),
    config: Path = CONFIG_OPT,
) -> None:
    """Prepare a DRAFT post for one clip, for you to review and publish.

    Draft-only by design (VERIFICATION.md, 2026-07-27 amendment): the
    automation fills in the upload and stops. It never presses Publish,
    Post, or Share — you do, in the platform's own UI.

    The `--draft` flag is gone because there is nothing to contrast it with.
    """
    import time
    from clipforge.poster.scheduler import execute_post_job
    from clipforge.schemas.poster import PostJob

    clip_path = Path(clip)
    if not clip_path.exists():
        console.print(f"[red]Clip file not found: {clip_path}[/]")
        raise typer.Exit(1)

    cfg, ws = _boot(config, sweep_partials=False)
    auth_dir = ws.root / "auth"
    headless = not headed if headed else cfg.posting.headless

    job = PostJob(
        job_id=f"job_{int(time.time())}",
        clip_path=str(clip_path),
        platform=platform,  # type: ignore
        title=title,
        caption=caption,
        hashtags=["#Shorts", "#Viral", "#ClipForge"],
    )

    # Per-clip human approval — the third pin of the amendment. Approval is
    # a decision about THIS clip, so it is taken here and passed explicitly;
    # config can never grant it.
    console.print(f"[bold]Draft {job.platform}[/] post for [cyan]{clip_path.name}[/]")
    console.print(f"  title:   {title}")
    console.print(f"  caption: {caption[:80]}")
    if not yes:
        approved = typer.confirm("Prepare this draft now?", default=False)
    else:
        approved = True
    if not approved:
        console.print("[yellow]cancelled - nothing was uploaded[/]")
        raise typer.Exit(1)

    console.print(f"[green]Preparing draft {job.job_id} for {platform}...[/]")
    res = execute_post_job(job, auth_dir=auth_dir, headless=headless,
                           approved=approved)

    if res.status == "draft_saved":
        console.print(f"[green]Draft saved on {platform}. Open the platform "
                      "and publish it yourself when you're happy with it.[/]")
    else:
        console.print(f"[red]Draft failed: {res.error_message}[/]")
        raise typer.Exit(1)


@app.command()
def dub(
    clip: Path = typer.Option(None, "--clip", "-f",
                              help="Rendered clip filename or path"),
    lang: str = typer.Option(None, "--lang", "-l",
                             help="Target language code, e.g. es, pt, ja"),
    config: Path = CONFIG_OPT,
    subtitles_only: bool = typer.Option(
        False, "--subtitles-only",
        help="Translate and write the .srt, skip the voiced track"),
    keep_original: float = typer.Option(
        0.0, "--keep-original", min=0.0, max=1.0,
        help="Duck the original audio under the dub at this gain "
             "(0 = replace it entirely)"),
    languages: bool = typer.Option(
        False, "--languages", help="List targets and exit"),
) -> None:
    """Translate a clip's subtitles, and voice them where a voice exists.

    Subtitles work for every language the translator handles. Dubbed
    AUDIO additionally needs an installed Kokoro voice for the target —
    without one this writes the .srt and says so, rather than running the
    text through an English phonemiser and calling the result a dub.
    """
    from clipforge.dubbing import dub_clip, language_options

    cfg, ws = _boot(config, sweep_partials=False)

    if languages:
        for opt in language_options():
            mark = "audio+subs" if opt["audio"] else "subs only"
            console.print(f"  {opt['code']:4} {opt['label']:12} {mark}")
        return

    if clip is None or not lang:
        console.print("[red]--clip and --lang are required "
                      "(or use --languages to list targets)[/]")
        raise typer.Exit(2)

    name = clip.name
    rejected = False
    if not (Path(ws.clips) / name).is_file():
        if (Path(ws.clips) / "rejected" / name).is_file():
            rejected = True
        else:
            console.print(f"[red]clip not found in workspace/clips: {name}[/]")
            raise typer.Exit(1)

    console.print(f"[green]dubbing[/] {name} -> {lang}")
    try:
        res = dub_clip(ws, name, target=lang, rejected=rejected,
                       audio=not subtitles_only,
                       keep_original_at=keep_original)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        console.print(f"[red]dub failed:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]subtitles:[/] {res.subtitles_path}")
    if res.video_path:
        console.print(f"[green]dubbed video:[/] {res.video_path} "
                      f"(voice {res.voice})")
    for note in res.notes:
        console.print(f"[yellow]note:[/] {note}")


def _resolve_clip_name(ws, clip: Path) -> tuple[str, bool]:
    """Locate a clip by name in clips/ or clips/rejected/, or exit.

    Shared by `dub`, `voiceover` and `upscale` so the three cannot drift on
    where a clip may live or on what a missing one prints.
    """
    name = clip.name
    if (Path(ws.clips) / name).is_file():
        return name, False
    if (Path(ws.clips) / "rejected" / name).is_file():
        return name, True
    console.print(f"[red]clip not found in workspace/clips: {name}[/]")
    raise typer.Exit(1)


@app.command()
def voiceover(
    clip: Path = typer.Option(..., "--clip", "-f",
                              help="Rendered clip filename or path"),
    script: str = typer.Option(
        None, "--script", "-s", help="What the voice should say"),
    script_file: Path = typer.Option(
        None, "--script-file",
        help="Read the script from a file (use for anything with quotes, "
             "newlines or non-ASCII that a shell would mangle)"),
    voice: str = typer.Option(
        None, "--voice", help="Voice name; defaults to the engine's own"),
    duck_db: float = typer.Option(
        -12.0, "--duck-db",
        help="How far to drop the clip's own audio under the voice"),
    gain_db: float = typer.Option(0.0, "--gain-db", help="Voice gain"),
    config: Path = CONFIG_OPT,
) -> None:
    """Speak a script over a clip, ducking the clip's own audio under it.

    Writes `<clip>.vo.mp4` beside the original, which is left untouched.
    Kokoro is used when its weights are installed and flite otherwise; the
    result reports which one actually spoke rather than leaving you to
    infer it from how the output sounds.
    """
    from clipforge.enhance import voiceover_clip

    cfg, ws = _boot(config, sweep_partials=False)

    if script_file is not None:
        if script:
            console.print("[red]pass --script or --script-file, not both[/]")
            raise typer.Exit(2)
        try:
            script = Path(script_file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]cannot read {script_file}:[/] {exc}")
            raise typer.Exit(1) from exc
    if not (script or "").strip():
        console.print("[red]--script or --script-file is required[/]")
        raise typer.Exit(2)

    name, rejected = _resolve_clip_name(ws, clip)
    console.print(f"[green]voiceover[/] {name} ({len(script)} chars)")
    try:
        res = voiceover_clip(ws, name, script=script, rejected=rejected,
                             voice=voice, duck_db=duck_db,
                             voice_gain_db=gain_db)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        console.print(f"[red]voiceover failed:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]voice track:[/] {res.audio_path} "
                  f"({res.engine}, {res.voice})")
    console.print(f"[green]mixed clip:[/] {res.video_path}")
    for note in res.notes:
        console.print(f"[yellow]note:[/] {note}")


@app.command()
def upscale(
    clip: Path = typer.Option(..., "--clip", "-f",
                              help="Rendered clip filename or path"),
    height: int = typer.Option(
        2560, "--height", "-h",
        help="Target long edge in pixels (2560 = 1440p vertical, "
             "3840 = 4K vertical). Width follows the source aspect."),
    sharpen: float = typer.Option(
        None, "--sharpen", min=0.0, max=1.0,
        help="Post-resample sharpening, 0 to disable"),
    config: Path = CONFIG_OPT,
) -> None:
    """Resample a clip up to a larger frame.

    This is RESAMPLING, not learned super-resolution: libplacebo on the GPU
    where available, Lanczos otherwise, plus contrast-adaptive sharpening.
    It will not invent detail the source does not contain, and it refuses a
    target that is not actually larger rather than quietly degrading the
    clip while reporting success.

    Writes `<clip>.upscaled.mp4`; the original is untouched.
    """
    from clipforge.enhance import DEFAULT_SHARPEN, upscale_clip

    cfg, ws = _boot(config, sweep_partials=False)
    name, rejected = _resolve_clip_name(ws, clip)

    console.print(f"[green]upscaling[/] {name} -> long edge {height}px")
    try:
        out = upscale_clip(
            ws, name, height=height, rejected=rejected,
            sharpen=DEFAULT_SHARPEN if sharpen is None else sharpen)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        console.print(f"[red]upscale failed:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]upscaled:[/] {out}")


@app.command()
def web(host: str = typer.Option("127.0.0.1", "--host",
                                 help="Interface to bind. --lan is the "
                                      "shorthand for 0.0.0.0."),
        port: int = typer.Option(8000, "--port"),
        lan: bool = typer.Option(False, "--lan", "-l",
                                 help="Reachable from other devices on this "
                                      "network. Requires the access token."),
        tunnel: str = typer.Option("off", "--tunnel",
                                   help="off | cloudflare — a public https "
                                        "URL for reaching it from anywhere."),
        token: str = typer.Option("", "--token",
                                  help="Use this access token instead of the "
                                       "one saved in the workspace."),
        rotate: bool = typer.Option(False, "--rotate-token",
                                    help="Mint a new token, unpairing every "
                                         "device that has the old one."),
        insecure: bool = typer.Option(False, "--insecure-no-auth",
                                      help="Bind wide open with NO token. "
                                           "Every device on the network can "
                                           "then run this pipeline."),
        ):
    """Launch the BTA control API and dashboard.

    Bound to loopback it behaves as it always has: no token, no login,
    because anything that can reach it could already run the CLI. `--lan`
    and `--tunnel` change that, so they turn on the access token — this
    API starts subprocesses on this machine, and an open port that does
    that is a remote shell with a nicer front end.
    """
    import uvicorn

    from clipforge import remote

    if lan:
        host = "0.0.0.0"
    wide = host not in ("127.0.0.1", "localhost", "::1")
    want_tunnel = (tunnel or "off").strip().lower()
    if want_tunnel not in ("off", "cloudflare", "cloudflared"):
        console.print("[red]--tunnel takes 'off' or 'cloudflare'.[/]")
        raise typer.Exit(2)

    ws_root = Path("workspace")
    try:
        ws_root = Path(load_config(Path("config/config.toml")).workspace.root)
    except Exception:  # noqa: BLE001 - the default is the documented fallback
        pass
    ws_root.mkdir(parents=True, exist_ok=True)

    access_token: str | None = None
    if insecure:
        if want_tunnel != "off":
            # A tunnel with no token puts a subprocess-spawning API on the
            # public internet. There is no operator intent that makes that
            # the right call, so it is refused rather than warned about.
            console.print("[red]--insecure-no-auth cannot be combined with "
                          "--tunnel.[/] A public URL with no token is an open "
                          "shell.")
            raise typer.Exit(2)
        console.print("[bold red]No access token: every device that can reach "
                      f"{host}:{port} can run this pipeline.[/]")
    elif wide or want_tunnel != "off":
        access_token = (token.strip() or None)
        if access_token is None:
            access_token = (remote.rotate_token(ws_root) if rotate
                            else remote.load_or_create_token(ws_root))
    elif token.strip():
        access_token = token.strip()

    os.environ[remote.ENV_TOKEN] = access_token or ""
    os.environ[remote.ENV_REQUIRE_AUTH] = "0" if access_token is None else "1"
    os.environ["BTA_WEB_PORT"] = str(port)
    # A tunnel connects to us FROM 127.0.0.1, so every public request
    # arrives wearing a loopback address. Leaving loopback trusted while
    # one is up hands the whole internet an API that spawns subprocesses
    # on this machine — the token is set, and nothing ever checks it.
    # Loopback trust is therefore off for the life of the tunnel; the
    # banner prints a local link with the token already in it so the
    # operator's own browser is one click, exactly as before.
    os.environ[remote.ENV_TRUST_LOOPBACK] = "0" if want_tunnel != "off" else "1"

    live_tunnel = None
    if want_tunnel != "off":
        console.print("[dim]starting cloudflared quick tunnel…[/]")
        try:
            live_tunnel = remote.start_cloudflare_tunnel(port)
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]tunnel failed:[/] {exc}")
            raise typer.Exit(1) from exc
        os.environ[remote.ENV_PUBLIC_URL] = live_tunnel.url or ""

    _print_access_banner(host, port, access_token,
                         tunnel_url=(live_tunnel.url if live_tunnel else None))

    try:
        # reload=False, deliberately. The dashboard spawns pipeline runs as
        # subprocesses and tracks them in memory; a reload restarts the app and
        # empties that registry while the spawned render keeps holding the GPU
        # — orphaned, uncancellable, and invisible to both /api/tasks and
        # `bta swarm status`. Auto-reload is a development convenience that
        # here loses running work.
        uvicorn.run("clipforge.web:app", host=host, port=port, reload=False)
    finally:
        if live_tunnel is not None:
            live_tunnel.stop()


def _print_access_banner(host: str, port: int, token: str | None, *,
                         tunnel_url: str | None = None) -> None:
    """Print every URL this server is reachable at, and what each costs.

    An operator who cannot find the address types 0.0.0.0 into a phone,
    fails, and turns auth off. Printing the working links — with the
    token already in them — is what makes the secure path the easy one.
    """
    from clipforge import remote

    console.print()
    console.print("[bold green]BTA Studio[/bold green]")
    # With a tunnel up, loopback is no longer trusted (a tunnelled request
    # wears a loopback address), so the local link has to carry the token
    # or the operator's own browser lands on the pairing page.
    local_suffix = (f"/?{remote.TOKEN_QUERY}={token}"
                    if (tunnel_url and token) else "/")
    console.print(
        f"  [bold]On this machine[/]  http://127.0.0.1:{port}{local_suffix}")

    if host in ("127.0.0.1", "localhost", "::1") and not tunnel_url:
        console.print("  [dim]Loopback only. Add --lan to reach it from your "
                      "phone, or --tunnel cloudflare from anywhere.[/]")
        console.print()
        return

    urls = remote.access_urls(port, token)
    if urls["lan"]:
        console.print("  [bold]On this network[/]")
        for u in urls["lan"]:
            console.print(f"    {u}")
    if urls["tailscale"]:
        console.print("  [bold]Over Tailscale[/] (works anywhere, encrypted)")
        for u in urls["tailscale"]:
            console.print(f"    {u}")
    if tunnel_url:
        suffix = f"/?{remote.TOKEN_QUERY}={token}" if token else "/"
        console.print("  [bold]Public[/] (anywhere, no VPN)")
        console.print(f"    {tunnel_url.rstrip('/')}{suffix}")

    if token:
        console.print()
        console.print(f"  [dim]access token[/] {token}")
        console.print("  [dim]Or open the bare address on the other device "
                      "and pair it with the six-digit code from[/] "
                      "[bold]Connect a device[/] [dim]in the sidebar.[/]")
    if tunnel_url:
        console.print("  [yellow]This URL is on the public internet. Anyone "
                      "with the token can drive this machine — stop the "
                      "server when you are done.[/]")
    console.print()


@app.command()
def test_render(video: Path, s1_json: Path, s2_json: Path):
    """Force a render of the top candidate to test the UI pipeline."""
    import json
    from clipforge.schemas.candidates import CandidatesArtifact
    from clipforge.stages.s5_subtitles import S5Subtitles
    from clipforge.stages.s6_render import S6Render
    from clipforge.paths import Workspace
    from clipforge.config import load_config
    
    try:
        cfg = load_config(Path("config/config.toml"))
        ws = Workspace(cfg.workspace.root).ensure()
    except Exception:
        ws = Workspace(Path("workspace")).ensure()
    
    s2_data = json.loads(s2_json.read_text(encoding="utf-8"))
    s2 = CandidatesArtifact.model_validate(s2_data)
    
    if not s2.candidates:
        console.print("[red]No candidate windows found in S2 artifact![/red]")
        raise typer.Exit(1)
        
    best_cand = s2.candidates[0]
    
    console.print(f"[bold cyan]Generating subtitles for {best_cand.start}s...[/bold cyan]")
    subs_path = ws.tmp / "tmp_subs.ass"
    subs_path.parent.mkdir(parents=True, exist_ok=True)
    S5Subtitles().run(s1_json, best_cand, subs_path)
    
    console.print("[bold cyan]Spinning up NVENC renderer...[/bold cyan]")
    out_path = S6Render().run(video, best_cand, subs_path)
    
    console.print(f"[bold green]Render complete: {out_path.name}! Check your UI dashboard.[/bold green]")


@app.command()
def live(target: str = typer.Argument(..., help="A YouTube live URL, a "
                                               "channel @handle, or a Twitch "
                                               "channel name"),
         quality: str = typer.Option("", "--quality",
                                     help="streamlink quality; defaults to "
                                          "[ingest] quality"),
         clips: int = typer.Option(0, "--clips",
                                   help="Clips per captured window; defaults "
                                        "to [orchestration] clips_per_window"),
         platform: str = typer.Option("youtube", "--platform",
                                      help="youtube | twitch | kick"),
         segment: float = typer.Option(
             300.0, "--segment",
             help="Seconds of stream per clipping window. Shorter = clips "
                  "appear sooner; longer = better context for the ranker."),
         config: Path = CONFIG_OPT) -> None:
    """Capture ONE live stream and clip it while it is still running.

    `watch` polls a configured watchlist; this points at a single stream
    that is on air right now, which is what an operator actually does when
    they see something worth clipping.

    The stream is segmented as it arrives and each finished segment goes
    straight into the clip pipeline, so clips land in the gallery minutes
    into a broadcast rather than after it ends. YouTube live went through
    the VOD path before this — the monitor counts `platform != "youtube"`
    as the live set — so a YouTube broadcast produced nothing at all until
    it finished and became a downloadable video.

    Ctrl+C stops it and finalises the tail.
    """
    import threading

    from clipforge.dispatch import ClipDispatcher
    from clipforge.ingest import kick, twitch, youtube
    from clipforge.ingest.chunker import ChunkerConfig, ChunkerSession
    from clipforge.ingest.monitor import disk_allows
    from clipforge.ingest.retention import workspace_lock
    from clipforge.state import StateDB

    cfg, ws = _boot(config, sweep_partials=False)
    platform = platform.strip().lower()
    if platform not in ("youtube", "twitch", "kick"):
        console.print("[red]--platform must be youtube, twitch or kick[/]")
        raise typer.Exit(2)

    quality = quality or cfg.ingest.quality
    per_window = clips or cfg.orchestration.clips_per_window
    # `watch` uses [ingest] segment_time_s, which is 900 s — right for an
    # unattended overnight run and wrong here: this command exists to be
    # watched, and a first clip fifteen minutes in looks like nothing is
    # happening. Five minutes is still enough context for the ranker to
    # find a moment.
    if not 30.0 <= segment <= 3600.0:
        console.print("[red]--segment must be between 30 and 3600 seconds[/]")
        raise typer.Exit(2)

    # Probe before committing to a capture: streamlink will happily sit on
    # a non-live URL until its timeout, which looks exactly like a working
    # capture that produces nothing.
    console.print(f"[dim]checking whether {target} is live…[/]")
    if platform == "youtube":
        state = youtube.is_live(target)
        url_or_handle = youtube.live_url(target)
        args = youtube.chunker_args(target, quality)
        label = youtube.live_title(target) or url_or_handle
        # NOT the raw target: the chunker builds `chunks/{platform}_{handle}/`
        # from this, and a URL contains ':' and '/'. Passing the URL made
        # every connect fail with WinError 123 and retry forever — the
        # capture looked alive and produced nothing (measured against the
        # live ISS stream, 2026-08-12).
        handle = youtube.capture_handle(target)
    elif platform == "twitch":
        state = twitch.is_live(target)
        args = twitch.chunker_args(target, quality,
                                   disable_ads=cfg.ingest.twitch_disable_ads)
        label = f"twitch/{target}"
        handle = target.strip().lstrip("@")[:80]
    else:
        state = kick.is_live(target)
        args = kick.chunker_args(target, quality)
        label = f"kick/{target}"
        handle = target.strip().lstrip("@")[:80]

    if state is False:
        console.print(f"[yellow]{target} is not live right now.[/] Nothing to "
                      "capture. For a finished video use `bta grab <url>`.")
        raise typer.Exit(1)
    if state is None:
        console.print("[yellow]could not confirm it is live[/] — trying "
                      "anyway; the capture ends on its own if there is no "
                      "stream.")

    with workspace_lock(ws) as acquired:
        if not acquired:
            console.print("[red]another clipforge is already using this "
                          f"workspace ({ws.root}). Stop it first.[/]")
            raise typer.Exit(3)

        db = StateDB(ws.state_db)
        stop = threading.Event()

        def _clip_window(path: Path, abs_start_s: float) -> None:
            process(input_path=path, config=config, abs_offset=abs_start_s,
                    clips=per_window, jumpcut=None)

        dispatcher = ClipDispatcher(
            handler=_clip_window,
            maxsize=cfg.orchestration.queue_maxsize).start()

        # Segments reach the clipper through the dispatcher's queue, never
        # inline: a live stream is not replayable, and a four-minute render
        # on the capture thread is four minutes of stream lost for good.
        captured = {"segments": 0}

        def _on_segment(event) -> None:
            captured["segments"] += 1
            console.print(f"[green]segment {event.seg_index}[/] "
                          f"({event.duration_s:.0f}s) -> clipping")
            dispatcher.submit(Path(event.path), event.abs_start_s)

        session = ChunkerSession(
            db=db, chunks_root=ws.chunks, quarantine_dir=ws.quarantine,
            platform=platform, handle=handle,
            streamlink_args=args,
            cfg=ChunkerConfig(
                segment_time_s=segment,
                ready_stable_s=cfg.ingest.segment_ready_stable_s,
                backoff_base_s=cfg.ingest.backoff_base_s,
                backoff_max_s=cfg.ingest.backoff_max_s,
            ),
            on_segment_ready=_on_segment,
            disk_ok=lambda: disk_allows(ws.root, cfg.disk.free_floor_gb),
            log_dir=ws.logs,
        )

        console.print(f"[bold green]capturing[/] {label}")
        console.print(f"[dim]{segment:.0f}s segments · "
                      f"{per_window} clip(s) per segment · first clip in about "
                      f"{segment/60:.0f} min · Ctrl+C to stop[/]")
        try:
            session.run(stop)
        except KeyboardInterrupt:
            console.print("[yellow]stopping — finalizing the tail[/]")
            stop.set()
        finally:
            stop.set()
            dispatcher.stop()
            stats = dispatcher.stats.snapshot()
            console.print(f"[yellow]capture ended — {captured['segments']} "
                          f"segment(s) captured, {stats['processed']} "
                          f"window(s) clipped, {stats['failed']} failed, "
                          f"{stats['dropped']} dropped[/]")
            db.close()

        # Exiting 0 after capturing nothing is how a broken capture gets
        # reported as a finished one. The dashboard reads the exit code,
        # and "completed · 0 clips" is indistinguishable from success at a
        # glance — which is exactly what happened on the first live run.
        if captured["segments"] == 0:
            console.print(
                "[red]nothing was captured.[/] The stream may have ended, or "
                "streamlink could not open it. The log above says which.")
            raise typer.Exit(1)


swarm_app = typer.Typer(name="swarm", no_args_is_help=True,
                        help="Durable multi-role task swarm: plan work, "
                             "run it, watch it.")
app.add_typer(swarm_app, name="swarm")


def _swarm_board(config: Path):
    from clipforge.swarm import TaskBoard

    cfg, ws = _boot(config, sweep_partials=False)
    return cfg, ws, TaskBoard(ws.root / "swarm.sqlite3")


@swarm_app.command("plan")
def swarm_plan(
        brief: str = typer.Option(None, "--brief", help="One piece"),
        briefs_file: Path = typer.Option(
            None, "--briefs-file",
            help="One brief per line -> one generate task per line"),
        source: str = typer.Option(None, "--source",
                                   help="A file or URL to clip"),
        niche: str = typer.Option(None, "--niche",
                                  help="Niche name (see: bta swarm niches)"),
        shots: int = typer.Option(None, "--shots"),
        clips: int = typer.Option(3, "--clips"),
        aspect: str = typer.Option(None, "--aspect"),
        model: str = typer.Option(None, "--model"),
        priority: int = typer.Option(50, "--priority",
                                     help="Lower runs first"),
        goal: str = typer.Option(None, "--goal",
                                 help="A label tying related tasks together"),
        config: Path = CONFIG_OPT) -> None:
    """Post work to the board without running it. Combine with `serve`."""
    brief = _cli_value(brief, None)
    briefs_file = _cli_value(briefs_file, None)
    source = _cli_value(source, None)
    niche = _cli_value(niche, None)

    payload: dict = {"niche": niche, "shots": shots, "aspect": aspect,
                     "model": model, "clips": clips}
    if briefs_file is not None:
        lines = [ln.strip() for ln in briefs_file.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            console.print(f"[red]{briefs_file} has no non-empty lines[/]")
            raise typer.Exit(2)
        payload["briefs"] = lines
    elif brief is not None:
        payload["brief"] = brief
    elif source is not None:
        payload["source"] = source
    else:
        console.print("[red]give one of --brief, --briefs-file, --source[/]")
        raise typer.Exit(2)

    _cfg, _ws, board = _swarm_board(config)
    try:
        task_id = board.submit("plan", payload, priority=priority, goal=goal)
    finally:
        board.close()
    console.print(f"[green]queued plan task #{task_id}[/] — run "
                  f"`bta swarm serve` (or `--once`) to work it")


@swarm_app.command("serve")
def swarm_serve(
        config: Path = CONFIG_OPT,
        cpu_workers: int = typer.Option(4, "--cpu-workers"),
        once: bool = typer.Option(
            False, "--once",
            help="Drain everything currently on the board, then exit"),
        timeout: float = typer.Option(
            3600.0, "--timeout", help="Only with --once: give up after "
                                      "this many seconds of no progress"),
        brief: str = typer.Option(
            None, "--brief", help="Seed a plan task before serving"),
        niche: str = typer.Option(None, "--niche"),
        shots: int = typer.Option(None, "--shots")) -> None:
    """Run the swarm. Generation and clipping share the one GPU permit;
    everything else runs concurrently up to --cpu-workers."""
    from clipforge.swarm.roles import build_swarm

    brief = _cli_value(brief, None)
    niche = _cli_value(niche, None)
    cfg, ws, board = _swarm_board(config)
    try:
        if brief is not None:
            tid = board.submit("plan", {"brief": brief, "niche": niche,
                                        "shots": shots})
            console.print(f"[green]seeded plan task #{tid}[/]")

        sup = build_swarm(board, cpu_workers=cpu_workers)
        desc = sup.describe()
        console.print(f"[green]swarm up[/] — {len(desc['agents'])} role(s), "
                      f"{desc['cpu_workers']} CPU worker(s), "
                      f"1 GPU permit shared by "
                      f"{sum(1 for a in desc['agents'] if a['gpu'])} role(s)")
        for a in desc["agents"]:
            console.print(f"    {a['name']:10s} {'GPU' if a['gpu'] else 'cpu':4s}"
                          f"  {', '.join(a['kinds'])}")

        if once:
            stats = sup.run_until_drained(timeout_s=timeout)
            console.print(f"[green]drained[/] — {stats.snapshot()}")
        else:
            console.print("[green]serving — Ctrl+C to stop[/]")
            try:
                sup.serve()
            except KeyboardInterrupt:
                sup.stop()
                console.print("[yellow]stopped[/]")
    finally:
        board.close()


@swarm_app.command("status")
def swarm_status(config: Path = CONFIG_OPT,
                 limit: int = typer.Option(20, "--limit")) -> None:
    """Board counts and the most recently touched tasks."""
    _cfg, _ws, board = _swarm_board(config)
    try:
        counts = board.stats()
        console.print("[green]board:[/] " +
                      ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                      or "empty")
        for t in board.recent(limit):
            mark = {"done": "[green]done[/]", "failed": "[red]failed[/]",
                    "running": "[yellow]running[/]"}.get(t.status, t.status)
            console.print(f"  #{t.id:<4} {t.kind:10s} {mark:20s} "
                          f"attempts={t.attempts}"
                          + (f"  {t.error[:80]}" if t.error else ""))
    finally:
        board.close()


@swarm_app.command("niches")
def swarm_niches() -> None:
    """List selectable niches."""
    from clipforge.niches import niche_summary

    for n in niche_summary():
        console.print(f"[bold]{n['name']}[/] — {n['label']}: {n['summary']}")


@app.command("models")
def models_cmd() -> None:
    """List generation models: what is downloaded, and what is proven.

    `--model` needs a key, and until now the only way to learn one was to
    pass a wrong one and read the error. Downloaded and verified are
    shown apart on purpose: an UNVERIFIED model can be run — an explicit
    request is how a model gets verified — but it is never chosen for you.
    """
    from clipforge.genvideo.models import REGISTRY, weights_present

    for spec in REGISTRY.values():
        here = "downloaded" if weights_present(spec) else "NOT downloaded"
        proof = "verified here" if spec.verified else "UNVERIFIED"
        extra = f", needs {spec.interpreter}" if spec.interpreter else ""
        console.print(f"[bold]{spec.key}[/] - {spec.label}")
        console.print(f"    {here}, {proof}{extra}")
        console.print(f"    up to {spec.max_pixels:,} px/frame, "
                      f"{spec.vram_gb:g} GB VRAM, {spec.steps} steps"
                      + (f", {spec.requires_quantization} required"
                         if spec.requires_quantization else ""))
        if spec.notes:
            console.print(f"    [dim]{spec.notes}[/]")


@app.command()
def holdout(config: Path = CONFIG_OPT,
            sources: list[str] = typer.Option(
                None, "--source", "-s",
                help="Restrict to clips cut from these source filenames "
                     "(repeatable). Omit to measure every clip on disk - a "
                     "snapshot, not a comparable holdout set."),
            save: bool = typer.Option(
                True, "--save/--no-save",
                help="Record this measurement under the workspace's holdout/ "
                     "directory so the next run can diff against it.")) -> None:
    """Measure the clips the pipeline actually shipped - one comparable number.

    Every stage already wrote down how good its output was (S2/S3's scorecard,
    S7's QA verdict, S6's measured loudness); nothing ever read them back as a
    single figure, so the only number anyone could quote was the test count -
    which says the code does what it was told, never whether the clips are
    worth posting. This reads those sidecars back and reports mean score, QA
    pass rate and loudness-in-band for the clips on disk.

    With --source it measures a FIXED set, so two runs over the same sources
    produce a delta worth reading; without it, it is a snapshot of whatever is
    there and the next run is reported as incomparable rather than diffed into
    a meaningless number.
    """
    from clipforge import holdout as _holdout

    _, ws = _boot(config, sweep_partials=False)
    srcs = _cli_value(sources, None)
    srcs = list(srcs) if srcs else None
    report = _holdout.measure(ws, sources=srcs)

    if not report["clips"]:
        where = f" from {', '.join(srcs)}" if srcs else ""
        console.print(f"[yellow]no accepted clips to measure[/]{where}. "
                      "Run `bta process` first.")
        return

    tail = f", {report['rejected']} rejected" if report["rejected"] else ""
    console.print(f"[bold]{report['clips']} clip(s) measured[/]{tail}")

    def _num(v: Any) -> str:
        return "n/a" if v is None else f"{v:g}"

    def _pct(v: Any) -> str:
        return "n/a" if v is None else f"{v * 100:.0f}%"

    median = report["median_score"]
    median_note = f"  (median {median:g})" if median is not None else ""
    console.print(f"  mean score       {_num(report['mean_score'])}{median_note}")
    console.print(f"  QA pass rate     {_pct(report['qa_pass_rate'])}")
    console.print(f"  loudness in-band {_pct(report['loudness_in_band'])}  "
                  f"(target {_holdout.TARGET_LUFS:g} +/- "
                  f"{_holdout.LUFS_TOLERANCE:g} LUFS)")
    if report["mean_duration_s"] is not None:
        console.print(f"  mean duration    {report['mean_duration_s']:g}s")
    if report["grades"]:
        grades = ", ".join(f"{g} x{n}" for g, n in report["grades"].items())
        console.print(f"  grades           {grades}")
    if report["framing_modes"]:
        modes = ", ".join(f"{m} x{n}" for m, n in report["framing_modes"].items())
        console.print(f"  framing          {modes}")

    # Diff against the most recent PRIOR measurement, before this one is saved.
    prev = _holdout.previous(ws)
    delta = _holdout.compare(report, prev)
    if delta.get("comparable"):
        moved = {k: v for k, v in delta.items()
                 if k != "comparable" and isinstance(v, (int, float)) and v}
        if moved:
            console.print("[bold]vs previous run:[/]")
            for key, val in moved.items():
                sign = "+" if val > 0 else ""
                console.print(f"  {key:<16} {sign}{val:g}")
        else:
            console.print("[dim]vs previous run: no change[/]")
    elif prev is not None:
        console.print(f"[dim]not comparable to the previous run: "
                      f"{delta.get('reason')}[/]")

    if save:
        path = _holdout.save(ws, report)
        console.print(f"[green]saved[/] {path}")


def main() -> None:  # console_scripts shim
    # Windows cp1252 guard: with stdout piped/redirected (scheduled task,
    # `doctor > report.txt`), sys.stdout.encoding is cp1252 and any stray
    # non-ASCII in output crashes with UnicodeEncodeError. Console strings
    # are kept ASCII-safe on top of this, but reconfigure is the backstop.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and \
                (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # exotic redirection targets
                pass
    try:
        app()
    except ClipForgeError as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
