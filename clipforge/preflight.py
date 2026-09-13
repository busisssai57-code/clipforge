"""`clipforge doctor` — diagnose every external prerequisite, actionably.

Rules:
  * A check NEVER raises — it returns a :class:`CheckResult` with ``ok``,
    a human message, and a concrete ``fix`` (command or URL).
  * Checks are independent; one failure never hides another.
  * Severity: ``required`` blocks the pipeline; ``optional`` degrades a
    feature (e.g. Kick ingest, NVENC → libx264).

The known traps this file exists for: T5 (gated pyannote), T6 (cuDNN DLL
hell on Windows), missing ffmpeg, missing CUDA.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from clipforge import ffmpeg as ff
from clipforge.config import Secrets
from clipforge.dllpaths import ensure_nvidia_dll_dirs

Severity = Literal["required", "optional"]

# T5: every gated repo diarization touches must be accepted, or S1's
# diarization dies mid-run at CP2.
#
# The spec names 3.1 + segmentation-3.0, correct for the whisperx of its
# day. A real run here on whisperx 3.8 failed on
# `pyannote/speaker-diarization-community-1` instead — the default model
# moved. Probing only the spec's two repos would have reported everything
# green right up until S1 failed on the machine, so all three are checked.
PYANNOTE_GATED_REPOS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    severity: Severity
    message: str
    fix: str = ""


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --------------------------------------------------------------------------
# individual checks — each is total (never raises)
# --------------------------------------------------------------------------


def check_python() -> CheckResult:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    return CheckResult(
        "python", ok, "required",
        f"Python {v.major}.{v.minor}.{v.micro}",
        "" if ok else "Use Python 3.11/3.12 - the CUDA AI stack (ctranslate2, "
                      "torch cu12x wheels) is not reliable on other versions here.")


def check_ffmpeg() -> CheckResult:
    path = ff.find_binary("ffmpeg")
    if path is None:
        return CheckResult(
            "ffmpeg", False, "required", "ffmpeg not found",
            "winget install Gyan.FFmpeg   (or set CLIPFORGE_FFMPEG_DIR to a "
            "folder containing ffmpeg.exe + ffprobe.exe)")
    return CheckResult("ffmpeg", True, "required", f"found: {path}")


def check_ffprobe() -> CheckResult:
    path = ff.find_binary("ffprobe")
    if path is None:
        return CheckResult(
            "ffprobe", False, "required", "ffprobe not found",
            "Install the full ffmpeg distribution (imageio-ffmpeg does NOT "
            "ship ffprobe): winget install Gyan.FFmpeg")
    return CheckResult("ffprobe", True, "required", f"found: {path}")


def check_ffmpeg_capabilities() -> list[CheckResult]:
    """NVENC encoder + libass filter, probed from the actual binary."""
    results: list[CheckResult] = []
    try:
        encoders = ff.list_encoders()
        filters = ff.list_filters()
    except Exception as exc:
        return [CheckResult("ffmpeg-caps", False, "required",
                            f"could not probe ffmpeg capabilities: {exc}",
                            "Reinstall ffmpeg; the binary is present but not runnable.")]
    # LISTED is not WORKING. This asked ffmpeg which encoders it knows
    # about and reported PASS, while every real render on this machine
    # failed with `-22 Invalid argument` and fell back to libx264 — the
    # driver (596.49) is older than the ffmpeg 8.x nvenc path needs. A
    # doctor that says the fast encoder is available while the pipeline
    # silently uses the slow one is the report-vs-reality drift this
    # whole module exists to catch, so it encodes one frame and looks.
    listed = "h264_nvenc" in encoders
    works, why = (_nvenc_encodes_a_frame() if listed
                  else (False, "h264_nvenc is not in this ffmpeg build"))
    results.append(CheckResult(
        "h264_nvenc", works, "optional",
        "NVENC encodes" if works else f"NVENC unusable: {why}",
        "" if works else
        ("Renders fall back to libx264, which works and is slower. "
         + ("Update the NVIDIA driver: ffmpeg 8.x needs a newer one than "
            "this machine has." if listed and "driver" in why.lower() else
            "The reason above is not the driver version; it names the "
            "parameter the encoder rejected." if listed else
            "Install an ffmpeg build with --enable-nvenc."))))
    has_ass = " ass " in filters or "\nass" in filters or " ass\n" in filters
    results.append(CheckResult(
        "libass", has_ass, "required",
        "ass subtitle filter available" if has_ass else "ass filter missing",
        "" if has_ass else "Install a full ffmpeg build with libass "
                           "(Gyan.FFmpeg full)."))
    return results


#: The frame the NVENC probe encodes. NOT 128x128: that is below the
#: encoder's minimum frame dimension, so the probe failed with "Frame
#: Dimension less than the minimum supported value" on a machine where every
#: real 1080x1920 render used h264_nvenc - and then told the operator to
#: update a driver (616.56) that was already current. The driver WAS the
#: cause once, at 596.49 in August; the too-small frame kept the verdict
#: alive after the driver was fixed. 320x240 clears every NVENC minimum and
#: still encodes in well under a second.
NVENC_PROBE_SIZE = "320x240"

#: Lines ffmpeg prints on EVERY run. "Stream #0:0 -> #0:0 (... h264_nvenc)"
#: contains "nvenc", so a substring search quoted the stream mapping as the
#: reason the encoder failed.
_FFMPEG_BOILERPLATE = ("Stream #", "Stream mapping", "Input #", "Duration:",
                       "Output #", "frame=")


def _nvenc_encodes_a_frame() -> tuple[bool, str]:
    """Encode one frame with nvenc and report what happened.

    Two seconds at most, and it is the only question worth asking: the
    encoder being listed says nothing about whether this driver can run
    it. Failure text is trimmed to the last line ffmpeg wrote, which is
    where the actual reason lives.
    """
    import subprocess  # noqa: PLC0415

    from clipforge.ffmpeg import require_binary  # noqa: PLC0415

    try:
        proc = subprocess.run(
            [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
             "-f", "lavfi", "-i",
             f"testsrc=size={NVENC_PROBE_SIZE}:rate=1:d=1",
             "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001 - a check must not raise
        return False, f"{type(exc).__name__}: {exc}"[:120]
    if proc.returncode == 0:
        return True, ""
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines()
             if ln.strip() and not ln.strip().startswith(_FFMPEG_BOILERPLATE)]
    # ffmpeg's LAST line is "Conversion failed!", which names nothing. The
    # line that explains it is higher up. Most specific marker first: the
    # generic "nvenc" used to win over "InitializeEncoder", so the tile quoted
    # a stream-mapping line instead of the actual rejection.
    for marker in ("driver", "InitializeEncoder", "Cannot load",
                   "No capable devices", "Invalid argument", "nvenc"):
        for ln in lines:
            if marker.lower() in ln.lower():
                return False, ln[:150]
    return False, (lines[-1][:150] if lines else f"exit {proc.returncode}")


def check_cuda() -> CheckResult:
    if not _module_available("torch"):
        return CheckResult(
            "cuda", False, "required", "torch not installed",
            "pip install torch --index-url https://download.pytorch.org/whl/cu121")
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult(
                "cuda", False, "required", "torch present but CUDA unavailable",
                "Check NVIDIA driver (nvidia-smi) and that torch is a +cu12x "
                "wheel, not CPU-only: python -c \"import torch; print(torch.__version__)\"")
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return CheckResult("cuda", True, "required", f"{name} ({total_gb:.0f} GB)")
    except Exception as exc:  # driver/DLL breakage manifests as import-time errors
        return CheckResult("cuda", False, "required", f"CUDA probe failed: {exc}",
                           "Reinstall the NVIDIA driver and the cu12x torch wheel.")


def check_cudnn_dlls() -> CheckResult:
    """T6 — ctranslate2 (faster-whisper) needs cuBLAS/cuDNN DLLs loadable
    from the process. On Windows they typically come from the pip packages
    ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` (whisperx pulls them in),
    but the DLL directory must be on the search path."""
    if sys.platform != "win32":  # pragma: no cover
        return CheckResult("cudnn-dlls", True, "required", "non-Windows: skipped")
    if not _module_available("ctranslate2"):
        return CheckResult(
            "cudnn-dlls", False, "optional", "ctranslate2 not installed yet",
            "Installed with whisperx at CP2; re-run doctor afterwards.")
    # Probe with the SAME search path the pipeline runs with: register the
    # pip-installed nvidia/*/bin dirs first (clipforge.dllpaths, also called
    # at CLI boot), THEN attempt the loads.
    ensure_nvidia_dll_dirs()
    candidates = ["cudnn_ops64_9.dll", "cudnn_ops_infer64_8.dll", "cublas64_12.dll"]
    loaded: list[str] = []
    for dll in candidates:
        try:
            ctypes.WinDLL(dll)
            loaded.append(dll)
        except OSError:
            continue
    if any(d.startswith("cudnn") for d in loaded) and any(d.startswith("cublas") for d in loaded):
        return CheckResult("cudnn-dlls", True, "required", f"loadable: {', '.join(loaded)}")
    return CheckResult(
        "cudnn-dlls", False, "required",
        f"cuDNN/cuBLAS DLLs not loadable (found only: {loaded or 'none'})",
        "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 - clipforge "
        "registers the site-packages nvidia/*/bin dirs via os.add_dll_directory "
        "at startup (clipforge/dllpaths.py); see README, cuDNN section.")


def _probe_gated_repo(repo: str, token: str) -> bool | None:
    """True = accessible, False = gated/denied, None = could not determine
    (offline, HF outage). Uses only stdlib — the doctor must not depend on
    huggingface_hub being installed at CP0."""
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return None
            # A gated repo the token cannot access can also surface as
            # 200 + {"gated": ...} without config; require real metadata.
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("id") == repo)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        return None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def check_hf_token(*, probe: Callable[[str, str], bool | None] | None = None,
                   ) -> CheckResult:
    """T5 — pyannote diarization models are gated behind accepted terms.

    Token PRESENCE is not enough: the exact failure this trap exists for is
    a valid token whose owner never clicked "accept" on the model pages —
    which then 403s twenty minutes into the first S1 run. So when a token
    exists, actually probe both gated repos (graceful on offline machines).
    """
    if probe is None:
        # Resolved at CALL time so tests can stub the module attribute —
        # a def-time default would bind the original function forever.
        probe = _probe_gated_repo
    accept_urls = "\n".join(f"  https://huggingface.co/{r}" for r in PYANNOTE_GATED_REPOS)
    token = Secrets().hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return CheckResult(
            "hf-token", False, "optional", "No Hugging Face token configured",
            f"Set CLIPFORGE_HF_TOKEN (or HF_TOKEN). Then accept terms at BOTH:\n{accept_urls}")

    results = {r: probe(r, token) for r in PYANNOTE_GATED_REPOS}  # one probe each
    denied = [r for r, ok in results.items() if ok is False]
    if denied:
        return CheckResult(
            "hf-token", False, "optional",
            f"token present but gated access DENIED for: {', '.join(denied)}",
            f"Log in as the token's owner and accept the terms at:\n{accept_urls}")
    unknown = [r for r, ok in results.items() if ok is None]
    if unknown:
        return CheckResult(
            "hf-token", True, "optional",
            "token present; gated access could not be verified (offline?) - "
            "will be enforced on first S1 run")
    return CheckResult("hf-token", True, "optional",
                       "token present; both gated pyannote repos accessible")


def check_torchcodec() -> CheckResult:
    """pyannote decodes audio through torchcodec, whose native library only
    supports FFmpeg 4-7. With a current ffmpeg (8.x) it fails to load and
    file-path diarization dies with a DLL error that names nothing relevant.

    ClipForge sidesteps it by decoding audio itself, so a broken torchcodec
    is NOT fatal — but the operator should know why the warning appears.
    """
    if not _module_available("torchcodec"):
        return CheckResult("torchcodec", True, "optional",
                           "not installed (ClipForge decodes audio itself)")
    try:
        import torchcodec._core  # noqa: F401, PLC0415

        return CheckResult("torchcodec", True, "optional", "loadable")
    except Exception:
        return CheckResult(
            "torchcodec", True, "optional",
            "present but its native library will not load (expected with "
            "ffmpeg 8.x; harmless - S1 decodes audio via ffmpeg itself)")


def check_tools() -> list[CheckResult]:
    out: list[CheckResult] = []
    for tool, sev, fix in (
        ("streamlink", "required", "pip install streamlink"),
        ("yt-dlp", "required", "pip install yt-dlp"),
    ):
        found = shutil.which(tool) is not None or _module_available(tool.replace("-", "_"))
        out.append(CheckResult(tool, found, sev,
                               "available" if found else f"{tool} not found",
                               "" if found else fix))
    return out


def check_disk(workspace_root: Path, floor_gb: float) -> CheckResult:
    # The workspace may not exist yet — measure the nearest existing ancestor
    # on the same volume (resolve() also anchors relative paths to cwd).
    probe_path = workspace_root.resolve()
    while not probe_path.exists() and probe_path.parent != probe_path:
        probe_path = probe_path.parent
    try:
        free = shutil.disk_usage(probe_path).free / 1024**3
    except OSError as exc:
        return CheckResult("disk", False, "required", f"cannot stat {workspace_root}: {exc}")
    ok = free > floor_gb
    return CheckResult(
        "disk", ok, "required", f"{free:.0f} GB free (floor {floor_gb:.0f} GB)",
        "" if ok else "Free disk space or lower disk.free_floor_gb; ingestion "
                      "pauses below the floor.")


def check_workspace_writable(workspace_root: Path) -> CheckResult:
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        probe = workspace_root / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult("workspace", True, "required", f"writable: {workspace_root}")
    except OSError as exc:
        return CheckResult("workspace", False, "required",
                           f"not writable: {workspace_root} ({exc})",
                           "Fix permissions or point workspace.root elsewhere.")


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def check_ranking_weights(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                          ) -> CheckResult:
    """Whether S3's vision-language weights are on disk, before a run needs them.

    A first run with an empty cache spends an hour downloading ~7 GB from
    inside the stage, printing "S3: ranking candidate windows" and then
    nothing. MEASURED here: the cache held 16 MB of config and tokenizer
    files — enough to look present to anything checking for the folder —
    and the run sat there. Reported here so the wait is a known cost
    before it is a mystery.

    Size, not presence: the folder exists as soon as one config file has
    been fetched.
    """
    from clipforge.paths import hf_cache_dir  # noqa: PLC0415

    folder = hf_cache_dir(model_id)
    # blobs/ ONLY. On Windows the snapshot is a COPY of each blob rather
    # than a symlink, so walking the whole folder counts every weight
    # twice and reported 13.9 GB for a 6.9 GB model.
    have = sum(f.stat().st_size for f in (folder / "blobs").glob("*")
               if f.is_file()) if (folder / "blobs").is_dir() else 0
    # A half-finished fetch leaves `.incomplete` blobs, and a folder of
    # 2.8 GB with two of those in it is not a model — it is a download
    # that will resume inside the stage. Both halves are asked.
    partial = (any(folder.glob("blobs/*.incomplete"))
               if folder.is_dir() else False)
    # The AWQ checkpoint is ~7 GB; anything under a gigabyte is configs.
    ok = have > 1e9 and not partial
    return CheckResult(
        "s3-weights", ok, "optional",
        f"{model_id.split('/')[-1]}: {have / 1e9:.1f} GB cached"
        + ("" if ok else
           " (a fetch is unfinished)" if partial else " (weights missing)"),
        "" if ok else
        ("The first clip run will download ~7 GB inside S3 with no "
         "progress in the console. Pre-fetch it if you would rather not "
         "wait mid-run: python -c \"from huggingface_hub import "
         f"snapshot_download as d; d('{model_id}')\""))


def run_all(workspace_root: Path, *, disk_floor_gb: float = 50.0) -> list[CheckResult]:
    """Run every check; returns results in a stable order."""
    results: list[CheckResult] = [check_python()]
    results.append(check_ffmpeg())
    results.append(check_ffprobe())
    if results[-2].ok and results[-1].ok:
        results.extend(check_ffmpeg_capabilities())
    results.append(check_cuda())
    results.append(check_cudnn_dlls())
    results.append(check_torchcodec())
    results.append(check_hf_token())
    results.append(check_ranking_weights())
    results.extend(check_tools())
    results.append(check_disk(workspace_root, disk_floor_gb))
    results.append(check_workspace_writable(workspace_root))
    return results


def render(results: list[CheckResult]) -> tuple[str, bool]:
    """(report_text, all_required_ok) — CLI prints the text, exits on the bool."""
    lines: list[str] = []
    required_ok = True
    for r in results:
        mark = "PASS" if r.ok else ("WARN" if r.severity == "optional" else "FAIL")
        if not r.ok and r.severity == "required":
            required_ok = False
        lines.append(f"[{mark:4}] {r.name:14} {r.message}")
        if not r.ok and r.fix:
            for fix_line in r.fix.splitlines():
                # ASCII arrow on purpose: doctor output must survive piping
                # through cp1252 stdout (cmd redirects, scheduled tasks).
                lines.append(f"       {'':14} -> {fix_line}")
    verdict = "DOCTOR: all required checks passed" if required_ok else \
              "DOCTOR: required checks FAILED - fix the items above"
    lines.append(verdict)
    return "\n".join(lines), required_ok
