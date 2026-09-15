# BTA — Beyond The Average

Turns long video into render-ready 9:16 shorts, and turns text briefs into
video, **entirely on your own machine**. No credits, no watermark, no
uploads. Every model is open-source and runs locally.

Built and verified on Windows 11 / RTX 3090 (24 GB).

---

## Quick start

First time on a machine, from `D:\clipforge`:

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

That is the CPU-side install: the CLI, the dashboard and everything that
does not need a GPU. It is enough to run `doctor`, browse the Library and
serve clips. The model stack is a separate, much larger step — `torch`
from the CUDA wheel index **first**, then the rest of
`requirements.txt`, in that order, because several of those packages will
otherwise resolve torch themselves and quietly replace a working CUDA
build with a CPU one. The file says so at the top, at length, with the
three times it has happened here.

Everything then runs through `bta`:

```bash
.\.venv\Scripts\bta.exe doctor
```

That checks ffmpeg, CUDA, disk and models before you burn a long run on a
broken prerequisite. Fix anything it flags, then:

```bash
.\.venv\Scripts\bta.exe web --port 8011
```

Open <http://127.0.0.1:8011/> — that's the dashboard, and it drives
everything below without touching a terminal again.

---

## The dashboard

| Tab | What it does |
|---|---|
| **Generate** | Text brief → video. Pick a style, write a brief, hit ↑ |
| **Clip** | Long video or URL → vertical shorts, scored and QA-gated |
| **Library** | Everything produced, with a score breakdown per clip |
| | Hover a clip (tap on phone) for Details, Download, Delete |
| **Studio** | What this machine can actually do, probed live |
| **Activity** | Running jobs with streaming logs and a cancel button |

### From your phone

```bash
.\.venv\Scripts\bta.exe web --lan --port 8011
```

`--lan` binds every interface **and turns on the access token**, because
this API starts pipeline runs on this machine and an open port that does
that is a remote shell with a nicer front end. Loopback stays open, so
nothing changes for the browser on the PC itself.

The banner then prints the addresses that actually work, with the token
already in the link — open one on the phone and it is paired. If you
would rather not paste a 43-character token into a phone keyboard, open
the bare address instead and use **Connect a device** in the sidebar: it
shows six digits that are good for one exchange, for ten minutes, and
die after five wrong guesses.

If it does not load at all, Windows Firewall is blocking the port. In an
**Administrator** PowerShell:

```bash
New-NetFirewallRule -DisplayName "BTA Studio 8011" -Direction Inbound -Protocol TCP -LocalPort 8011 -Action Allow -Profile Private,Public
```

Reaching the dashboard by a *name* rather than an address (an mDNS alias,
a hosts-file entry) needs that name listed, or the server answers 421:

```bash
set BTA_WEB_ALLOWED_HOSTS=studio.lan
```

That check exists because a web page can point its own DNS name at
127.0.0.1 and reach a localhost server from your browser as though it
owned it. Addresses and tailnet names need no listing.

### From anywhere

```bash
.\.venv\Scripts\bta.exe web --tunnel cloudflare --port 8011
```

A public https URL, no VPN, token enforced — including for the tunnel's
own traffic, which arrives looking like it came from this machine. Stop
the server when you are done: the URL is on the public internet and
anyone holding the token can drive this machine.

Tailscale, if you have it, is strictly safer and the banner offers it
first — a tailnet address works from any device on your account, over an
encrypted link, with nothing exposed publicly.

> `--insecure-no-auth` exists and does what it says: no token, every
> device on the network can run this pipeline. It refuses to combine with
> `--tunnel`.

---

## Command line

### Clip a video

```bash
.\.venv\Scripts\bta.exe process "D:\path\to\video.mp4" --clips 3
```

```bash
.\.venv\Scripts\bta.exe grab "https://youtu.be/VIDEO_ID" --clips 3
```

Useful flags:

| Flag | Effect |
|---|---|
| `--niche dark_mindset` | Apply a whole look: captions, grade, pacing |
| `--jumpcut` / `--no-jumpcut` | Cut silences between words |
| `--enhance gentle` | Speech cleanup: `off`, `gentle`, `strong` |
| `--broll` | Generate B-roll locally and cut it into the pauses |
| `--manifest out.json` | Machine-readable result, for automation |

**Leave `--jumpcut` off for music.** Silence removal cuts between bars and
wrecks musical timing. It is off by default for exactly that reason.

### Generate video from text

```bash
.\.venv\Scripts\bta.exe generate "A lone fishing boat cuts through dark morning water" --preset cinematic_doc --shots 4
```

Add `--clip` to run the result straight through the clipper.

### Niches

A niche carries the entire look — generation style, colour grade, caption
styling, pacing — so picking one is usually the only decision you need.

```bash
.\.venv\Scripts\bta.exe swarm niches
```

- **dark_mindset** — monochrome, slow, small quiet centred text
- **viral_clips** — loud bottom-third karaoke captions, fast cuts
- **cinematic_doc** — filmic grade, restrained captions, patient cutting

### Autonomous mode

```bash
.\.venv\Scripts\bta.exe swarm serve
```

Runs a durable task board with five roles (plan → generate/clip →
critique → package). Work survives a crash, and a dead worker's task
returns to the pool on its own. Queue work without running it:

```bash
.\.venv\Scripts\bta.exe swarm plan --brief "your idea" --niche dark_mindset
```

```bash
.\.venv\Scripts\bta.exe swarm status
```

### Always-on recording

```bash
.\.venv\Scripts\bta.exe watch
```

Records the channels in `config/channels.toml` and clips each window as it
lands. **Only list channels you are authorised to record.**

---

## What runs locally

| Job | Model | Note |
|---|---|---|
| Transcription | WhisperX / faster-whisper | word-level timestamps |
| Clip ranking | Qwen2.5-VL (4-bit) | scores hook, action, clarity |
| Speaker tracking | YOLO11-pose + MediaPipe | picks the talker by lip motion |
| Video generation | LTX-Video, Wan 2.2 TI2V-5B | auto-selected per style |
| Voiceover | Kokoro-82M | neural, ~3.8× realtime on CPU |
| Render / audio | ffmpeg + libass | captions, grade, loudness |

Model selection is automatic — photoreal and human-subject work routes to
Wan, atmospheric and fast work to LTX. Override with `--model`.

---

## Checking your setup

```bash
.\.venv\Scripts\bta.exe verify all
```

The Studio tab shows the same thing visually. Each capability is
**probed**, not declared, so a feature never appears available when the
model or ffmpeg filter behind it is missing.

Run the test suite with:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

It passes on a machine with **no GPU at all** — the tests that need
CUDA, the generation weights or a second interpreter skip and say which,
so a red run means something is broken rather than something is missing.
That is what CI runs (`.github/workflows/tests.yml`): Python, ffmpeg,
`pip install -e .`.

To deselect the heavy ones explicitly:

```bash
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu and not network"
```

---

## Publishing

The pipeline **produces files and stops**. It does not post anywhere.

Every clip that passes QA gets an export pack beside it
(`<clip>.export.json`): the caption, hashtags derived from that clip's own
words, a thumbnail frame, chapters, and per-platform text already fitted
to each platform's character limit. Copy, paste, post.

Auto-posting is deliberately not built — it needs platform credentials and
acts outside this machine on your behalf, which should stay a decision you
make per clip.

---

## Known limits

Stated plainly so none of it surprises you mid-run:

- **Dubbed audio** is not built. Translated *subtitles* work (Whisper
  translates into English); dubbed *audio* needs a translation model and a
  multilingual voice, neither installed.
- **Upscaling is resampling**, not learned super-resolution. It resizes
  and sharpens well; it cannot invent detail the source lacks.
- **NVENC is unavailable** on NVIDIA drivers below 610, so renders fall
  back to libx264 — roughly 5× slower, same quality. Updating the driver
  fixes it.
- **Diarization** (labelling *which* speaker) needs a Hugging Face token
  with the pyannote terms accepted. Without it transcripts still work;
  only speaker labels are missing.
- **Generation resolution is capped by measurement**, not preference.
  Video models return *blank frames* rather than errors past their trained
  envelope, so generation stays inside a verified budget and upscales to
  delivery size afterwards.

---

## Where things live

```
workspace/
  clips/          finished shorts (+ .export.json, .thumb.jpg)
    rejected/     clips that failed QA, kept for inspection
  generated/      text-to-video pieces
  artifacts/      per-stage cache — safe to delete, costs a re-run
  trash/          clips deleted from the Library — drag one back to restore
  models/         downloaded weights
  logs/           structured JSONL
  state.sqlite3   job and stage history
config/
  config.toml     all settings
  channels.toml   channels `watch` is allowed to record
```

Nothing in `workspace/` must be kept — deleting it costs recomputation,
not correctness.
