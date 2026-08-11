"""Mission control — BTA's product-grade local dashboard for the workspace.

``bta dashboard`` writes ``workspace/dashboard.html``: playable clip
previews, VL impact scores with per-dimension bars, the editor's copy pack
(title, hook, per-platform captions, hashtags) with one-click copy, the full
QA health table per clip, and status filters. One self-contained file —
videos are referenced relatively so previews play straight off disk, no
server, no network, no external assets.
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from clipforge.ffmpeg import find_binary
from clipforge.log import get_logger
from clipforge.paths import Workspace, atomic_write_text

log = get_logger(__name__)

_CSS = """
:root { --bg:#07090f; --bg2:#0b0e17; --card:#10141f; --edge:#1c2333;
  --edge2:#28324a; --text:#e6ebf5; --dim:#8b96ad; --ok:#3ddc84;
  --warn:#ffb454; --bad:#ff5370; --accent:#5ccfe6; --accent2:#c792ea; }
* { box-sizing:border-box; margin:0; padding:0; }
html { scroll-behavior:smooth; }
body { background:
  radial-gradient(1200px 500px at 15% -10%, rgba(92,207,230,.10), transparent 60%),
  radial-gradient(900px 420px at 95% -5%, rgba(199,146,234,.09), transparent 55%),
  var(--bg);
  color:var(--text); font:14px/1.55 'Segoe UI',system-ui,sans-serif;
  padding:32px 36px 60px; min-height:100vh; }
header { display:flex; align-items:baseline; gap:18px; flex-wrap:wrap; }
h1 { font-size:26px; letter-spacing:3px; text-transform:uppercase;
     font-weight:800; }
h1 em { font-style:normal; background:linear-gradient(90deg,var(--accent),var(--accent2));
        -webkit-background-clip:text; background-clip:text; color:transparent; }
.sub { color:var(--dim); font-size:12px; }
h2 { font-size:12px; letter-spacing:2px; text-transform:uppercase;
     color:var(--dim); margin:34px 0 14px; }
.stats { display:flex; gap:14px; flex-wrap:wrap; margin-top:22px; }
.stat { background:linear-gradient(180deg,var(--card),var(--bg2));
  border:1px solid var(--edge); border-radius:14px; padding:14px 22px;
  min-width:150px; }
.stat b { display:block; font-size:26px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.stat i { font-style:normal; color:var(--dim); font-size:10px;
  text-transform:uppercase; letter-spacing:1.5px; }
.pills { display:flex; gap:8px; margin:26px 0 6px; flex-wrap:wrap; }
.pill { border:1px solid var(--edge2); background:var(--card); color:var(--dim);
  border-radius:24px; padding:6px 18px; font-size:12px; cursor:pointer;
  letter-spacing:.5px; transition:.15s; }
.pill:hover { border-color:var(--accent); color:var(--text); }
.pill.on { background:linear-gradient(90deg,rgba(92,207,230,.18),rgba(199,146,234,.18));
  border-color:var(--accent); color:var(--text); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
  gap:20px; }
.card { background:linear-gradient(180deg,var(--card),var(--bg2));
  border:1px solid var(--edge); border-radius:16px; overflow:hidden;
  transition:transform .15s, border-color .15s; }
.card:hover { transform:translateY(-2px); border-color:var(--edge2); }
.media { position:relative; background:#000; }
.media video { width:100%; display:block; aspect-ratio:9/16; max-height:420px;
  object-fit:contain; background:#000; }
.score-ring { position:absolute; top:12px; right:12px; width:64px; height:64px;
  border-radius:50%; display:flex; align-items:center; justify-content:center;
  background:conic-gradient(var(--accent) calc(var(--p)*1%), #232b3f 0);
  box-shadow:0 4px 18px rgba(0,0,0,.55); }
.score-ring b { width:50px; height:50px; border-radius:50%; background:#0d1119;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  font-size:17px; }
.score-ring b i { font-style:normal; font-size:7px; color:var(--dim);
  letter-spacing:1px; }
.body { padding:14px 16px 16px; }
.badge { display:inline-block; padding:3px 12px; border-radius:20px;
  font-size:10px; font-weight:700; letter-spacing:1.2px; }
.pass { background:rgba(61,220,132,.12); color:var(--ok); border:1px solid var(--ok); }
.warn { background:rgba(255,180,84,.12); color:var(--warn); border:1px solid var(--warn); }
.fail { background:rgba(255,83,112,.12); color:var(--bad); border:1px solid var(--bad); }
.title { font-weight:700; font-size:15px; margin:10px 0 2px; }
.hook { color:var(--dim); font-size:12px; font-style:italic; }
.dims { margin:12px 0 4px; }
.dim { display:grid; grid-template-columns:110px 1fr 30px; gap:10px;
  align-items:center; font-size:11px; color:var(--dim); margin:4px 0; }
.bar { height:5px; border-radius:4px; background:#1a2133; overflow:hidden; }
.bar i { display:block; height:100%; border-radius:4px;
  background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.meta { display:flex; gap:14px; flex-wrap:wrap; color:var(--dim);
  font-size:11px; margin-top:10px; }
details { margin-top:10px; border-top:1px solid var(--edge); padding-top:8px; }
summary { cursor:pointer; font-size:11px; color:var(--dim);
  letter-spacing:1px; text-transform:uppercase; }
.qa-table { width:100%; border-collapse:collapse; font-size:11px; margin-top:8px; }
.qa-table td { padding:3px 6px; border-bottom:1px solid var(--edge); }
.qa-table .ok { color:var(--ok); } .qa-table .w { color:var(--warn); }
.qa-table .f { color:var(--bad); }
.copy { margin-top:8px; }
.copy .row { display:flex; gap:8px; align-items:flex-start; margin:6px 0; }
.copy .row span { flex:1; font-size:11px; color:var(--dim); word-break:break-word; }
.copy button { border:1px solid var(--edge2); background:#141a29; color:var(--accent);
  border-radius:6px; font-size:10px; padding:3px 10px; cursor:pointer; }
.copy button:hover { border-color:var(--accent); }
.just { font-size:11px; color:var(--dim); margin-top:8px; border-left:2px solid
  var(--accent); padding-left:10px; }
.timeline { background:linear-gradient(180deg,var(--card),var(--bg2));
  border:1px solid var(--edge); border-radius:14px; padding:18px 20px 14px; }
.track { position:relative; height:46px; background:#0d1220;
  border:1px solid var(--edge); border-radius:8px; overflow:hidden; }
.track .win { position:absolute; top:0; bottom:0; border-radius:6px;
  background:linear-gradient(180deg,rgba(92,207,230,.55),rgba(199,146,234,.35));
  border:1px solid var(--accent); display:flex; align-items:center;
  justify-content:center; font-size:10px; font-weight:700; color:#04070d;
  cursor:pointer; transition:.15s; overflow:hidden; }
.track .win:hover { filter:brightness(1.25); }
.track .win.rej { background:rgba(255,83,112,.35); border-color:var(--bad);
  color:var(--bad); }
.ticks { display:flex; justify-content:space-between; color:var(--dim);
  font-size:10px; margin-top:6px; letter-spacing:1px; }
.tx { font-size:11px; color:var(--dim); margin-top:8px; max-height:52px;
  overflow:hidden; line-height:1.45; }
.howto { background:linear-gradient(180deg,var(--card),var(--bg2));
  border:1px solid var(--edge); border-radius:16px; padding:18px 22px;
  font-size:13px; max-width:980px; }
.howto p { margin:7px 0; color:var(--dim); }
/* ---- hero + launcher (the layout a hosted dashboard uses, minus the
   upgrade banner, the credit meter and the watermark) ---- */
.hero { max-width:560px; margin:26px auto 6px; background:
  linear-gradient(180deg,rgba(19,24,38,.9),rgba(11,14,23,.9));
  border:1px solid var(--edge2); border-radius:18px; padding:18px;
  box-shadow:0 18px 60px rgba(0,0,0,.5); }
.hero label { display:block; font-size:10px; letter-spacing:1.6px;
  text-transform:uppercase; color:var(--dim); margin-bottom:8px; }
.hero input { width:100%; background:#0a0e18; border:1px solid var(--edge2);
  border-radius:10px; color:var(--text); padding:12px 14px; font-size:13px;
  font-family:inherit; }
.hero input:focus { outline:none; border-color:var(--accent); }
.hero .cta { width:100%; margin-top:10px; padding:13px; border-radius:10px;
  border:none; cursor:pointer; font-size:14px; font-weight:700;
  letter-spacing:.4px; color:#04070d;
  background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.hero .cta:hover { filter:brightness(1.1); }
.hero .out { margin-top:10px; font-family:ui-monospace,Consolas,monospace;
  font-size:11px; color:var(--accent); background:#0a0e18; padding:10px 12px;
  border-radius:8px; border:1px solid var(--edge); word-break:break-all;
  display:none; }
.launch { display:flex; gap:10px; flex-wrap:wrap; justify-content:center;
  margin:26px auto 8px; max-width:1100px; }
.feat { width:104px; text-align:center; position:relative; }
.feat .ic { width:52px; height:52px; margin:0 auto 7px; border-radius:15px;
  display:flex; align-items:center; justify-content:center; font-size:22px;
  background:linear-gradient(180deg,var(--card),var(--bg2));
  border:1px solid var(--edge2); transition:.15s; }
.feat:hover .ic { border-color:var(--accent); transform:translateY(-2px); }
.feat span { font-size:10.5px; color:var(--dim); line-height:1.3;
  display:block; }
.feat.off .ic { opacity:.32; } .feat.off span { opacity:.42; }
.tag { position:absolute; top:-5px; right:6px; font-size:8px;
  letter-spacing:.6px; padding:1px 6px; border-radius:8px; font-weight:700; }
.tag.live { background:var(--ok); color:#04220f; }
.tag.soon { background:#2b3450; color:var(--dim); }
code { background:#0d1220; border:1px solid var(--edge); padding:2px 7px;
  border-radius:5px; font-size:12px; color:var(--accent); }
footer { margin-top:40px; color:var(--dim); font-size:11px; }
"""

_JS = """
function flt(k, el) {
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('on'));
  el.classList.add('on');
  document.querySelectorAll('.card').forEach(c => {
    c.style.display = (k === 'all' || c.dataset.status === k) ? '' : 'none';
  });
}
function srt(key, el) {
  document.querySelectorAll('.sortpill').forEach(p => p.classList.remove('on'));
  el.classList.add('on');
  const grid = document.querySelector('.grid');
  const cards = Array.from(grid.children);
  cards.sort((a, b) => {
    const av = parseFloat(a.dataset[key] || 0), bv = parseFloat(b.dataset[key] || 0);
    return key === 'start' ? av - bv : bv - av;
  });
  cards.forEach(c => grid.appendChild(c));
}
function jump(id) {
  const c = document.getElementById(id);
  if (!c) return;
  c.scrollIntoView({block: 'center'});
  c.style.borderColor = 'var(--accent)';
  setTimeout(() => c.style.borderColor = '', 1400);
}
function mkcmd() {
  const v = document.getElementById('src').value.trim();
  const n = document.getElementById('nclips').value.trim() || '3';
  const out = document.getElementById('cmdout');
  if (!v) { out.style.display = 'none'; return; }
  // A static page cannot start a render, and pretending otherwise would be
  // a fake button. It writes the exact command instead — and copies it.
  const isUrl = /^https?:\/\//i.test(v);
  const cmd = isUrl
    ? `bta grab "${v}" --clips ${n}`
    : `bta process "${v}" --clips ${n}`;
  out.textContent = cmd;
  out.style.display = 'block';
  cp({textContent: ''}, cmd);
  out.textContent = cmd + '   ← copied, paste in your terminal';
}
function cp(btn, text) {
  const done = () => { btn.textContent = 'copied';
    setTimeout(() => btn.textContent = 'copy', 1200); };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done);
  } else {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove(); done();
  }
}
"""


def _thumb_b64(ffmpeg: str | None, clip: Path, at_s: float = 2.0) -> str | None:
    """Poster frame as base64, or None — the dashboard renders without
    posters on a machine with no ffmpeg; it must not crash there."""
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [str(ffmpeg), "-nostdin", "-v", "error", "-ss", f"{at_s}",
             "-i", str(clip), "-frames:v", "1", "-vf", "scale=270:-2",
             "-f", "mjpeg", "-"],
            capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return base64.b64encode(proc.stdout).decode("ascii")


def _load(dirpath: Path) -> list[dict[str, Any]]:
    out = []
    if dirpath.is_dir():
        for p in sorted(dirpath.glob("*.json"),
                        key=lambda q: q.stat().st_mtime, reverse=True):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return out


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _js_str(v: Any) -> str:
    return json.dumps(str(v))


def _generation_panel(ws: Workspace) -> str:
    """Generation Studio section: provider/quota state + pieces on disk.

    Reads the SAME ledger the router writes, so what it shows is what the
    next run will actually do — not a restatement of config. A provider
    with no key reads "not configured", never "ready".
    """
    import json as _json

    root = Path(ws.root)
    ledger_path = root / "genvideo_quota.json"
    providers: list[tuple[str, str, str]] = []
    try:
        blob = _json.loads(ledger_path.read_text(encoding="utf-8"))
        now = time.time()
        for name, st in sorted((blob.get("providers") or {}).items()):
            until = float(st.get("exhausted_until", 0.0) or 0.0)
            calls = int(st.get("calls", 0) or 0)
            secs = float(st.get("seconds_generated", 0.0) or 0.0)
            if until > now:
                mins = (until - now) / 60.0
                state, cls = f"metered out · {mins:.0f} min left", "soon"
            else:
                state, cls = "ready", "live"
            providers.append(
                (name, f'<span class="tag {cls}">{state}</span>',
                 f"{calls} call(s) · {secs:.0f}s generated"))
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - reporting never raises
        providers.append(("ledger", '<span class="tag soon">unreadable</span>',
                          str(ledger_path)))

    if not providers:
        providers = [("—", '<span class="tag soon">no runs yet</span>',
                      "run: bta generate \"your brief\"")]

    rows = "".join(
        f'<tr><td><b>{_esc(name)}</b></td><td>{tag}</td>'
        f'<td class="dim">{_esc(detail)}</td></tr>'
        for name, tag, detail in providers)

    pieces_dir = root / "generated"
    pieces: list[str] = []
    if pieces_dir.is_dir():
        for seq in sorted(pieces_dir.glob("*/sequence.mp4"),
                          key=lambda p: p.stat().st_mtime, reverse=True)[:6]:
            shots = len(list(seq.parent.glob("shot_*.mp4")))
            size = seq.stat().st_size / (1024 * 1024)
            pieces.append(
                f'<div class="card"><video controls preload="none" '
                f'src="{_esc(_rel(seq, root))}"></video>'
                f'<div class="meta"><b>{_esc(seq.parent.name)}</b>'
                f'<span class="dim">{shots} shot(s) · {size:.1f} MB</span>'
                f'</div></div>')
    pieces_html = ("".join(pieces) if pieces else
                   '<div class="dim" style="padding:14px">no generated '
                   'pieces yet</div>')

    return f"""
<section class="studio">
  <h2>Generation Studio <span class="dim">text → video</span></h2>
  <div class="dim" style="margin-bottom:10px">
    Documentary · storytelling · explainers · motion graphics. Shots are
    generated individually and cut together, because every current model
    loses coherence past a few seconds. The premium model runs until its
    quota is gone, the local open-source model takes over, and it switches
    back on its own when the window resets.
  </div>
  <table class="qa">{rows}</table>
  <div class="cards" style="margin-top:14px">{pieces_html}</div>
</section>"""


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_dashboard(ws: Workspace) -> Path:
    ffmpeg = find_binary("ffmpeg")
    clips_art = _load(ws.artifacts / "s6_render")
    qa_by_clip = {a["source_clip"]: a for a in _load(ws.artifacts / "s7_qa")}
    subs_by_key = {a["cache_key"]: a
                   for a in _load(ws.artifacts / "s5_subtitles")}
    cams_by_key = {a["cache_key"]: a
                   for a in _load(ws.artifacts / "s4_tracking")}
    ranks_by_key = {a["cache_key"]: a
                    for a in _load(ws.artifacts / "s3_semantic")}
    cands_by_key = {a["cache_key"]: a
                    for a in _load(ws.artifacts / "s2_prefilter")}
    editors = _load(ws.artifacts / "editor")

    cards: list[str] = []
    n_pass = n_warn = n_reject = 0
    vl_used = False
    #: (start_s, end_s, impact, status, card_id, label) for the source strip.
    spans: list[tuple[float, float, int, str, str, str]] = []
    source_dur = 0.0

    for art in clips_art:
        clip_path = Path(art["clip_path"])
        status = "pass"
        rejected_path = ws.clips / "rejected" / clip_path.name
        if not clip_path.exists() and rejected_path.exists():
            clip_path = rejected_path
        if not clip_path.exists():
            continue

        qa = qa_by_clip.get(art["cache_key"])
        verdict, badge = "NO QA", "warn"
        qa_rows = ""
        if qa is not None:
            if not qa["passed"]:
                verdict, badge, status = \
                    f"REJECTED · {qa['failed_count']} FAIL", "fail", "reject"
                n_reject += 1
            elif qa["warned_count"]:
                verdict = f"PASSED · {qa['warned_count']} WARN"
                badge, status = "warn", "warn"
                n_warn += 1
            else:
                verdict, badge = "QA PASSED", "pass"
                n_pass += 1
            qa_rows = "".join(
                f"<tr><td class=\"{'ok' if c['passed'] else ('f' if c['severity'] == 'fail' else 'w')}\">"
                f"{'✓' if c['passed'] else '✗'}</td>"
                f"<td>{_esc(c['name'])}</td><td>{_esc(c['measured'])}</td></tr>"
                for c in qa["checks"])

        # Chain joins: clip -> subs -> campath -> ranking -> candidates.
        sub = subs_by_key.get(art.get("source_subtitles", ""))
        cam = cams_by_key.get(sub["source_campath"]) if sub else None
        rank = ranks_by_key.get(cam["source_ranking"]) if cam else None
        cands = (cands_by_key.get(rank["source_candidates"])
                 if rank else None)

        # Candidate index by window match; editor copy pack by candidate_id.
        # EditorArtifact carries no source pointer (schema gap, recorded in
        # the ledger), so the join is candidate_id + recency — fine for one
        # workspace, honest about being a heuristic.
        item = None
        editor = None
        if cam and cands:
            starts = [(abs(float(c["start"]) - float(cam["clip_start"])), i)
                      for i, c in enumerate(cands.get("candidates", []))]
            if starts:
                delta, idx = min(starts)
                if delta <= 1.0:
                    cid = f"cand_{idx:03d}"
                    editor = next((e for e in editors
                                   if e.get("candidate_id") == cid), None)
                    if rank:
                        item = next((r for r in rank.get("items", [])
                                     if r.get("candidate_index") == idx),
                                    None)

        if rank and rank.get("ranking_source") == "semantic":
            vl_used = True

        score_html = ""
        dims_html = ""
        just_html = ""
        if item is not None:
            va = float(item.get("visual_action") or 0)
            hs = float(item.get("hook_strength") or 0)
            co = float(item.get("comprehensibility") or 0)
            impact = round((va + hs + co) / 3 * 10)
            score_html = (f'<div class="score-ring" style="--p:{impact}">'
                          f'<b>{impact}<i>IMPACT</i></b></div>')
            dims_html = '<div class="dims">' + "".join(
                f'<div class="dim"><span>{label}</span>'
                f'<div class="bar"><i style="width:{v * 10:.0f}%"></i></div>'
                f'<span>{v:.1f}</span></div>'
                for label, v in (("visual action", va), ("hook", hs),
                                 ("clarity", co))) + "</div>"
            if item.get("justification"):
                just_html = (f'<div class="just">'
                             f'{_esc(item["justification"])}</div>')

        title_html = ""
        copy_html = ""
        if editor:
            title_html = (f'<div class="title">{_esc(editor["title"])}</div>'
                          f'<div class="hook">hook: '
                          f'{_esc(editor["hook_text"])}</div>')
            rows = []
            for platform, caption in sorted(
                    editor.get("captions", {}).items()):
                rows.append(
                    f'<div class="row"><span><b>{_esc(platform)}</b> — '
                    f'{_esc(caption[:110])}…</span>'
                    f'<button onclick=\'cp(this,{_js_str(caption)})\'>'
                    f'copy</button></div>')
            tags = " ".join(editor.get("hashtags", []))
            if tags:
                rows.append(f'<div class="row"><span>{_esc(tags)}</span>'
                            f'<button onclick=\'cp(this,{_js_str(tags)})\'>'
                            f'copy</button></div>')
            copy_html = ('<details><summary>copy pack</summary>'
                         f'<div class="copy">{"".join(rows)}</div></details>')

        framing = ""
        if cam:
            framing = f'{cam["framing_mode"]} · {len(cam.get("frames", []))} fr'
        thumb = _thumb_b64(ffmpeg, clip_path)
        poster = f'poster="data:image/jpeg;base64,{thumb}"' if thumb else ""
        rel_src = clip_path.relative_to(ws.root).as_posix()

        # Source-position strip: WHERE in the original this clip came from.
        # A hosted tool shows you its picks; this shows you the picking.
        card_id = f"c{art['cache_key'][:10]}"
        start_s = float(cam["clip_start"]) if cam else 0.0
        end_s = float(cam["clip_end"]) if cam else start_s + art["duration_s"]
        impact_v = 0
        if item is not None:
            impact_v = round((float(item.get("visual_action") or 0)
                              + float(item.get("hook_strength") or 0)
                              + float(item.get("comprehensibility") or 0))
                             / 3 * 10)
        spans.append((start_s, end_s, impact_v, status, card_id,
                      f"{int(start_s // 60)}:{int(start_s % 60):02d}"))
        source_dur = max(source_dur, end_s)

        # Opening line of the clip — read it without playing it. The editor's
        # hook_text is exactly the first seconds of speech in this window.
        tx = ""
        if editor and editor.get("hook_text"):
            tx = f'<div class="tx">“{_esc(editor["hook_text"][:220])}”</div>'

        cards.append(f"""
<div class="card" id="{card_id}" data-status="{status}"
     data-impact="{impact_v}" data-start="{start_s:.1f}"
     data-dur="{art['duration_s']:.1f}">
  <div class="media">
    <video controls preload="none" {poster} src="{_esc(rel_src)}"></video>
    {score_html}
  </div>
  <div class="body">
    <span class="badge {badge}">{verdict}</span>
    {title_html}
    {tx}
    {dims_html}{just_html}
    <div class="meta">
      <span>{art['duration_s']:.1f}s</span>
      <span>{art['width']}x{art['height']}</span>
      <span>{_esc(art['encoder'])}</span>
      <span>{art['loudness_i']:.1f} LUFS</span>
      <span>{art['loudness_tp']:.1f} dBTP</span>
      <span>{_esc(framing)}</span>
    </div>
    {copy_html}
    <details><summary>qa health · {_esc(verdict)}</summary>
      <table class="qa-table">{qa_rows or
        '<tr><td>no QA artifact</td></tr>'}</table></details>
  </div>
</div>""")

    # Source strip: every rendered window drawn at its real position in the
    # original video. This is the view a hosted tool cannot give you — it
    # would expose which parts of your upload it ignored.
    timeline_html = ""
    if spans and source_dur > 0:
        wins = []
        for s, e, imp, st, cid, label in sorted(spans):
            left = 100.0 * s / source_dur
            width = max(1.2, 100.0 * (e - s) / source_dur)
            cls = "win rej" if st == "reject" else "win"
            wins.append(
                f'<div class="{cls}" style="left:{left:.2f}%;'
                f'width:{width:.2f}%" onclick="jump(\'{cid}\')" '
                f'title="{label} · impact {imp}">{label}</div>')
        mins = int(source_dur // 60)
        timeline_html = f"""
<h2>Source timeline · {mins}:{int(source_dur % 60):02d} analysed</h2>
<div class="timeline">
  <div class="track">{''.join(wins)}</div>
  <div class="ticks"><span>0:00</span><span>where these clips came
  from — click a block to jump to its card</span>
  <span>{mins}:{int(source_dur % 60):02d}</span></div>
</div>"""

    # Feature launcher. Every tile states its REAL status: "live" means it
    # ran in the pipeline that produced the clips below, "soon" means not
    # built. A launcher that advertises unbuilt features is a brochure.
    _FEATURES = [
        ("🎬", "Long → shorts", True), ("💬", "AI captions", True),
        ("🎯", "AI reframe", True), ("🔍", "Punch-ins", True),
        ("⚡", "Viral presets", True), ("🧠", "VL ranking", True),
        ("✅", "QA gate", True), ("📊", "Analytics", True),
        ("🎙️", "Enhance speech", True), ("🔊", "Auto SFX", False),
        ("🎞️", "AI B-roll", False), ("🗣️", "Voiceover", False),
        ("💎", "Upscale", False), ("🌍", "Dubbing", False),
    ]
    launcher_html = "".join(
        f'<div class="feat{"" if live else " off"}">'
        f'<span class="tag {"live" if live else "soon"}">'
        f'{"LIVE" if live else "SOON"}</span>'
        f'<div class="ic">{icon}</div><span>{_esc(name)}</span></div>'
        for icon, name, live in _FEATURES)

    studio_html = _generation_panel(ws)

    page = f"""<meta charset="utf-8">
<meta http-equiv="refresh" content="120">
<title>BTA — Mission Control</title>
<style>{_CSS}</style>
<header>
  <h1>B<em>T</em>A</h1>
  <div class="sub">beyond the average · mission control · local · open source
  · nothing leaves this machine</div>
</header>
<div class="sub">generated {time.strftime('%Y-%m-%d %H:%M:%S')} ·
workspace {_esc(ws.root)}</div>

<div class="stats">
  <div class="stat"><b>{len(cards)}</b><i>clips rendered</i></div>
  <div class="stat"><b>{n_pass + n_warn}</b><i>shipped (QA)</i></div>
  <div class="stat"><b>{n_reject}</b><i>quarantined</i></div>
  <div class="stat"><b>{'VL' if vl_used else 'heuristic'}</b><i>ranking</i></div>
  <div class="stat"><b>∞</b><i>credits</i></div>
  <div class="stat"><b>none</b><i>watermark</i></div>
</div>

<div class="hero">
  <label>source · local file or url</label>
  <input id="src" placeholder="D:\\video.mp4   or   https://youtu.be/..."
         onkeydown="if(event.key==='Enter')mkcmd()">
  <div style="display:flex;gap:10px;margin-top:10px;align-items:center">
    <label style="margin:0;flex:0 0 auto">clips</label>
    <input id="nclips" value="3" style="width:70px;padding:8px 10px">
  </div>
  <button class="cta" onclick="mkcmd()">Build my command →</button>
  <div class="out" id="cmdout"></div>
</div>

<div class="launch">{launcher_html}</div>

{studio_html}

{timeline_html}

<div class="pills">
  <button class="pill on" onclick="flt('all',this)">all</button>
  <button class="pill" onclick="flt('pass',this)">passed</button>
  <button class="pill" onclick="flt('warn',this)">warnings</button>
  <button class="pill" onclick="flt('reject',this)">rejected</button>
  <span style="flex:1"></span>
  <button class="pill sortpill on" onclick="srt('impact',this)">sort: impact</button>
  <button class="pill sortpill" onclick="srt('start',this)">source order</button>
  <button class="pill sortpill" onclick="srt('dur',this)">longest</button>
</div>

<div class="grid">{''.join(cards) if cards else
  '<p class="sub">no clips yet — run <code>bta process</code></p>'}</div>

<h2>Run it yourself</h2>
<div class="howto">
  <p><code>bta process video.mp4 --clips 3</code> — transcribe → score
  → VL-rank → track (shot-aware, face-centred) → karaoke subtitles + hook →
  9:16 render with progress bar → mechanical QA. Passing clips land in
  <code>workspace/clips/</code>; failures are quarantined in
  <code>clips/rejected/</code> with the measured reason.</p>
  <p><code>bta watch</code> — record authorized channels continuously.
  Files stay local; publishing is draft-only behind a human gate.</p>
  <p><code>bta dashboard</code> — regenerate this page.</p>
  <p><code>pytest tests/unit tests/integration</code> +
  <code>bta verify all</code> — the two-command gate. Run both.</p>
</div>
<footer>BTA · Beyond The Average — deterministic, resumable, honest about every
fallback. Rejected clips stay quarantined with the exact measurement that
failed. No account, no credits, no upload.</footer>
<script>{_JS}</script>
"""
    dest = ws.root / "dashboard.html"
    atomic_write_text(dest, page)
    log.info("dashboard.written", path=str(dest), clips=len(cards))
    return dest
