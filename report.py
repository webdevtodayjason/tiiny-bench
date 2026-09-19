#!/usr/bin/env python3
"""Builds report.html from everything in bench-results/.

One self-contained file. No CDN, no fonts fetched, no JavaScript needed to read
it - the charts are SVG emitted here, so the report works from a USB stick and
will still work in ten years.

Design notes, since they were deliberate:

  The whole page is set in monospace. That is not laziness. Every figure on it
  is a measurement, and a measurement wants a column it lines up in; the Tiiny
  wordmark is a pixel face and Titanium's mark is an element tile, so the
  technical register is already the brand's. Hierarchy comes from size, weight
  and colour instead of from a second typeface, which also keeps the file free
  of any webfont request.

  Colours are lifted from the two marks that appear on it: near-black ground
  and brushed steel from the Titanium badge, its copper accent for the primary
  series, and one desaturated blue as the second series so the comparisons stay
  legible to a colourblind reader. Nothing else gets a hue.
"""
import base64
import hashlib
import html
import json
import math
import mimetypes
import pathlib


def data_uri(path: pathlib.Path) -> str:
    """An asset inlined, or an empty string if it is not there. Keeps the
    report one file, which is the whole point of it."""
    if not path.exists():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

# ---------------------------------------------------------------- palette
C = {
    "ground":  "#0b0c0e",
    "s1":      "#131518",
    "s2":      "#1b1e23",
    "line":    "#272b31",
    "steel":   "#8a8f98",
    "ink":     "#e8e6e3",
    "copper":  "#d97a2b",
    "hot":     "#ffb968",
    "cool":    "#6fb3c9",
    "good":    "#7fb069",
}
SERIES = [C["copper"], C["cool"], C["hot"], "#9b8ec4", "#c9a227", "#6f9c8f"]


# ------------------------------------------------------------------ load
def load(outdir: pathlib.Path):
    """Every result file, old shape and new, as one flat list of runs.

    v1 files carry a single model at the top level; v2 files carry a list under
    `models`. Both are real data and the report shows both rather than quietly
    dropping the history."""
    runs = []
    for f in sorted(outdir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        base = {"label": d.get("label") or f.stem, "stamp": d.get("stamp") or f.stem[:15],
                "build": d.get("build"), "file": f.name}
        if "models" in d:
            for m in d["models"]:
                if m.get("results"):
                    runs.append({**base, **m})
        elif d.get("results"):
            runs.append({**base, "model": d.get("model"), "results": d["results"],
                         "npu_mem_total_mb": d.get("npu_mem_total_mb")})
    return runs


def short(model: str) -> str:
    return (model or "?").split("/")[-1]


# ------------------------------------------------------------------- svg
def _scale(v, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) * (b - a) / (hi - lo)


def line_chart(series, xlabel, ylabel, w=620, h=260, logx=False, fmt="{:.0f}"):
    """series: [(name, [(x, y), ...], colour)]. Points labelled directly on the
    line - a legend makes the eye travel for no reason when there is room."""
    pad_l, pad_r, pad_t, pad_b = 58, 96, 22, 38
    pts = [p for _, ps, _ in series for p in ps]
    if not pts:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0, max(ys) * 1.16 or 1
    if logx:
        x0, x1 = math.log10(max(x0, 1)), math.log10(max(x1, 10))

    def px(x):
        return _scale(math.log10(max(x, 1)) if logx else x, x0, x1, pad_l, w - pad_r)

    def py(y):
        return _scale(y, y0, y1, h - pad_b, pad_t)

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
           f'aria-label="{html.escape(ylabel)} against {html.escape(xlabel)}">']
    # Horizontal reference lines only. Vertical gridlines on a four-point
    # series are pure decoration.
    for i in range(4):
        y = y0 + (y1 - y0) * i / 3
        out.append(f'<line x1="{pad_l}" y1="{py(y):.1f}" x2="{w - pad_r}" y2="{py(y):.1f}" '
                   f'stroke="{C["line"]}" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 10}" y="{py(y) + 4:.1f}" text-anchor="end" '
                   f'class="tick">{fmt.format(y)}</text>')
    for name, ps, col in series:
        ps = sorted(ps)
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(ps))
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.25" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in ps:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" fill="{C["ground"]}" '
                       f'stroke="{col}" stroke-width="2"/>')
        lx, ly = ps[-1]
        out.append(f'<text x="{px(lx) + 12:.1f}" y="{py(ly) + 4:.1f}" class="lbl" '
                   f'fill="{col}">{html.escape(name)}</text>')
    for x in sorted({p[0] for p in pts}):
        out.append(f'<text x="{px(x):.1f}" y="{h - pad_b + 20:.1f}" text-anchor="middle" '
                   f'class="tick">{x:,.0f}</text>')
    out.append(f'<text x="{pad_l}" y="{h - 6}" class="axis">{html.escape(xlabel)}</text>')
    out.append(f'<text x="{pad_l}" y="14" class="axis">{html.escape(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


def bars(rows, ylabel, w=620, h=240, fmt="{:.0f}", colour=None):
    """rows: [(label, value, note)]. Zero-based, always - a truncated bar axis
    is the oldest way to make a number look like something it is not."""
    if not rows:
        return ""
    pad_l, pad_r, pad_t, pad_b = 58, 18, 26, 44
    vmax = max(v for _, v, _ in rows) * 1.18 or 1
    # Cap the slot width. Two bars across 880px reads as a billboard, not as a
    # measurement, and the eye stops comparing heights and starts comparing area.
    span = w - pad_l - pad_r
    bw = min(span / len(rows), 132)
    span_used = bw * len(rows)
    pad_l += (span - span_used) / 2
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
           f'aria-label="{html.escape(ylabel)}">']
    for i in range(4):
        y = vmax * i / 3
        yy = _scale(y, 0, vmax, h - pad_b, pad_t)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" '
                   f'stroke="{C["line"]}" stroke-width="1"/>')
        out.append(f'<text x="{pad_l - 10}" y="{yy + 4:.1f}" text-anchor="end" '
                   f'class="tick">{fmt.format(y)}</text>')
    for i, (lab, v, note) in enumerate(rows):
        col = colour or SERIES[i % len(SERIES)]
        x = pad_l + i * bw + bw * 0.18
        bwid = bw * 0.64
        yy = _scale(v, 0, vmax, h - pad_b, pad_t)
        out.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bwid:.1f}" '
                   f'height="{h - pad_b - yy:.1f}" fill="{col}" opacity=".88" rx="2"/>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{yy - 8:.1f}" text-anchor="middle" '
                   f'class="val">{fmt.format(v)}</text>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{h - pad_b + 19:.1f}" '
                   f'text-anchor="middle" class="tick">{html.escape(str(lab))}</text>')
        if note:
            out.append(f'<text x="{x + bwid / 2:.1f}" y="{h - pad_b + 34:.1f}" '
                       f'text-anchor="middle" class="tick dim">{html.escape(note)}</text>')
    out.append(f'<text x="{pad_l}" y="14" class="axis">{html.escape(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------ colour per model
# A model has to be the same colour in every chart on the page, so the colour
# comes from the id rather than from its position in a list: adding a model,
# or one dropping out of a sweep, must not repaint the others. The hue is a
# hash, and the saturation and lightness are fixed at values that stay legible
# on the near-black ground and stay apart from each other in greyscale print.
HUE_STEPS = 30


def _hue_of(model):
    h = hashlib.sha1((model or "?").encode("utf-8")).digest()
    return (h[0] << 8 | h[1]) % HUE_STEPS


def model_colours(models):
    """{model: css colour}, stable per id and nudged apart when two collide.

    Two models landing on the same hue inside one report is confusing even
    though it is rare, so a collision walks to the next free slot. The walk is
    deterministic in sorted id order, so the same set of models always gets the
    same answer.
    """
    taken, out = {}, {}
    for m in sorted(models or []):
        slot = _hue_of(m)
        # A collision walks by a stride coprime with the wheel rather than to
        # the next slot along. Two models nudged into neighbouring hues are
        # 15 degrees apart and indistinguishable, which is worse than the
        # collision was; a stride of 7 lands the loser on the far side and
        # still visits every slot before giving up.
        for step in range(HUE_STEPS):
            cand = (slot + step * 7) % HUE_STEPS
            if cand not in taken:
                slot = cand
                break
        taken[slot] = m
        deg = slot * (360 / HUE_STEPS)
        # Warm hues read brighter than cool ones at the same lightness, so the
        # blues and violets are lifted a little to keep the set even. Alternate
        # slots are lightened again: a dozen models on one wheel puts some of
        # them 12 degrees apart, and a difference in value separates those two
        # when the difference in hue no longer can, including in greyscale.
        lift = 8 if 200 <= deg <= 300 else 0
        out[m] = "hsl(%.0f 62%% %d%%)" % (deg, 56 + lift + (12 if slot % 2 else 0))
    return out


def legend(models, colours, title=""):
    chips = "".join(
        f'<span class="lg"><i style="background:{colours[m]}"></i>'
        f'{html.escape(short(m))}</span>' for m in sorted(models))
    head = f'<div class="lgt">{html.escape(title)}</div>' if title else ""
    return f'<div class="legend">{head}{chips}</div>'


# ----------------------------------------------------------- new charts
def dual_axis(points, w=660, h=290):
    """Time to first token and decode rate against prompt length, on one chart.

    Prefill rate alone hides the thing a person actually feels, which is how
    long they wait before the first word appears. Both series share an x axis
    and get their own y axis: TTFT solid on the left, decode dashed on the
    right, every point labelled on both.
    """
    pts = sorted(points, key=lambda p: p["prompt_tokens"])
    if len(pts) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 62, 88, 26, 46
    xs = [p["prompt_tokens"] for p in pts]
    lo, hi = math.log10(max(min(xs), 1)), math.log10(max(max(xs), 10))
    ttft = [p.get("ttft_s") or 0 for p in pts]
    dec = [p.get("decode_tok_s") or 0 for p in pts]
    t1 = max(ttft) * 1.28 or 1
    d1 = max(dec) * 1.28 or 1

    def px(x):
        return _scale(math.log10(max(x, 1)), lo, hi, pad_l, w - pad_r)

    def pl(y):
        return _scale(y, 0, t1, h - pad_b, pad_t)

    def pr(y):
        return _scale(y, 0, d1, h - pad_b, pad_t)

    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="'
         f'time to first token and decode rate against prompt length">']
    for i in range(4):
        y = t1 * i / 3
        yy = pl(y)
        o.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" '
                 f'stroke="{C["line"]}" stroke-width="1"/>')
        o.append(f'<text x="{pad_l-9}" y="{yy+4:.1f}" text-anchor="end" class="tick" '
                 f'fill="{C["copper"]}">{y:.2f}s</text>')
        o.append(f'<text x="{w-pad_r+22}" y="{pr(d1*i/3)+4:.1f}" class="tick" '
                 f'fill="{C["cool"]}">{d1*i/3:.0f}</text>')
    dt = " ".join(f"{'M' if i == 0 else 'L'}{px(p['prompt_tokens']):.1f},"
                  f"{pl(p.get('ttft_s') or 0):.1f}" for i, p in enumerate(pts))
    dd = " ".join(f"{'M' if i == 0 else 'L'}{px(p['prompt_tokens']):.1f},"
                  f"{pr(p.get('decode_tok_s') or 0):.1f}" for i, p in enumerate(pts))
    o.append(f'<path d="{dd}" fill="none" stroke="{C["cool"]}" stroke-width="2" '
             f'stroke-dasharray="6 4" stroke-linejoin="round"/>')
    o.append(f'<path d="{dt}" fill="none" stroke="{C["copper"]}" stroke-width="2.4" '
             f'stroke-linejoin="round"/>')
    for p in pts:
        x = px(p["prompt_tokens"])
        yt, yd = pl(p.get("ttft_s") or 0), pr(p.get("decode_tok_s") or 0)
        o.append(f'<circle cx="{x:.1f}" cy="{yd:.1f}" r="3.2" fill="{C["ground"]}" '
                 f'stroke="{C["cool"]}" stroke-width="2"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{yt:.1f}" r="3.4" fill="{C["ground"]}" '
                 f'stroke="{C["copper"]}" stroke-width="2"/>')
        o.append(f'<text x="{x:.1f}" y="{yt-10:.1f}" text-anchor="middle" class="val" '
                 f'fill="{C["copper"]}">{p.get("ttft_s") or 0:.2f}s</text>')
        o.append(f'<text x="{x:.1f}" y="{yd+18:.1f}" text-anchor="middle" class="val" '
                 f'fill="{C["cool"]}">{p.get("decode_tok_s") or 0:.0f}</text>')
        o.append(f'<text x="{x:.1f}" y="{h-pad_b+20:.1f}" text-anchor="middle" '
                 f'class="tick">{p["prompt_tokens"]:,}</text>')
    o.append(f'<text x="{pad_l}" y="14" class="axis" fill="{C["copper"]}">'
             f'time to first token</text>')
    o.append(f'<text x="{w-pad_r}" y="14" text-anchor="end" class="axis" '
             f'fill="{C["cool"]}">decode tok/s (dashed)</text>')
    o.append(f'<text x="{pad_l}" y="{h-6}" class="axis">prompt tokens</text>')
    o.append("</svg>")
    return "".join(o)


def grouped_bars(groups, series, w=660, h=280, fmt="{:.0f}"):
    """groups: [(label, [v1, v2], note)]. series: [(name, colour), ...].

    Aggregate against per-stream at each concurrency level, side by side, so
    the gap between them is the thing you see rather than something you have to
    hold in your head across two charts.
    """
    if not groups:
        return ""
    pad_l, pad_r, pad_t, pad_b = 58, 18, 30, 52
    vmax = max(max(vs) for _, vs, _ in groups) * 1.22 or 1
    span = w - pad_l - pad_r
    gw = min(span / len(groups), 150)
    pad_l += (span - gw * len(groups)) / 2
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="'
         f'{html.escape(" and ".join(n for n, _ in series))} at each level">']
    for i in range(4):
        y = vmax * i / 3
        yy = _scale(y, 0, vmax, h - pad_b, pad_t)
        o.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" '
                 f'stroke="{C["line"]}" stroke-width="1"/>')
        o.append(f'<text x="{pad_l-9}" y="{yy+4:.1f}" text-anchor="end" '
                 f'class="tick">{fmt.format(y)}</text>')
    n = len(series)
    for gi, (lab, vs, note) in enumerate(groups):
        base = pad_l + gi * gw
        bw = gw * 0.66 / n
        for si, v in enumerate(vs):
            x = base + gw * 0.17 + si * bw
            yy = _scale(v, 0, vmax, h - pad_b, pad_t)
            o.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw*0.86:.1f}" '
                     f'height="{h-pad_b-yy:.1f}" fill="{series[si][1]}" '
                     f'opacity=".9" rx="2"/>')
            o.append(f'<text x="{x+bw*0.43:.1f}" y="{yy-7:.1f}" text-anchor="middle" '
                     f'class="val">{fmt.format(v)}</text>')
        o.append(f'<text x="{base+gw/2:.1f}" y="{h-pad_b+19:.1f}" text-anchor="middle" '
                 f'class="tick">{html.escape(str(lab))}</text>')
        if note:
            o.append(f'<text x="{base+gw/2:.1f}" y="{h-pad_b+34:.1f}" '
                     f'text-anchor="middle" class="tick dim">{html.escape(note)}</text>')
    lx = pad_l
    for name, col in series:
        o.append(f'<rect x="{lx:.1f}" y="{pad_t-22}" width="9" height="9" fill="{col}" rx="1"/>')
        o.append(f'<text x="{lx+14:.1f}" y="{pad_t-14}" class="tick">{html.escape(name)}</text>')
        lx += 20 + len(name) * 6.6
    o.append("</svg>")
    return "".join(o)


def hbars(rows, w=660, unit="", fmt="{:.1f}", colours=None, rowh=30):
    """rows: [(label, value, note)], drawn along the y axis.

    Long model names read fine here and are unreadable rotated under a vertical
    bar, which is the whole reason this exists: sixteen names under sixteen
    columns is a smear.

    The value and its note are drawn one after the other at the end of the bar
    rather than at opposite ends of the row, and the bars are scaled to leave
    room for the longest of them. Putting the note hard right let a long value
    label run straight into it, which is the same unreadable smear moved.
    """
    if not rows:
        return ""
    pad_l, pad_t, pad_b = 8, 10, 20
    h = pad_t + pad_b + rowh * len(rows)
    label_w = max(96, min(200, 8 + 7.0 * max(len(str(l)) for l, _, _ in rows)))
    # Monospace, so a character really is a fixed width and this arithmetic
    # holds rather than approximating.
    ch = 6.7
    tail = max(len(fmt.format(v) + unit + ("  " + n if n else ""))
               for _, v, n in rows) * ch + 16
    x0 = pad_l + label_w
    track = max(60, w - x0 - tail)
    vmax = max(v for _, v, _ in rows) or 1
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="{html.escape(unit.strip() or "comparison")}">']
    for i, (lab, v, note) in enumerate(rows):
        y = pad_t + i * rowh
        col = (colours or {}).get(lab) or C["copper"]
        bw = max(_scale(v, 0, vmax, 0, track), 1)
        o.append(f'<text x="{pad_l}" y="{y+rowh*0.64:.1f}" class="tick hlab">'
                 f'{html.escape(str(lab))}</text>')
        o.append(f'<rect x="{x0:.1f}" y="{y+rowh*0.2:.1f}" width="{bw:.1f}" '
                 f'height="{rowh*0.56:.1f}" fill="{col}" opacity=".9" rx="2"/>')
        o.append(f'<text x="{x0+bw+8:.1f}" y="{y+rowh*0.64:.1f}" class="val">'
                 f'{fmt.format(v)}{html.escape(unit)}'
                 + (f'<tspan class="dim">  {html.escape(note)}</tspan>' if note else "")
                 + '</text>')
    o.append("</svg>")
    return "".join(o)


def chips(items):
    """A strip of small labelled facts. Anything missing is left out rather
    than drawn as a dash, because an empty chip reads as a measurement of
    nothing."""
    live = [(k, v) for k, v in items if v not in (None, "", "None")]
    if not live:
        return ""
    return ('<div class="chips">' + "".join(
        f'<span class="chip"><b>{html.escape(str(v))}</b>'
        f'<span>{html.escape(k)}</span></span>' for k, v in live) + "</div>")


# --------------------------------------------------------------- sections
# Which endpoint each test's numbers came off. Printed under a model so a
# reader can go and reproduce it against their own box rather than take this
# page's word for anything.
ENDPOINT = {
    "prefill": "/v1/chat/completions",
    "sustained": "/v1/chat/completions",
    "concurrency": "/v1/chat/completions",
    "thinking": "/v1/chat/completions",
    "image": "/v1/image/generate",
    "speech": "/v1/audio/speech",
    "embed": "/v1/embeddings",
    "asr": "/v1/audio/transcriptions",
    "ocr": "/v1/ocr",
    "music": "/v1/music/generate",
    "rerank": "/v1/rerank",
}


def stat(value, label, note=""):
    n = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="stat"><div class="v">{value}</div>'
            f'<div class="k">{html.escape(label)}</div>{n}</div>')


def caption(text):
    """A sentence computed from this model's own numbers, never a stock line."""
    return f'<p class="cap">{text}</p>'


def points_table(pf):
    head = ("<tr><th>prompt tokens</th><th class=num>time to first token</th>"
            "<th class=num>prefill tok/s</th><th class=num>decode tok/s</th></tr>")
    body = "".join(
        f'<tr><td class="num">{p["prompt_tokens"]:,}</td>'
        f'<td class="num">{(p.get("ttft_s") or 0):.2f} s</td>'
        f'<td class="num">{(p.get("prefill_tok_s") or 0):,.0f}</td>'
        f'<td class="num">{(p.get("decode_tok_s") or 0):.1f}</td></tr>'
        for p in sorted(pf, key=lambda p: p["prompt_tokens"]))
    return f'<div class="tablewrap"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'


def model_section(run, idx, colour=None):
    r = run.get("results") or {}
    m = run.get("model") or "?"
    col = colour or C["copper"]
    parts = [f'<section class="model" id="m{idx}">']

    # --- header: what it is, then how it was run -------------------------
    pill = []
    if run.get("type"):
        pill.append(run["type"])
    if run.get("params"):
        pill.append(f'{run["params"]} params')
    if run.get("npu_usage"):
        pill.append(f'{run["npu_usage"]} of 100 NPU units')
    if run.get("total_size"):
        pill.append(f'{run["total_size"] / 1e9:.1f} GB on disk')
    meta = [f'run {run["label"]}', run["stamp"]]
    if run.get("build"):
        meta.append(f'TiinyOS {run["build"]}')
    if run.get("elapsed_s"):
        meta.append(f'{run["elapsed_s"]:.0f}s of measurement')
    if (r.get("thinking") or {}).get("on"):
        meta.append("measured with reasoning off and on")
    seen = [ENDPOINT[k] for k in r if k in ENDPOINT]
    src = sorted(set(seen))
    parts.append(
        f'<header class="mhead"><div class="mtop">'
        f'<span class="dot" style="background:{col}"></span>'
        f'<h3>{html.escape(short(m))}</h3></div>'
        f'<div class="pills">'
        + "".join(f'<span class="pill">{html.escape(x)}</span>' for x in pill)
        + f'</div><div class="mfull">{html.escape(m)}</div>'
        f'<div class="mrun">{html.escape(" &middot; ".join(meta))}</div>'.replace(
            "&amp;middot;", "&middot;")
        + (f'<div class="msrc">measured through {html.escape(", ".join(src))} '
           f'on a Tiiny Pocket Lab</div>' if src else "")
        + '</header>')

    # --- the five headline numbers ---------------------------------------
    s = r.get("sustained") or {}
    srun = s.get("run") or {}
    cc = r.get("concurrency") or []
    pf = [p for p in (r.get("prefill") or []) if p.get("prompt_tokens")]
    th = r.get("thinking") or {}
    cards = []
    if pf:
        first = min(pf, key=lambda p: p["prompt_tokens"])
        if first.get("decode_tok_s"):
            cards.append(stat(
                f'{first["decode_tok_s"]:.1f}<span class="u">tok/s</span>',
                "single stream",
                f'first token after {first.get("ttft_s", 0):.2f}s at '
                f'{first["prompt_tokens"]:,} tokens in'))
    if srun.get("decode_tok_s"):
        cards.append(stat(f'{srun["decode_tok_s"]:.1f}<span class="u">tok/s</span>',
                          "sustained decode",
                          f'{srun.get("out_tokens", 0):,} tokens unbroken'))
    if cc:
        top = max(cc, key=lambda x: x["aggregate_tok_s"])
        cards.append(stat(f'{top["aggregate_tok_s"]:.1f}<span class="u">tok/s</span>',
                          f'batched at {top["parallel"]} streams',
                          "every caller added together"))
    if pf:
        deep = max(pf, key=lambda p: p["prompt_tokens"])
        cards.append(stat(f'{deep.get("ttft_s", 0):.2f}<span class="u">s</span>',
                          "deepest prefill",
                          f'wait at {deep["prompt_tokens"]:,} tokens of prompt'))
    if th.get("on") and th.get("off") and th["off"].get("wall_s"):
        ratio = th["on"]["wall_s"] / th["off"]["wall_s"]
        cards.append(stat(f'{ratio:.1f}<span class="u">x</span>', "reasoning tax",
                          "wall clock, thinking on against off"))
    if cards:
        parts.append('<div class="stats">' + "".join(cards) + "</div>")

    # --- prefill ---------------------------------------------------------
    if len(pf) > 1:
        lo = min(pf, key=lambda p: p["prompt_tokens"])
        hi = max(pf, key=lambda p: p["prompt_tokens"])
        grew = hi["prompt_tokens"] / max(lo["prompt_tokens"], 1)
        waited = (hi.get("ttft_s") or 0) / max(lo.get("ttft_s") or 0.0001, 0.0001)
        rate_lo = lo.get("prefill_tok_s") or 0
        rate_hi = hi.get("prefill_tok_s") or 0
        if rate_hi > rate_lo * 1.15:
            reading = (f'Reading got <b>cheaper</b> per token as the prompt grew, from '
                       f'{rate_lo:,.0f} to {rate_hi:,.0f} tok/s.')
        elif rate_hi < rate_lo * 0.85:
            reading = (f'Reading got <b>dearer</b> per token as the prompt grew, from '
                       f'{rate_lo:,.0f} down to {rate_hi:,.0f} tok/s.')
        else:
            reading = f'Reading held steady at about {rate_hi:,.0f} tok/s at every length.'
        cap = (f'{grew:.0f}x the prompt cost <b>{waited:.1f}x</b> the wait, '
               f'{(lo.get("ttft_s") or 0):.2f}s at {lo["prompt_tokens"]:,} tokens against '
               f'{(hi.get("ttft_s") or 0):.2f}s at {hi["prompt_tokens"]:,}. {reading} '
               f'Decode stayed near {(hi.get("decode_tok_s") or 0):.0f} tok/s throughout, '
               f'so the prompt is paid for once at the front and not again per word.')
        parts.append(
            '<div class="panel"><div class="ptitle">Prefill against prompt length</div>'
            '<p class="lede">What a long prompt costs to read, and what it does to the '
            'wait before the first word. Solid is time to first token on the left axis; '
            'dashed is decode rate on the right.</p>'
            + dual_axis(pf) + caption(cap) + points_table(pf) + "</div>")

    # --- concurrency -----------------------------------------------------
    if cc:
        groups = [(f'{c["parallel"]}x', [c["aggregate_tok_s"], c["per_stream_tok_s"]],
                   f'{c["wall_s"]:.0f}s wall') for c in cc]
        one = cc[0]["aggregate_tok_s"] or 0
        best = max(c["aggregate_tok_s"] for c in cc)
        gain = best / one if one else 0
        deepest = max(cc, key=lambda c: c["parallel"])
        if 0 < gain < 1.25:
            cap = (f'{deepest["parallel"]} streams deliver <b>{gain:.2f}x</b> the aggregate '
                   f'of one, and wall time grows in step with the callers, '
                   f'{cc[0]["wall_s"]:.0f}s to {deepest["wall_s"]:.0f}s. The box is not '
                   f'sharing itself between requests, it is queueing them. Plan for one '
                   f'inference at a time.')
        else:
            cap = (f'{deepest["parallel"]} streams deliver <b>{gain:.2f}x</b> the aggregate '
                   f'of one. Batching is where this hardware pays off.')
        failed = sum(c.get("failed") or 0 for c in cc)
        if failed:
            cap += f' {failed} request(s) failed and are not in these figures.'
        parts.append(
            '<div class="panel"><div class="ptitle">More than one caller at once</div>'
            '<p class="lede">Every stream added together, against what each single caller '
            'saw, at each level.</p>'
            + grouped_bars(groups, [("aggregate", C["cool"]), ("per stream", C["copper"])])
            + caption(cap) + "</div>")

    # --- reasoning -------------------------------------------------------
    if th.get("on") and th.get("off"):
        off, on = th["off"], th["on"]
        rows = [("thinking off", off.get("wall_s") or 0,
                 f'{off.get("out_tokens", 0):,} tokens'),
                ("thinking on", on.get("wall_s") or 0,
                 f'{on.get("out_tokens", 0):,} tokens')]
        extra = (on.get("out_tokens", 0) - off.get("out_tokens", 0))
        secs = (on.get("wall_s") or 0) - (off.get("wall_s") or 0)
        cap = (f'Reasoning added <b>{secs:.1f}s</b> and {extra:,} tokens to the same '
               f'question. At {(on.get("decode_tok_s") or 0):.0f} tok/s that is the '
               f'model thinking rather than the box slowing down: the decode rate barely '
               f'moved, {(off.get("decode_tok_s") or 0):.0f} against '
               f'{(on.get("decode_tok_s") or 0):.0f} tok/s.')
        parts.append(
            '<div class="panel"><div class="ptitle">What reasoning costs</div>'
            '<p class="lede">The same question asked twice. Hidden reasoning tokens are '
            'paid for in wall time whether or not anyone reads them.</p>'
            + hbars(rows, unit="s", fmt="{:.1f}",
                    colours={"thinking off": C["steel"], "thinking on": C["copper"]})
            + caption(cap) + "</div>")

    # --- device telemetry ------------------------------------------------
    during = s.get("during") or {}
    after = s.get("telemetry_after") or {}
    before = s.get("telemetry_before") or {}
    strip = chips([
        ("peak NPU", f'{during["npu_util_peak"]:.0f}%' if during.get("npu_util_peak") else None),
        ("median NPU", f'{during["npu_util_median"]:.0f}%' if during.get("npu_util_median") is not None else None),
        ("NPU memory at peak",
         (f'{during["npu_mem_peak_mb"]:,.0f} of '
          f'{during.get("npu_mem_total_mb") or before.get("npu_mem_total_mb") or 0:,.0f} MB')
         if during.get("npu_mem_peak_mb") and
            (during.get("npu_mem_total_mb") or before.get("npu_mem_total_mb"))
         else (f'{during["npu_mem_peak_mb"]:,.0f} MB'
               if during.get("npu_mem_peak_mb") else None)),
        ("CPU after", f'{after["cpu_total_pct"]:.0f}%' if after.get("cpu_total_pct") else None),
        ("samples", during.get("samples")),
    ])
    if strip:
        parts.append(
            '<div class="panel"><div class="ptitle">What the device was doing</div>'
            + strip
            + caption('Sampled on a separate thread <b>while the long generation ran</b>, '
                      'not before or after it. A reading taken once the request has '
                      'returned always catches the box going idle and says nothing.')
            + "</div>")

    parts.append("</section>")
    return "".join(parts)


def multi_line(series, xlabel, ylabel, w=880, h=340, logx=True, fmt="{:.0f}"):
    """series: [(name, [(x, y), ...], colour)]. Many models on one axis.

    Direct point labels stop working past about three lines, so this one leans
    on the shared legend above it and labels only the axis. The line is the
    shape; the table underneath is for reading exact values off.
    """
    pts = [p for _, ps, _ in series for p in ps]
    if not pts:
        return ""
    pad_l, pad_r, pad_t, pad_b = 62, 22, 22, 42
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = (math.log10(max(min(xs), 1)), math.log10(max(max(xs), 10))) if logx \
        else (min(xs), max(xs))
    y1 = max(ys) * 1.14 or 1

    def px(x):
        return _scale(math.log10(max(x, 1)) if logx else x, x0, x1, pad_l, w - pad_r)

    def py(y):
        return _scale(y, 0, y1, h - pad_b, pad_t)

    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
         f'aria-label="{html.escape(ylabel)} against {html.escape(xlabel)}, every model">']
    for i in range(5):
        y = y1 * i / 4
        o.append(f'<line x1="{pad_l}" y1="{py(y):.1f}" x2="{w-pad_r}" y2="{py(y):.1f}" '
                 f'stroke="{C["line"]}" stroke-width="1"/>')
        o.append(f'<text x="{pad_l-9}" y="{py(y)+4:.1f}" text-anchor="end" '
                 f'class="tick">{fmt.format(y)}</text>')
    for name, ps, col in series:
        ps = sorted(ps)
        if not ps:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                     for i, (x, y) in enumerate(ps))
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round" opacity=".92"/>')
        for x, y in ps:
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" '
                     f'fill="{col}"/>')
    # A dozen models asked for a dozen slightly different prompt lengths, so a
    # tick per distinct value is a smear of overlapping numbers. Keep the ones
    # that are far enough apart to read and drop the rest; the table under the
    # chart is where exact values are read anyway.
    last = -1e9
    for x in sorted({p[0] for p in pts}):
        at = px(x)
        if at - last < 54:
            continue
        last = at
        o.append(f'<text x="{at:.1f}" y="{h-pad_b+20:.1f}" text-anchor="middle" '
                 f'class="tick">{x:,.0f}</text>')
    o.append(f'<text x="{pad_l}" y="{h-6}" class="axis">{html.escape(xlabel)}</text>')
    o.append(f'<text x="{pad_l}" y="14" class="axis">{html.escape(ylabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def newest_per_model(runs):
    """One run per model, the newest, which is what the box does now.

    Best-of would flatter the box, and this whole report exists because spec
    sheets flatter. A model whose newest run is a partial is plotted with what
    it has and says so, rather than reaching back for a fuller old one.
    """
    best = {}
    for r in runs:
        m = r.get("model")
        if not m:
            continue
        if m not in best or (r.get("stamp") or "") > (best[m].get("stamp") or ""):
            best[m] = r
    return [best[m] for m in sorted(best)]


def _efficiency_caption(best, worst, picks, meta, speed_rows):
    """What the per-unit figure means for somebody choosing a model.

    The ratio on its own is a fact about arithmetic. What a reader wants is the
    consequence: the box has a hundred units, a resident model holds its share
    of them until it is unloaded, and that is the budget every other thing on
    the device has to fit into. So the caption spends the budget out loud.
    """
    units = {short(r["model"]): (r.get("npu_usage")
                                 or (meta.get(r["model"]) or {}).get("npu_usage"))
             for r in picks}
    speed = {n: v for n, v, _ in speed_rows}
    bu, wu = units.get(best[0]), units.get(worst[0])
    bits = [f'<b>{html.escape(best[0])}</b> returns {best[1]:.2f} tok/s for every unit it '
            f'holds and <b>{html.escape(worst[0])}</b> returns {worst[1]:.2f}, a '
            f'{best[1] / worst[1]:.1f}x difference in what the same slice of the box buys.']
    if bu and wu:
        bits.append(f'In plain terms: {html.escape(best[0])} occupies {bu} of the 100 units '
                    f'and leaves {100 - bu} for everything else, while '
                    f'{html.escape(worst[0])} occupies {wu} and leaves {100 - wu}.')
    # The trap worth naming: fastest is not the same question as cheapest.
    fastest = speed_rows[0][0] if speed_rows else None
    if fastest and fastest != best[0]:
        fs, fu = speed.get(fastest), units.get(fastest)
        if fs and fu:
            bits.append(f'The fastest model measured, {html.escape(fastest)} at '
                        f'{fs:.1f} tok/s, is not the most efficient: it costs {fu} units '
                        f'to get there, against {bu} for {html.escape(best[0])} at '
                        f'{speed.get(best[0], 0):.1f} tok/s. If one model is going to sit '
                        f'resident while the box does other work, that gap is the whole '
                        f'decision; if the box only ever runs one thing, it does not '
                        f'matter and the previous chart is the one to read.')
    return " ".join(bits)


def cross_model(runs, cat=None):
    """Every model on one axis, four ways, with one legend for the section."""
    picks = newest_per_model(runs)
    if len(picks) < 2:
        return ""
    ids = [r["model"] for r in picks]
    colours = model_colours(ids)
    meta = {m.get("id"): m for m in (cat or [])}
    stamps = sorted({r["stamp"][:8] for r in picks})
    labels = sorted({r["label"] for r in picks})
    when = stamps[0] if len(stamps) == 1 else f"{stamps[0]} to {stamps[-1]}"
    runword = labels[0] if len(labels) == 1 else f"{len(labels)} runs"

    drawn = set()
    out = []

    # 1. sustained decode, horizontally, which is also what fixes the labels
    rows = []
    for r in picks:
        v = ((r["results"].get("sustained") or {}).get("run") or {}).get("decode_tok_s")
        if v:
            npu = r.get("npu_usage") or (meta.get(r["model"]) or {}).get("npu_usage")
            rows.append((short(r["model"]), v, f'{npu}u' if npu else ""))
            drawn.add(r["model"])
    if len(rows) > 1:
        rows.sort(key=lambda x: -x[1])
        by_short = {short(m): colours[m] for m in ids}
        top, bottom = rows[0], rows[-1]
        out.append(
            '<div class="panel wide"><div class="ptitle">Sustained decode</div>'
            '<p class="lede">One unbroken generation each, same prompt, same box. This is '
            'the number that decides whether something feels instant.</p>'
            + hbars(rows, w=880, unit=" tok/s", fmt="{:.1f}", colours=by_short)
            + caption(f'<b>{html.escape(top[0])}</b> leads at {top[1]:.1f} tok/s, '
                      f'{top[1]/bottom[1]:.1f}x the slowest measured, '
                      f'{html.escape(bottom[0])} at {bottom[1]:.1f}. Every one of these is '
                      f'a single stream on an idle box.')
            + "</div>")

    # 2. tokens per second per NPU unit: the efficiency question
    eff = []
    for r in picks:
        v = ((r["results"].get("sustained") or {}).get("run") or {}).get("decode_tok_s")
        npu = r.get("npu_usage") or (meta.get(r["model"]) or {}).get("npu_usage")
        if v and npu:
            eff.append((short(r["model"]), v / npu, f'{v:.0f} tok/s on {npu}u'))
            drawn.add(r["model"])
    if len(eff) > 1:
        eff.sort(key=lambda x: -x[1])
        by_short = {short(m): colours[m] for m in ids}
        best, worst = eff[0], eff[-1]
        out.append(
            '<div class="panel wide"><div class="ptitle">Throughput per NPU unit</div>'
            '<p class="lede">The box has 100 NPU units and a loaded model holds its share '
            'of them for as long as it is resident. This is what each one returns for what '
            'it occupies, which is the question that decides what to keep loaded.</p>'
            + hbars(eff, w=880, unit=" tok/s per unit", fmt="{:.2f}", colours=by_short)
            + caption(_efficiency_caption(best, worst, picks, meta, rows))
            + "</div>")

    # 3. prefill curves, all models on one axis
    series = []
    for r in picks:
        pf = [(p["prompt_tokens"], p["prefill_tok_s"])
              for p in (r["results"].get("prefill") or [])
              if p.get("prefill_tok_s") and p.get("prompt_tokens")]
        if len(pf) > 1:
            series.append((short(r["model"]), pf, colours[r["model"]]))
            drawn.add(r["model"])
    if len(series) > 1:
        tops = [(n, max(y for _, y in ps)) for n, ps, _ in series]
        tops.sort(key=lambda x: -x[1])
        out.append(
            '<div class="panel wide"><div class="ptitle">Prefill, every model</div>'
            '<p class="lede">How fast each model reads a prompt, at four lengths. The x '
            'axis is logarithmic because the prompts are.</p>'
            + multi_line(series, "prompt tokens", "prefill tok/s")
            + caption(f'<b>{html.escape(tops[0][0])}</b> reads fastest at its best length, '
                      f'{tops[0][1]:,.0f} tok/s, against {tops[-1][1]:,.0f} for '
                      f'{html.escape(tops[-1][0])}. A line that climbs to the right is a '
                      f'model that gets cheaper per token as the prompt grows.')
            + "</div>")

    # 4. aggregate throughput against stream count, all models
    cser = []
    for r in picks:
        cc = [(c["parallel"], c["aggregate_tok_s"])
              for c in (r["results"].get("concurrency") or [])
              if c.get("aggregate_tok_s")]
        if len(cc) > 1:
            cser.append((short(r["model"]), cc, colours[r["model"]]))
            drawn.add(r["model"])
    if len(cser) > 1:
        gains = []
        for n, cc, _ in cser:
            cc = sorted(cc)
            if cc[0][1]:
                gains.append((n, max(y for _, y in cc) / cc[0][1]))
        gains.sort(key=lambda x: -x[1])
        flat = sum(1 for _, g in gains if g < 1.25)
        verdict = (f'All {len(gains)} of them are flat, the best managing '
                   f'{gains[0][1]:.2f}x at its deepest level. '
                   if flat == len(gains) else
                   f'{flat} of {len(gains)} are flat; the best, '
                   f'<b>{html.escape(gains[0][0])}</b>, reaches {gains[0][1]:.2f}x. ')
        out.append(
            '<div class="panel wide"><div class="ptitle">Aggregate throughput against '
            'stream count</div>'
            '<p class="lede">Every caller added together as more of them arrive at once. '
            'A line that climbs is a box that batches; a flat line is a box that queues.</p>'
            + multi_line(cser, "streams at once", "aggregate tok/s", logx=False)
            + caption(verdict + 'This is one accelerator serving one sequence at a time, so '
                      'a second caller waits rather than shares. It is the single most '
                      'important thing to know before putting a Tiiny behind anything with '
                      'more than one user.')
            + "</div>")

    if not out:
        return ""
    head = ['<section class="cross" id="compare">',
            '<h2>Every model, side by side</h2>',
            f'<p class="lede">The newest run for each of the {len(drawn)} models with '
            f'something to compare, not the best run: this is what the box does now. '
            f'Colour is the same for a given model in every chart below. Run '
            f'<b>{html.escape(runword)}</b>, {html.escape(when)}.</p>',
            legend(sorted(drawn), colours)]
    return "".join(head + out + ["</section>"])


def catalog_table(cat):
    if not cat:
        return ""
    rows = []
    for m in cat:
        gb = (m.get("total_size") or 0) / 1e9
        rows.append(
            f'<tr><td class="mono">{html.escape(m.get("id", ""))}</td>'
            f'<td class="num">{html.escape(str(m.get("params") or "-"))}</td>'
            f'<td class="num">{gb:.1f} GB</td>'
            f'<td class="num">{html.escape(str(m.get("npu_usage") or "-"))}</td>'
            f'<td>{html.escape(m.get("type") or "-")}</td></tr>')
    return (
        '<section id="catalog"><h2>What is on the box</h2>'
        '<p class="lede">Read off the device itself, not a list kept in this repo. '
        'NPU units are the budget that decides what can run at the same time; the box '
        'has 100 of them.</p>'
        '<div class="tablewrap"><table><thead><tr><th>model</th><th class="num">params</th>'
        '<th class="num">size</th><th class="num">NPU</th><th>type</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>')


# ------------------------------------------------------------- markdown
# The same data, rendered as text. Not a transcription of the HTML: a chart
# becomes the table it was drawn from, because ASCII art of a line chart is
# worse than the numbers it hides. Everything here reads the result files
# through the same loader the page does, so the two cannot drift.

def _md_table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def markdown(outdir: pathlib.Path, cat=None) -> str:
    runs = load(outdir)
    if not runs:
        return "# TiinyBench\n\nNo results yet. Run `tiiny-bench --label first-run`.\n"
    picks = newest_per_model(runs)
    newest = max(runs, key=lambda r: r["stamp"])
    meta = {m.get("id"): m for m in (cat or [])}
    L = ["# TiinyBench",
         "",
         "An independent measurement of what a Tiiny Pocket Lab actually does, as "
         "opposed to what a spec sheet says it does.",
         "",
         f"- runs: {len(runs)}",
         f"- models: {len({r['model'] for r in runs if r.get('model')})}",
         f"- latest: {newest['stamp']}"]
    if newest.get("build"):
        L.append(f"- firmware: {newest['build']}")
    L += ["", "Every number below was measured on one Tiiny Pocket Lab. Nothing here is "
              "quoted from a datasheet and nothing is an average across devices.", ""]

    # ---- cross-model ---------------------------------------------------
    rows = []
    for r in picks:
        v = ((r["results"].get("sustained") or {}).get("run") or {}).get("decode_tok_s")
        npu = r.get("npu_usage") or (meta.get(r["model"]) or {}).get("npu_usage")
        if v:
            rows.append((short(r["model"]), f"{v:.1f}", npu or "-",
                         f"{v / npu:.2f}" if npu else "-", r["stamp"][:8]))
    if len(rows) > 1:
        rows.sort(key=lambda x: -float(x[1]))
        L += ["## Every model, side by side", "",
              "The newest run for each model, not the best one.", "",
              _md_table(["model", "sustained tok/s", "NPU units",
                         "tok/s per unit", "measured"], rows), ""]

    # ---- per model -----------------------------------------------------
    L += ["## Every run", "", "Newest first.", ""]
    for r in sorted(runs, key=lambda x: x["stamp"], reverse=True):
        res = r.get("results") or {}
        L.append(f"### {short(r['model'])}")
        L.append("")
        bits = [x for x in (r.get("type"), r.get("params") and f"{r['params']} params",
                            r.get("npu_usage") and f"{r['npu_usage']} of 100 NPU units",
                            r.get("total_size") and f"{r['total_size'] / 1e9:.1f} GB")
                if x]
        L.append("`" + r["model"] + "`")
        L.append("")
        if bits:
            L.append(" · ".join(str(b) for b in bits))
            L.append("")
        stamp = [f"run {r['label']}", r["stamp"]]
        if r.get("build"):
            stamp.append(f"TiinyOS {r['build']}")
        L.append("*" + ", ".join(stamp) + ", on a Tiiny Pocket Lab.*")
        L.append("")

        pf = [p for p in (res.get("prefill") or []) if p.get("prompt_tokens")]
        if pf:
            L += ["**Prefill against prompt length**", "",
                  _md_table(["prompt tokens", "time to first token", "prefill tok/s",
                             "decode tok/s"],
                            [(f'{p["prompt_tokens"]:,}', f'{p.get("ttft_s", 0):.2f} s',
                              f'{p.get("prefill_tok_s", 0):,.0f}',
                              f'{p.get("decode_tok_s", 0):.1f}')
                             for p in sorted(pf, key=lambda p: p["prompt_tokens"])]), ""]
        srun = (res.get("sustained") or {}).get("run") or {}
        if srun.get("decode_tok_s"):
            L += ["**Sustained generation**", "",
                  f'{srun.get("out_tokens", 0):,} tokens unbroken at '
                  f'{srun["decode_tok_s"]:.1f} tok/s, first token after '
                  f'{srun.get("ttft_s", 0):.2f} s, {srun.get("wall_s", 0):.1f} s of wall '
                  f'clock.', ""]
        during = (res.get("sustained") or {}).get("during") or {}
        if during.get("npu_util_peak"):
            L += [f'NPU peaked at {during["npu_util_peak"]:.0f}% and held a median of '
                  f'{during.get("npu_util_median", 0):.0f}% across '
                  f'{during.get("samples", 0)} samples taken while the generation ran, '
                  f'with {during.get("npu_mem_peak_mb", 0):,.0f} MB resident at peak.', ""]
        cc = res.get("concurrency") or []
        if cc:
            L += ["**More than one caller at once**", "",
                  _md_table(["streams", "aggregate tok/s", "per stream tok/s", "wall",
                             "ok", "failed"],
                            [(c["parallel"], f'{c["aggregate_tok_s"]:.1f}',
                              f'{c["per_stream_tok_s"]:.1f}', f'{c["wall_s"]:.1f} s',
                              c.get("ok", "-"), c.get("failed", 0)) for c in cc]), ""]
            one = cc[0]["aggregate_tok_s"] or 0
            best = max(c["aggregate_tok_s"] for c in cc)
            if one:
                gain = best / one
                L += [f'Deepest level returns {gain:.2f}x the aggregate of one stream. '
                      + ("The box is queueing, not batching." if gain < 1.25
                         else "There is real headroom in running more than one caller."),
                      ""]
        th = res.get("thinking") or {}
        if th.get("on") and th.get("off"):
            L += ["**What reasoning costs**", "",
                  _md_table(["", "wall", "output tokens", "decode tok/s",
                             "time to first token"],
                            [(k, f'{th[k].get("wall_s", 0):.1f} s',
                              f'{th[k].get("out_tokens", 0):,}',
                              f'{th[k].get("decode_tok_s", 0):.1f}',
                              f'{th[k].get("ttft_s", 0):.2f} s') for k in ("off", "on")]), ""]
        for key, title, unit in (("image", "Illustration", "s_per_image"),
                                 ("speech", "Speech", "rtf"),
                                 ("embed", "Embeddings", "emb_per_s"),
                                 ("asr", "Transcription", "rtf"),
                                 ("ocr", "Page reading", "s_per_page"),
                                 ("music", "Music", "audio_per_s"),
                                 ("rerank", "Reranking", "pairs_per_s")):
            blk = res.get(key) or {}
            if blk.get(unit) is not None:
                L += [f"**{title}**", "",
                      f"{unit.replace('_', ' ')}: {blk[unit]}", ""]

    L += ["## How it is measured", "",
          "- **Prefill.** The same deterministic filler repeated to four lengths, each "
          "asked for a four-token answer so decode barely registers.",
          "- **Sustained.** One unbroken 1500-token generation, with NPU utilisation "
          "sampled on a separate thread while it runs rather than after it.",
          "- **Concurrency.** 1, 2, 4 and 8 identical requests fired at once from "
          "separate threads.",
          "- **Reasoning.** One arithmetic word problem, asked with thinking off and "
          "then on. The ratio is wall time.",
          "",
          "No quality evaluation of any kind. Nothing here says a model is good, only "
          "how fast it is.", ""]
    stale = [r for r in runs if (r.get("stamp") or "") < "20260919"]
    if stale:
        L += [f"*The `bench_version` field on the {len(stale)} runs measured before "
              "2026-09-19 is not reliable: the version constant was not bumped for two "
              "releases, so files written by 0.1.5 and 0.1.6 are stamped 0.1.4. The "
              "measurements themselves are unaffected.*", ""]
    return "\n".join(L)


# -------------------------------------------------------------------- css
# The tokens are generated from C so the palette lives in exactly one place.
# The rest of the sheet is a plain literal - it is full of percentages, and
# %-formatting a stylesheet is a trap that only shows up at runtime.
TOKENS = (":root{"
          + "".join(f"--{k}:{v};" for k, v in C.items())
          + '--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,'
            '"DejaVu Sans Mono",Consolas,monospace;'
          + "--pad:clamp(20px,4vw,64px);}")

CSS = TOKENS + """
/* ---- print, which is also how a PDF is made ------------------------
   No PDF library: the browser has one and it is better than anything that
   would fit in here. What that needs from the page is a light ground the
   toner can survive, real page breaks between model blocks, and charts that
   do not get cut in half at the bottom of a sheet. */
@media print{
  @page{margin:14mm 12mm}
  html,body{background:#fff !important}
  body{color:#111;padding:0;font-size:10.5pt;
       background-image:none !important}
  .wrap{max-width:none}
  a{color:#111;text-decoration:none}
  /* Hand the dark palette back as ink on paper. The charts read their
     colours from these tokens, so redefining them repaints the SVG too. */
  :root{
    --ground:#fff; --s1:#fff; --s2:#f6f5f3; --line:#d8d5d0;
    --steel:#55585e; --ink:#111316;
  }
  .top{border:0;padding-top:0}
  .top h1{font-size:30pt}
  header.top .tag{max-width:none}
  /* A model block is two or three sheets tall, so asking print to keep it
     whole only pushes it onto the next page and leaves its border stretched
     down a mostly empty one. It starts a page and is then allowed to flow;
     the panels inside it are the things worth keeping whole. */
  section.model{break-before:page; page-break-before:always;
                border:0; border-top:1px solid var(--line); border-radius:0;
                padding-top:6mm}
  /* An empty grid cell prints as a grey block, which reads as a missing
     measurement rather than as a gap in the layout. */
  .stats{background:none !important}
  .stats>*{break-inside:avoid; page-break-inside:avoid}
  /* :first-of-type would look for the first <section> on the page, which is
     the cross-model one, so it never matched and the first model block got a
     page break that left its own heading stranded on the sheet before. */
  .runsintro + section.model{break-before:auto; page-break-before:auto}
  .runsintro{break-after:avoid; page-break-after:avoid}
  .cross{break-after:page; page-break-after:always}
  .panel{break-inside:avoid-page; page-break-inside:avoid}
  .chart{break-inside:avoid; page-break-inside:avoid; max-width:100%}
  .stats{break-inside:avoid; page-break-inside:avoid}
  table{break-inside:auto}
  tr{break-inside:avoid; page-break-inside:avoid}
  thead{display:table-header-group}
  /* A horizontally scrolling box prints as a clipped column, so on paper it
     stops scrolling and wraps its own way. */
  .tablewrap{overflow:visible !important}
  h2,h3{break-after:avoid; page-break-after:avoid}
  .legend{break-inside:avoid; page-break-inside:avoid}
  footer{break-before:avoid}
  .noprint{display:none !important}
}

/* ---- added for the rebuilt report ---------------------------------- */
.mtop{display:flex;align-items:center;gap:10px}
.mtop h3{margin:0}
.dot{width:11px;height:11px;border-radius:50%;flex:none;display:inline-block}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px}
.pill{font-size:11.5px;letter-spacing:.02em;color:var(--steel);background:var(--s2);
  border:1px solid var(--line);border-radius:999px;padding:2px 10px;white-space:nowrap}
.msrc{font-size:11.5px;color:var(--steel);opacity:.75;margin-top:6px}
.cap{font-size:13.5px;color:var(--steel);line-height:1.6;margin:10px 0 0;
  border-left:2px solid var(--line);padding-left:12px;max-width:74ch}
.cap b{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 2px}
.chip{background:var(--s2);border:1px solid var(--line);border-radius:6px;
  padding:8px 12px;display:flex;flex-direction:column;gap:2px;min-width:96px}
.chip b{font-size:17px;color:var(--ink);font-weight:600;letter-spacing:-.02em}
.chip span{font-size:11px;color:var(--steel);letter-spacing:.04em}
.legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 16px;margin:14px 0 20px;
  padding:12px 14px;background:var(--s1);border:1px solid var(--line);border-radius:8px}
.lgt{width:100%;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--steel)}
.lg{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;color:var(--steel);
  white-space:nowrap}
.lg i{width:10px;height:10px;border-radius:2px;flex:none;display:inline-block}
.hlab{fill:var(--ink);font-size:12px}
.cross{margin:0 0 56px}
.cross h2{margin-bottom:6px}

*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--ground)}
body{
  font-family:var(--mono); color:var(--ink); line-height:1.55;
  font-size:15px; letter-spacing:-.01em;
  background:
    radial-gradient(120% 70% at 50% -10%, #16181c 0%, transparent 60%),
    var(--ground);
  padding:0 var(--pad) 120px; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto}

/* ---- masthead ------------------------------------------------------- */
header.top{padding:clamp(40px,9vh,110px) 0 clamp(28px,5vh,54px);border-bottom:1px solid var(--line)}
.brandrow{display:flex;align-items:center;gap:18px;margin-bottom:clamp(26px,5vh,52px)}
.brandrow img.badge{width:44px;height:44px;border-radius:9px;display:block}
.brandrow .by{font-size:11px;color:var(--steel);letter-spacing:.16em;text-transform:uppercase;line-height:1.5}
.brandrow .by b{color:var(--ink);font-weight:500;letter-spacing:.1em}
h1{
  font-size:clamp(38px,8.2vw,86px); font-weight:600; line-height:.96;
  letter-spacing:-.045em;
}
h1 em{font-style:normal;color:var(--copper)}
.tag{margin-top:20px;max-width:62ch;color:var(--steel);font-size:clamp(14px,1.6vw,17px);text-wrap:pretty}
.runmeta{margin-top:30px;display:flex;flex-wrap:wrap;gap:10px 26px;font-size:12px;color:var(--steel)}
.runmeta b{color:var(--ink);font-weight:500}

/* ---- sections ------------------------------------------------------- */
section{padding:clamp(40px,7vh,80px) 0;border-bottom:1px solid var(--line)}
h2{font-size:clamp(20px,2.6vw,27px);font-weight:600;letter-spacing:-.03em;margin-bottom:12px}
h3{font-size:clamp(19px,2.4vw,25px);font-weight:600;letter-spacing:-.03em}
.lede{color:var(--steel);max-width:66ch;margin-bottom:26px;text-wrap:pretty;font-size:14px}
.lede b{color:var(--ink);font-weight:500}

/* ---- stat cards ----------------------------------------------------- */
.stats{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);
  grid-template-columns:repeat(auto-fit,minmax(168px,1fr));margin:26px 0 34px;border-radius:3px;overflow:hidden}
.stat{background:var(--s1);padding:18px 18px 16px}
.stat .v{font-size:clamp(24px,3.4vw,34px);font-weight:600;letter-spacing:-.04em;color:var(--hot);line-height:1.05}
.stat .v .u{font-size:.44em;color:var(--steel);margin-left:.35em;font-weight:400;letter-spacing:0}
.stat .k{margin-top:7px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--steel)}
.stat .note{margin-top:4px;font-size:11px;color:var(--steel);opacity:.72}

/* ---- panels and charts ---------------------------------------------- */
.panel{background:var(--s1);border:1px solid var(--line);border-radius:3px;
  padding:clamp(18px,2.6vw,30px);margin-bottom:22px}
.panel.wide{background:none;border:0;padding:0}
.ptitle{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--steel);margin-bottom:10px}
.chart{width:100%;height:auto;display:block;margin-top:6px;overflow:visible}
.chart .tick{fill:var(--steel);font:11px var(--mono)}
.chart .tick.dim{opacity:.6}
.chart .val{fill:var(--ink);font:12px var(--mono);font-weight:600}
.chart .lbl{font:11px var(--mono);letter-spacing:.08em}
.chart .axis{fill:var(--steel);font:10px var(--mono);letter-spacing:.14em;text-transform:uppercase;opacity:.75}

/* ---- model blocks --------------------------------------------------- */
.runsintro{padding:clamp(40px,7vh,80px) 0 0}
.runsintro .lede{margin-bottom:0}
.model{border-bottom:1px solid var(--line);padding-top:clamp(26px,4vh,44px)}
.mhead{margin-bottom:8px}
.mmeta{margin-top:6px;font-size:12px;color:var(--hot);letter-spacing:.04em}
.mfull{margin-top:3px;font-size:12px;color:var(--steel);opacity:.7;word-break:break-all}
.mrun{margin-top:8px;font-size:11px;color:var(--steel);letter-spacing:.08em;text-transform:uppercase}
.mrun b{color:var(--ink);font-weight:500}

/* ---- table ---------------------------------------------------------- */
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:640px}
th{text-align:left;font-weight:500;font-size:10px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--steel);padding:12px 14px;background:var(--s2);border-bottom:1px solid var(--line)}
td{padding:10px 14px;border-bottom:1px solid var(--line);color:var(--ink)}
tr:last-child td{border-bottom:0}
tr:hover td{background:var(--s1)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.mono{color:var(--ink)}

/* ---- method --------------------------------------------------------- */
.method{display:grid;gap:22px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));margin-top:26px}
.method h4{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--hot);
  font-weight:500;margin-bottom:8px}
.method p{color:var(--steel);font-size:13px;text-wrap:pretty}
footer{padding-top:44px;color:var(--steel);font-size:12px;display:flex;
  flex-wrap:wrap;gap:16px 30px;align-items:center}
footer img{height:15px;opacity:.75;display:block}
footer a{color:var(--steel)}
@media (prefers-reduced-motion:no-preference){
  .stat,.panel{animation:rise .5s cubic-bezier(.22,.61,.36,1) both}
}
@keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
"""


def build(outdir: pathlib.Path, dest: pathlib.Path, cat=None, device=None):
    runs = load(outdir)
    if not runs:
        dest.write_text(
            "<!doctype html><meta charset=utf-8><title>TiinyBench</title>"
            "<body style='font:16px monospace;padding:3em'>"
            "No results yet. Run <code>tiiny-bench --label first-run</code>.",
            encoding="utf-8")
        return dest

    newest = max(runs, key=lambda r: r["stamp"])
    body = [f"<!doctype html><html lang=en><meta charset=utf-8>",
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            "<title>TiinyBench</title>",
            f"<style>{CSS}</style><body><div class=wrap>"]

    # Inlined, not linked. The file claims to be self-contained and a report
    # that breaks when you move it out of its folder is not.
    logo_t = data_uri(outdir.parent / "assets" / "tiiny-logo.svg")
    badge = data_uri(outdir.parent / "assets" / "titanium-sm.png")
    brand = '<div class="brandrow">'
    if badge:
        brand += f'<img class="badge" src="{badge}" alt="Titanium Computing">'
    brand += ('<div class="by">benchmarks by<br><b>Titanium Computing</b></div>')
    if logo_t:
        brand += ('<div class="by" style="margin-left:auto;text-align:right">measured on'
                  f'<br><img src="{logo_t}" alt="Tiiny" '
                  'style="height:17px;margin-top:4px;display:inline-block"></div>')
    brand += "</div>"

    models_seen = sorted({short(r["model"]) for r in runs})
    body.append(
        f'<header class="top">{brand}'
        f'<h1>Tiiny<em>Bench</em></h1>'
        '<p class="tag">An independent measurement of what a Tiiny Pocket actually does, '
        'as opposed to what a spec sheet says it does. Prefill against prompt length, '
        'throughput over a long generation, what happens when more than one person uses '
        'the box, and what a reasoning model charges for the tokens nobody reads.</p>'
        f'<div class="runmeta">'
        f'<span>runs <b>{len(runs)}</b></span>'
        f'<span>models <b>{len(models_seen)}</b></span>'
        f'<span>latest <b>{html.escape(newest["stamp"])}</b></span>'
        + (f'<span>firmware <b>{html.escape(str(newest["build"]))}</b></span>'
           if newest.get("build") else "")
        + '</div></header>')

    body.append(cross_model(runs, cat))
    colours = model_colours([r.get("model") for r in runs if r.get("model")])
    body.append('<div class="runsintro"><h2>Every run</h2>'
                '<p class="lede">Newest first. Each block is one model on one day, with the '
                'numbers exactly as the device reported them.</p></div>')
    for i, r in enumerate(sorted(runs, key=lambda r: r["stamp"], reverse=True)):
        body.append(model_section(r, i, colours.get(r.get("model"))))

    body.append(catalog_table(cat))

    body.append(
        '<section id="method"><h2>How it is measured</h2>'
        '<p class="lede">So you can decide whether to believe any of it.</p>'
        '<div class="method">'
        '<div><h4>Prefill</h4><p>The same deterministic filler repeated to four lengths, '
        'each asked for a four-token answer so decode barely registers. What is plotted is '
        "the device's own reported prefill rate, not a stopwatch.</p></div>"
        '<div><h4>Sustained</h4><p>One unbroken 1500-token generation from a single prompt, '
        'with NPU utilisation and memory sampled either side of it. Short bursts flatter a '
        'box; this is the number that holds.</p></div>'
        '<div><h4>Concurrency</h4><p>1, 2, 4 and 8 identical requests fired at once from '
        'separate threads. Aggregate is total tokens over wall time; per-stream is the '
        'median of what each caller saw.</p></div>'
        '<div><h4>Reasoning</h4><p>One arithmetic word problem, asked with thinking off and '
        'then on. The ratio is wall time, because that is what a person waits.</p></div>'
        '<div><h4>What it does not do</h4><p>No quality evaluation of any kind. Nothing here '
        'says a model is good, only how fast it is. A fast wrong answer is still wrong.</p></div>'
        '<div><h4>What it does not touch</h4><p>By default the suite loads and unloads '
        'nothing, and benchmarks whatever was already running. Sweeps that do load models '
        'put the box back the way they found it.</p></div>'
        "</div></section>")

    foot = ['<footer><span>TiinyBench</span>']
    if logo_t:
        foot.append(f'<span>measured on <img src="{logo_t}" alt="Tiiny" '
                    'style="vertical-align:-3px"></span>')
    foot.append('<span>built by Titanium Computing</span>')
    foot.append(f'<span>{html.escape(newest["stamp"])}</span></footer>')
    body.append("".join(foot))

    # Opened with ?print=1 from the app's Save PDF button, the report prints
    # itself once it has painted. Nothing else on the page needs script, and
    # this does not either: without it the page is just a page.
    body.append(
        "<script>if(location.search.indexOf('print=1')>=0){"
        "addEventListener('load',function(){setTimeout(function(){print();},250);});"
        "}</script>")
    body.append("</div></body></html>")
    dest.write_text("".join(body), encoding="utf-8")
    return dest


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    print(build(here / "bench-results", here / "report.html"))
