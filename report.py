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


# --------------------------------------------------------------- sections
def stat(value, label, note=""):
    n = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="stat"><div class="v">{value}</div>'
            f'<div class="k">{html.escape(label)}</div>{n}</div>')


def model_section(run, idx):
    r = run.get("results") or {}
    m = run.get("model") or "?"
    parts = [f'<section class="model" id="m{idx}">']
    meta = []
    if run.get("params"):
        meta.append(f'{run["params"]} params')
    if run.get("npu_usage"):
        meta.append(f'{run["npu_usage"]} NPU units')
    if run.get("total_size"):
        meta.append(f'{run["total_size"] / 1e9:.1f} GB')
    parts.append(
        f'<header class="mhead"><h3>{html.escape(short(m))}</h3>'
        f'<div class="mmeta">{html.escape(" · ".join(meta)) if meta else ""}</div>'
        f'<div class="mfull">{html.escape(m)}</div>'
        f'<div class="mrun">run <b>{html.escape(run["label"])}</b> · {html.escape(run["stamp"])}</div>'
        f'</header>')

    # --- headline numbers ------------------------------------------------
    s = r.get("sustained") or {}
    srun = (s or {}).get("run") or {}
    cc = r.get("concurrency") or []
    pf = [p for p in (r.get("prefill") or []) if p.get("prefill_tok_s")]
    cards = []
    if srun.get("decode_tok_s"):
        cards.append(stat(f'{srun["decode_tok_s"]:.1f}<span class="u">tok/s</span>',
                          "sustained decode",
                          f'{srun.get("out_tokens", 0)} tokens, one request'))
    if pf:
        best = max(p["prefill_tok_s"] for p in pf)
        cards.append(stat(f'{best:,.0f}<span class="u">tok/s</span>', "peak prefill",
                          "document ingestion"))
    if cc:
        top = max(cc, key=lambda x: x["aggregate_tok_s"])
        cards.append(stat(f'{top["aggregate_tok_s"]:.1f}<span class="u">tok/s</span>',
                          f'aggregate at {top["parallel"]}x',
                          "all streams together"))
    th = r.get("thinking") or {}
    if th.get("on") and th.get("off") and th["off"].get("wall_s"):
        ratio = th["on"]["wall_s"] / th["off"]["wall_s"]
        cards.append(stat(f'{ratio:.1f}<span class="u">x</span>', "reasoning tax",
                          "wall time, thinking on vs off"))
    during = (s or {}).get("during") or {}
    after = (s or {}).get("telemetry_after") or {}
    if during.get("npu_util_median") is not None:
        cards.append(stat(f'{during["npu_util_median"]:.0f}<span class="u">%</span>',
                          "NPU while running",
                          f'peak {during.get("npu_util_peak", 0):.0f}%, '
                          f'{during.get("npu_mem_peak_mb", 0):,.0f} MB'))
    elif after.get("npu_mem_used_mb"):
        # Older runs sampled utilisation only once the request had already
        # returned, which always caught the box going idle. That 0% said
        # nothing, so it is not shown; the resident memory from the same
        # sample is still real.
        cards.append(stat(f'{after["npu_mem_used_mb"]:,.0f}<span class="u">MB</span>',
                          "NPU memory resident", "sampled after the run"))
    if cards:
        parts.append('<div class="stats">' + "".join(cards) + "</div>")

    # --- prefill ---------------------------------------------------------
    if pf:
        pts = [(p["prompt_tokens"], p["prefill_tok_s"]) for p in pf]
        parts.append(
            '<div class="panel"><div class="ptitle">Prefill scaling</div>'
            '<p class="lede">What a long prompt costs to read. A flat line means '
            'context is cheap; a falling one means every extra page of prompt is '
            'taxed twice.</p>'
            + line_chart([("prefill", pts, C["copper"])],
                         "prompt tokens", "prefill tok/s", logx=True)
            + "</div>")

    # --- concurrency -----------------------------------------------------
    if cc:
        agg = [(str(c["parallel"]) + "x", c["aggregate_tok_s"],
                f'{c["per_stream_tok_s"]:.0f}/stream') for c in cc]
        one = cc[0]["aggregate_tok_s"] if cc else 0
        best = max(c["aggregate_tok_s"] for c in cc)
        gain = best / one if one else 0
        # A box that queues shows flat aggregate and linear wall time. Saying
        # "peak 1.03x" without naming that is technically true and useless.
        if 0 < gain < 1.25:
            verdict = (f'Aggregate is <b>flat</b> at {best:.0f} tok/s no matter how many '
                       f'callers arrive, and wall time grows in step with them. The box is '
                       f'not sharing itself between requests, it is queueing them. Plan for '
                       f'one inference at a time.')
        else:
            verdict = (f'Peak aggregate is <b>{gain:.2f}x</b> the single-stream rate, so '
                       f'there is real headroom in running more than one caller.')
        parts.append(
            '<div class="panel"><div class="ptitle">Concurrency</div>'
            f'<p class="lede">Aggregate throughput as more people use the box at once. '
            f'{verdict}</p>'
            + bars(agg, "aggregate tok/s", colour=C["cool"])
            + "</div>")

    # --- reasoning -------------------------------------------------------
    if th.get("on") and th.get("off"):
        rows = [("thinking off", th["off"].get("out_tokens", 0),
                 f'{th["off"].get("wall_s", 0):.1f}s'),
                ("thinking on", th["on"].get("out_tokens", 0),
                 f'{th["on"].get("wall_s", 0):.1f}s')]
        parts.append(
            '<div class="panel"><div class="ptitle">Reasoning cost</div>'
            '<p class="lede">The same question asked twice. Hidden reasoning tokens are '
            'billed in wall time whether or not anyone reads them.</p>'
            + bars(rows, "tokens generated")
            + "</div>")

    parts.append("</section>")
    return "".join(parts)


def comparison(runs):
    """Only drawn when there is something to compare. One model against itself
    is not a chart."""
    have = [(short(r["model"]), ((r["results"].get("sustained") or {}).get("run") or {})
             .get("decode_tok_s"), r["label"]) for r in runs]
    have = [(n, v, l) for n, v, l in have if v]
    if len(have) < 2:
        return ""
    have.sort(key=lambda x: -x[1])
    rows = [(n, v, l) for n, v, l in have]
    return ('<section class="panel wide" id="compare">'
            '<div class="ptitle">Sustained decode, every run side by side</div>'
            '<p class="lede">One long unbroken generation per model, same prompt, same box. '
            'This is the number that decides whether something feels instant.</p>'
            + bars(rows, "tok/s", w=880, h=300)
            + "</section>")


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

    body.append(comparison(runs))
    body.append('<div class="runsintro"><h2>Every run</h2>'
                '<p class="lede">Newest first. Each block is one model on one day, with the '
                'numbers exactly as the device reported them.</p></div>')
    for i, r in enumerate(sorted(runs, key=lambda r: r["stamp"], reverse=True)):
        body.append(model_section(r, i))

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

    body.append("</div></body></html>")
    dest.write_text("".join(body), encoding="utf-8")
    return dest


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    print(build(here / "bench-results", here / "report.html"))
