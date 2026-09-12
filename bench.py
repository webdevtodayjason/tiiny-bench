"""TiinyBench - an independent benchmark suite for the Tiiny AI Pocket Lab.

    tiiny-bench --label turbo-16aug          benchmark whatever is loaded
    tiiny-bench --all                        sweep every text model on the box
    tiiny-bench --model Qwen/Qwen3-8B        benchmark one model by name
    tiiny-bench --only prefill,concurrency   run a subset of the tests
    tiiny-bench --catalog                    list what is installed
    tiiny-bench --report                     build the HTML report from results

Measures the things a spec sheet does not: how prefill scales with prompt
length, whether throughput holds over a long generation, what happens when more
than one person uses the box at once, and what a reasoning model's hidden tokens
actually cost. Records NPU utilisation and memory alongside every number,
because tokens/sec without the load context is not evidence.

BY DEFAULT NOTHING IS LOADED OR UNLOADED. The suite runs against whatever model
is already running and leaves the box exactly as it found it. `--all` and
`--model` are the two flags that change that, and they say so before they start.
"""
import argparse
import glob
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import os

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "bench-results"

HOST = os.environ.get("TIINY_HOST", "tiiny.local")
GW = int(os.environ.get("TIINY_PORT", "8800"))

# Everything the suite says out loud goes through say(). The CLI prints it;
# the server hands it a sink as well so the same words stream to the browser.
# One indirection, so the tests below never have to know which one is watching.
SINK = None


def say(line=""):
    print(line)
    if SINK:
        try:
            SINK(line)
        except Exception:
            pass


def emit(kind, **data):
    """Structured progress, for the web UI's benefit. The CLI ignores it."""
    if SINK:
        try:
            SINK(None, kind, data)
        except Exception:
            pass


IMAGE_PROMPTS = [
    "a lighthouse on a rocky shore at dusk, painterly",
    "a bowl of oranges on a wooden table, soft window light",
    "a fox asleep under a fern, children's book illustration",
]
SPEECH_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Engineers measured the throughput carefully and recorded every result, "
    "then compared it against the figure printed on the box.",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief.",
]

# Deterministic filler so prompt lengths are repeatable across runs.
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Engineers measured the throughput carefully and recorded every result. ")

# The suite talks to chat completions, so these are the model types it can
# actually exercise. Everything else on the box is listed but skipped, with the
# reason shown, rather than silently dropped.
# Every class the suite has a test for. Anything else is listed and skipped
# with the reason shown, rather than silently dropped.
CHAT_TYPES = {"Text Generation", "Image-Text-to-Text", "Text-to-Image",
              "Text-to-Speech", "Text Embedding"}


def key():
    """TIINY_KEY if you set it. Otherwise dig it out of TiinyOS on this Mac.

    The scrape is a convenience for the machine running TiinyOS and nothing
    more: it reads the app's own local storage, tries each UUID it finds, and
    keeps the first one the device accepts."""
    env = os.environ.get("TIINY_KEY", "").strip()
    if env:
        return env
    hits = subprocess.run(
        ["grep", "-aoh", r"[0-9a-f]\{8\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{12\}"]
        + glob.glob(str(pathlib.Path.home() /
                       "Library/Application Support/TiinyOS/Local Storage/leveldb/*.ldb")),
        capture_output=True, text=True).stdout
    for c in dict.fromkeys(h.strip() for h in hits.split() if h.strip()):
        if "_error" not in api(f"http://{HOST}:{GW}/api/v1/models/running", c):
            return c
    sys.exit("No API key. Set TIINY_KEY, or run this on the Mac running TiinyOS.")


def api(url, tok, body=None, timeout=900, method=None):
    req = urllib.request.Request(
        url, method=method or ("POST" if body is not None else "GET"),
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {tok}",
                 **({"Content-Type": "application/json"} if body else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        return {"_error": str(e)[:140]}


def api_raw(url, tok, body=None, timeout=300):
    """Same as api() but for endpoints that hand back a file, not JSON."""
    req = urllib.request.Request(
        url, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Bearer {tok}",
                 **({"Content-Type": "application/json"} if body else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        return {"_error": str(e)[:140]}


def telemetry(tok):
    s = api(f"http://{HOST}/api/v1/sys/status", tok, timeout=20)
    n = (s.get("npus") or [{}])[0]
    cpu = s.get("cpu") or {}
    return {
        "npu_util_pct": n.get("utilization_percent"),
        "npu_mem_used_mb": n.get("memory_used_mb"),
        "npu_mem_total_mb": n.get("memory_total_mb"),
        "cpu_total_pct": cpu.get("total_percent"),
    }


def chat(tok, model, prompt, max_tokens, thinking=False):
    body = {"model": model, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    v = api(f"http://{HOST}:{GW}/v1/chat/completions", tok, body)
    wall = time.time() - t0
    if "_error" in v:
        return {"error": v["_error"], "wall_s": round(wall, 2)}
    t = v.get("timings") or {}
    u = v.get("usage") or {}
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": u.get("prompt_tokens", t.get("prompt_n", 0)),
        "out_tokens": u.get("completion_tokens", t.get("predicted_n", 0)),
        "prefill_tok_s": round(t.get("prompt_per_second") or 0, 2),
        "decode_tok_s": round(t.get("predicted_per_second") or 0, 2),
        "prefill_ms": round(t.get("prompt_ms") or 0, 1),
        "ttft_s": round((t.get("prompt_ms") or 0) / 1000
                        + (t.get("predicted_per_token_ms") or 0) / 1000, 3),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }


# ------------------------------------------------------------------ catalog
def catalog(tok):
    """Everything installed on the box, from the box. There is no hand-kept
    model list in this repo on purpose - it would be wrong within a week."""
    d = api(f"http://{HOST}:{GW}/api/v1/models", tok, timeout=60)
    if "_error" in d:
        sys.exit(f"could not read the model catalog: {d['_error']}")
    out = []
    for m in d.get("data", []):
        row = {k: m.get(k) for k in
               ("id", "display_name", "params", "type", "npu_usage",
                "total_size", "thinking", "version")}
        # At least one model ships its params field with a trailing newline,
        # which turns any table built from this into a mess.
        for k, v in row.items():
            if isinstance(v, str):
                row[k] = " ".join(v.split())
        out.append(row)
    return sorted(out, key=lambda m: -(m.get("total_size") or 0))


def running(tok):
    return (api(f"http://{HOST}:{GW}/api/v1/models/running", tok, timeout=30)
            .get("running") or [])


def npu_free(tok):
    s = api(f"http://{HOST}:{GW}/api/v1/models/npu/status", tok, timeout=30)
    return s.get("npu_available"), s.get("npu_total")


def load(tok, model, poll_s=420):
    """Start a model and wait for it to actually answer.

    The runtime is listed in /running before it is settled, and a call made in
    that window comes back 502. Two seconds of patience here is cheaper than a
    failed benchmark that looks like a slow model."""
    if model in running(tok):
        return True
    enc = urllib.parse.quote(model, safe="")
    say(f"    loading {model} ...")
    t0 = time.time()
    api(f"http://{HOST}:{GW}/api/v1/models/{enc}/start", tok, body={}, timeout=poll_s)
    while time.time() - t0 < poll_s:
        if model in running(tok):
            time.sleep(2.0)
            say(f"    up in {time.time() - t0:.0f}s")
            return True
        time.sleep(4.0)
    say("    TIMED OUT")
    return False


def unload(tok, model):
    enc = urllib.parse.quote(model, safe="")
    api(f"http://{HOST}:{GW}/api/v1/models/{enc}/stop", tok, body={}, timeout=180)
    time.sleep(1.5)


# ---------------------------------------------------------------- tests
def t_prefill(tok, model):
    """How fast does it ingest a document? Prefill is what long context costs."""
    say("\n  PREFILL SCALING  (document ingestion)")
    say(f"    {'approx tokens':>14} {'prefill tok/s':>15} {'prefill ms':>12}")
    rows = []
    for reps in (2, 12, 60, 240):
        prompt = (FILLER * reps) + "\n\nReply with the single word: ok"
        r = chat(tok, model, prompt, 4)
        if "error" in r:
            say(f"    {reps:>14} FAILED {r['error'][:50]}"); continue
        rows.append(r)
        say(f"    {r['prompt_tokens']:>14} {r['prefill_tok_s']:>15.2f} {r['prefill_ms']:>12.1f}")
    return rows


def t_sustained(tok, model, total=1500):
    """Does throughput hold, or does it sag as the box heats and KV grows?

    Utilisation is sampled on a thread WHILE the generation runs. Reading it
    once the request returns only ever catches the box going idle again, which
    is how an early version of this suite reported 0% NPU under load."""
    say(f"\n  SUSTAINED GENERATION  ({total} tokens, one unbroken request)")
    before = telemetry(tok)
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.wait(1.0):
            t = telemetry(tok)
            if t.get("npu_util_pct") is not None:
                samples.append(t)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        r = chat(tok, model,
                 "Write a detailed technical explanation of how speculative decoding works "
                 "in large language model inference. Cover the draft model, verification, "
                 "acceptance rates, and why throughput varies with content.", total)
    finally:
        stop.set()
        th.join(timeout=5)
    after = telemetry(tok)
    if "error" in r:
        say(f"    FAILED {r['error'][:70]}"); return None

    util = [s["npu_util_pct"] for s in samples]
    mem = [s.get("npu_mem_used_mb") or 0 for s in samples]
    during = {
        "samples": len(samples),
        "npu_util_peak": max(util) if util else None,
        "npu_util_median": round(statistics.median(util), 1) if util else None,
        "npu_mem_peak_mb": max(mem) if mem else None,
        "npu_mem_total_mb": (samples[0].get("npu_mem_total_mb") if samples
                             else after.get("npu_mem_total_mb")),
    }
    say(f"    generated {r['out_tokens']} tokens in {r['wall_s']}s at {r['decode_tok_s']} tok/s")
    say(f"    NPU util while running: median {during['npu_util_median']}% "
          f"peak {during['npu_util_peak']}%  ({during['samples']} samples)")
    say(f"    NPU mem peak {during['npu_mem_peak_mb']}/{during['npu_mem_total_mb']} MB")
    return {"run": r, "telemetry_before": before, "telemetry_after": after,
            "during": during}


def t_concurrency(tok, model, levels=(1, 2, 4, 8), per=160):
    """Aggregate throughput as more people use the box at the same time."""
    say(f"\n  CONCURRENCY  ({per} tokens per request)")
    say(f"    {'parallel':>9} {'aggregate tok/s':>17} {'per-stream':>12} {'wall s':>9}")
    rows = []
    for n in levels:
        res, errs = [], []
        def worker(i):
            r = chat(tok, model,
                     f"Explain concept number {i}: why memory bandwidth limits "
                     f"token generation on edge devices. Be specific.", per)
            (errs if "error" in r else res).append(r)
        t0 = time.time()
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.time() - t0
        if not res:
            say(f"    {n:>9} all failed"); continue
        agg = sum(x["out_tokens"] for x in res) / wall
        per_stream = statistics.median(x["decode_tok_s"] for x in res)
        rows.append({"parallel": n, "aggregate_tok_s": round(agg, 2),
                     "per_stream_tok_s": round(per_stream, 2),
                     "wall_s": round(wall, 2), "ok": len(res), "failed": len(errs)})
        say(f"    {n:>9} {agg:>17.2f} {per_stream:>12.2f} {wall:>9.2f}")
    return rows


def t_thinking(tok, model):
    """A reasoning model's hidden tokens are not free. What do they cost?"""
    say("\n  REASONING COST  (same prompt, thinking on vs off)")
    q = ("A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph. "
         "When does the second catch the first?")
    out = {}
    for name, flag in (("off", False), ("on", True)):
        r = chat(tok, model, q, 700, thinking=flag)
        if "error" in r:
            say(f"    thinking {name:<3} FAILED"); continue
        out[name] = r
        say(f"    thinking {name:<3} {r['out_tokens']:>4} tokens  "
              f"{r['wall_s']:>6.2f}s  {r['decode_tok_s']:>6.2f} tok/s")
    if "on" in out and "off" in out and out["off"]["wall_s"]:
        say(f"    -> reasoning costs {out['on']['wall_s'] / out['off']['wall_s']:.1f}x the wall time")
    return out


def t_image(tok, model):
    """Seconds per 512x512 plate. The only figure an image model is judged by
    on this box, because 512 is the only size the firmware will render."""
    say("\n  IMAGE GENERATION  (512x512, 8 steps)")
    rows = []
    for i, prompt in enumerate(IMAGE_PROMPTS):
        t0 = time.time()
        raw = api_raw(f"http://{HOST}:{GW}/v1/image/generate", tok,
                      {"model": model, "prompt": prompt, "negative_prompt": "",
                       "width": 512, "height": 512, "seed": 1000 + i, "steps": 8},
                      timeout=300)
        wall = time.time() - t0
        if not isinstance(raw, (bytes, bytearray)):
            say(f"    image {i+1} FAILED {str(raw)[:60]}"); continue
        rows.append({"wall_s": round(wall, 2), "bytes": len(raw), "seed": 1000 + i})
        say(f"    image {i+1}  {wall:6.2f}s  {len(raw)//1024:>5} KB")
    if not rows:
        return None
    med = statistics.median(r["wall_s"] for r in rows)
    say(f"    median {med:.2f}s per plate")
    return {"runs": rows, "s_per_image": round(med, 2)}


def _wav_seconds(raw):
    """Duration out of a RIFF header, so the real-time factor is measured
    rather than guessed from a character count."""
    try:
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return None
        i, rate, ch, bits = 12, None, None, None
        while i + 8 <= len(raw):
            cid = raw[i:i+4]
            size = int.from_bytes(raw[i+4:i+8], "little")
            body = raw[i+8:i+8+size]
            if cid == b"fmt ":
                ch = int.from_bytes(body[2:4], "little")
                rate = int.from_bytes(body[4:8], "little")
                bits = int.from_bytes(body[14:16], "little")
            elif cid == b"data" and rate and ch and bits:
                return size / (rate * ch * max(1, bits // 8))
            i += 8 + size + (size & 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def t_speech(tok, model):
    """Real-time factor: seconds of audio produced per second of wall clock.
    Above 1.0 means it can talk faster than a person listens, which is the only
    threshold that matters for anything conversational."""
    say("\n  SPEECH  (real-time factor)")
    rows = []
    for i, text in enumerate(SPEECH_TEXTS):
        t0 = time.time()
        raw = api_raw(f"http://{HOST}:{GW}/v1/audio/speech", tok,
                      {"model": model, "input": text, "response_format": "wav"},
                      timeout=300)
        wall = time.time() - t0
        if not isinstance(raw, (bytes, bytearray)):
            say(f"    clip {i+1} FAILED {str(raw)[:60]}"); continue
        secs = _wav_seconds(raw)
        rtf = round(secs / wall, 2) if secs and wall else None
        rows.append({"chars": len(text), "wall_s": round(wall, 2),
                     "audio_s": round(secs, 2) if secs else None, "rtf": rtf})
        say(f"    clip {i+1}  {len(text):>4} chars  {wall:6.2f}s  "
            + (f"{secs:5.1f}s audio  {rtf}x real time" if secs else "(duration unknown)"))
    good = [r["rtf"] for r in rows if r.get("rtf")]
    if good:
        say(f"    median {statistics.median(good):.2f}x real time")
    if not rows:
        return None
    return {"runs": rows,
            "rtf": round(statistics.median(good), 2) if good else None}


def t_embed(tok, model):
    """Embeddings per second, at one, eight and thirty-two at a time."""
    say("\n  EMBEDDINGS  (throughput)")
    rows = []
    for n in (1, 8, 32):
        batch = [FILLER[:180] + f" item {i}" for i in range(n)]
        t0 = time.time()
        v = api(f"http://{HOST}:{GW}/v1/embeddings", tok,
                {"model": model, "input": batch}, timeout=180)
        wall = time.time() - t0
        if "_error" in v:
            say(f"    batch {n:>3} FAILED {v['_error'][:55]}"); continue
        data = v.get("data") or []
        dim = len((data[0] or {}).get("embedding") or []) if data else 0
        rate = round(len(data) / wall, 1) if wall else 0
        rows.append({"batch": n, "wall_s": round(wall, 3), "returned": len(data),
                     "dim": dim, "per_s": rate})
        say(f"    batch {n:>3}  {wall:6.3f}s  {rate:8.1f} emb/s  dim {dim}")
    if not rows:
        return None
    return {"runs": rows, "emb_per_s": max(r["per_s"] for r in rows),
            "dim": rows[0]["dim"]}


TESTS = {"prefill": t_prefill, "sustained": t_sustained,
         "concurrency": t_concurrency, "thinking": t_thinking,
         "image": t_image, "speech": t_speech, "embed": t_embed}

# Which tests mean anything for which kind of model.
SUITES = {
    "Text Generation":    ["prefill", "sustained", "concurrency", "thinking"],
    "Image-Text-to-Text": ["prefill", "sustained", "concurrency", "thinking"],
    "Text-to-Image":      ["image"],
    "Text-to-Speech":     ["speech"],
    "Text Embedding":     ["embed"],
}

# The headline figure per class: (key in results, unit, what it means).
CLASS_METRIC = {
    "Text Generation":    ("decode_tok_s", "tok/s", "sustained decode"),
    "Image-Text-to-Text": ("decode_tok_s", "tok/s", "sustained decode"),
    "Text-to-Image":      ("s_per_image", "s/img", "per 512 plate"),
    "Text-to-Speech":     ("rtf", "x", "faster than real time"),
    "Text Embedding":     ("emb_per_s", "emb/s", "embeddings per second"),
}
LOWER_IS_BETTER = {"s_per_image"}


def suite(tok, model, want, meta):
    """One model, all the requested tests, returned as a record."""
    tel = telemetry(tok)
    say(f"\n  ---- {model} " + "-" * max(0, 56 - len(model)))
    results = {}
    t0 = time.time()
    # A leaderboard that ranks a speech model by tokens per second is measuring
    # nothing. Each class only runs the tests that mean something for it.
    allowed = SUITES.get(meta.get("type"), [])
    todo = [n for n in want if n in allowed] or allowed
    skipped = [n for n in want if n not in allowed]
    if skipped:
        say(f"    skipping {', '.join(skipped)}: not meaningful for a "
            f"{meta.get('type')} model")
    for i, name in enumerate(todo):
        emit("test", model=model, test=name, index=i, total=len(todo))
        results[name] = TESTS[name](tok, model)
        emit("test_done", model=model, test=name, index=i, total=len(todo))
    return {
        "model": model,
        "params": meta.get("params"),
        "type": meta.get("type"),
        "npu_usage": meta.get("npu_usage"),
        "total_size": meta.get("total_size"),
        "elapsed_s": round(time.time() - t0, 1),
        "npu_mem_total_mb": tel.get("npu_mem_total_mb"),
        "results": results,
    }


def selfcheck(tok):
    """Prove the install works, in about ten seconds.

    Five things, in the order they fail for a newcomer: can we reach the box,
    does the key work, is anything loaded, does a real inference come back, and
    can we write a result. A bad key or an empty box should say so plainly here
    rather than halfway through a benchmark."""
    ok = True

    def step(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        say(f"  {'ok  ' if passed else 'FAIL'} {label:<34} {detail}")

    say(f"\n  TiinyBench selfcheck   device {HOST}:{GW}\n")

    t0 = time.time()
    info = api(f"http://{HOST}/api/v1/sys/device_info", tok, timeout=20)
    step("device reachable", "_error" not in info,
         info.get("_error", "") or f"TiinyOS {info.get('tiiny_os', '?')}  "
                                   f"{(time.time()-t0)*1000:.0f}ms")

    cat = api(f"http://{HOST}:{GW}/api/v1/models", tok, timeout=60)
    n = len(cat.get("data") or [])
    step("api key accepted", "_error" not in cat and n > 0,
         cat.get("_error", "") or f"{n} models installed")

    live = running(tok)
    free, total = npu_free(tok)
    step("something is loaded", bool(live),
         (", ".join(m.split("/")[-1] for m in live) or
          "nothing loaded - load a model in TiinyOS")
         + (f"   NPU {total - free}/{total}" if total else ""))

    target = None
    for m in live:
        meta = next((x for x in (cat.get("data") or []) if x.get("id") == m), {})
        if meta.get("type") in ("Text Generation", "Image-Text-to-Text"):
            target = m
            break
    if target:
        r = chat(tok, target, "Reply with the single word: ok", 8)
        step("inference returns", "error" not in r,
             r.get("error", "") or
             f"{target.split('/')[-1]}  {r.get('decode_tok_s', 0):.1f} tok/s  "
             f"{r.get('wall_s', 0):.2f}s")
    else:
        step("inference returns", False,
             "no chat model loaded; load one to time a real call")

    try:
        OUT.mkdir(exist_ok=True)
        probe = OUT / ".selfcheck"
        probe.write_text("ok")
        probe.unlink()
        step("results directory writable", True, str(OUT))
    except Exception as exc:  # noqa: BLE001
        step("results directory writable", False, str(exc)[:60])

    say("")
    if ok:
        say("  All good. Run one with:   tiiny-bench --label first-run")
        say("  Or open the app with:     tiiny-bench --serve")
    else:
        say("  Something above needs fixing before a benchmark will mean anything.")
    say("")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(prog="tiiny-bench")
    p.add_argument("--label", help="name this run (required unless --catalog/--report)")
    p.add_argument("--only", default="", help="comma list: " + ",".join(TESTS))
    p.add_argument("--all", action="store_true",
                   help="sweep every text model. LOADS AND UNLOADS MODELS.")
    p.add_argument("--model", help="benchmark one model by id. LOADS AND UNLOADS IT.")
    p.add_argument("--catalog", action="store_true", help="list what is installed")
    p.add_argument("--selfcheck", action="store_true",
                   help="prove the install works: reach the device, time one real call")
    p.add_argument("--report", action="store_true", help="build report.html from results")
    p.add_argument("--show", help="print a saved result file")
    p.add_argument("--serve", nargs="?", const=8425, type=int, metavar="PORT",
                   help="run the web app (default port 8425)")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)

    if a.show:
        print(json.dumps(json.loads(pathlib.Path(a.show).read_text()), indent=2)[:4000])
        return 0
    if a.report:
        import report
        out = report.build(OUT, HERE / "report.html")
        print(f"  wrote {out}")
        return 0
    if a.serve:
        import serve
        return serve.run(a.serve)

    tok = key()

    if a.selfcheck:
        return selfcheck(tok)
    if a.catalog:
        rows = catalog(tok)
        live = set(running(tok))
        free, total = npu_free(tok)
        print(f"\n  {len(rows)} models installed   NPU {total - free}/{total} in use\n")
        print(f"  {'':1} {'params':>7} {'size':>7} {'npu':>4}  {'type':<20} id")
        for m in rows:
            mark = "*" if m["id"] in live else " "
            gb = (m.get("total_size") or 0) / 1e9
            print(f"  {mark} {str(m.get('params') or '-'):>7} {gb:>6.1f}G "
                  f"{str(m.get('npu_usage') or '-'):>4}  {(m.get('type') or '-')[:20]:<20} {m['id']}")
        print("\n  * currently running")
        return 0

    if not a.label:
        p.error("--label is required")

    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(TESTS)
    for w in want:
        if w not in TESTS:
            p.error(f"unknown test {w!r}; known: {', '.join(TESTS)}")

    info = api(f"http://{HOST}/api/v1/sys/device_info", tok, timeout=20)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec = {"label": a.label, "stamp": stamp, "build": info.get("tiiny_os"),
           "host": HOST, "suite_version": 2, "models": []}
    path = OUT / f"{stamp}-suite-{a.label}.json"

    cat = {m["id"]: m for m in catalog(tok)}
    was_running = running(tok)

    # ---- which models, and are we allowed to touch the box? ----------------
    if a.all:
        targets = [m for m in cat.values() if m.get("type") in CHAT_TYPES]
        skipped = [m for m in cat.values() if m.get("type") not in CHAT_TYPES]
        print(f"\n  SWEEP: {len(targets)} text models, one at a time.")
        print(f"  This LOADS AND UNLOADS models. Currently running: "
              f"{', '.join(was_running) or 'nothing'}")
        print(f"  Skipping {len(skipped)} non-chat models "
              f"({', '.join(sorted({m['type'] for m in skipped}))})")
        print(f"  Expect roughly {len(targets) * 7} minutes. Results are written after "
              f"every model, so this is safe to interrupt.\n")
    elif a.model:
        if a.model not in cat:
            p.error(f"{a.model!r} is not installed. tiiny-bench --catalog lists what is.")
        targets = [cat[a.model]]
        print(f"\n  Benchmarking {a.model}. This loads it and unloads it after.\n")
    else:
        live = running(tok)
        if not live:
            sys.exit("no model loaded. Load one in TiinyOS, or pass --model / --all.")
        targets = [cat.get(live[0], {"id": live[0]})]
        print(f"\n  Benchmarking whatever is loaded: {live[0]}")
        print("  Nothing will be loaded or unloaded.\n")

    touching = bool(a.all or a.model)

    for i, meta in enumerate(targets, 1):
        model = meta["id"]
        emit("model", model=model, index=i, total=len(targets))
        print(f"\n  [{i}/{len(targets)}] {model}")
        if touching:
            for other in running(tok):
                if other != model:
                    print(f"    unloading {other} to make room")
                    unload(tok, other)
            if not load(tok, model):
                rec["models"].append({"model": model, "error": "failed to load"})
                path.write_text(json.dumps(rec, indent=2))
                continue
        try:
            rec["models"].append(suite(tok, model, want, meta))
        except KeyboardInterrupt:
            print("\n  interrupted; what finished is saved")
            break
        finally:
            # Checkpoint after EVERY model. A sweep is an hour long and a box
            # that reboots at minute 50 should not cost the whole run.
            path.write_text(json.dumps(rec, indent=2))
        if touching:
            unload(tok, model)

    # Put the box back the way we found it.
    if touching and was_running:
        print("\n  restoring what was loaded before:", ", ".join(was_running))
        for m in running(tok):
            if m not in was_running:
                unload(tok, m)
        for m in was_running:
            load(tok, m)

    print(f"\n  saved {path}")
    print(f"  build the report with:  {sys.argv[0]} --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
