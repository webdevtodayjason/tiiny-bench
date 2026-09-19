#!/usr/bin/env python3
"""The web app: `tiiny-bench --serve`.

A benchmark you have to read a man page to start is a benchmark most people
never run. This puts the same engine behind a page: it shows you what is on
your box, you pick what to measure, you press one button, and you watch the
numbers arrive. The report that comes out is the same self-contained file the
CLI writes.

Stdlib only, same as everything else here. One background thread runs the
suite; every line it says is fanned out to whatever browsers are watching.
"""
import datetime
import html
import json
import os
import pathlib
import queue
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bench  # the engine, imported under a stable name by run()

HERE = pathlib.Path(__file__).resolve().parent


# Model operations that outlive a request: model id -> what is happening.
MGMT = {}


class Bus:
    """Fan-out to every open browser. Bounded per subscriber: a tab that stops
    reading must not be able to grow this without limit."""

    def __init__(self):
        self.subs = []
        self.lock = threading.Lock()
        self.backlog = []

    def publish(self, kind, data):
        msg = (kind, data)
        with self.lock:
            self.backlog.append(msg)
            del self.backlog[:-400]
            dead = []
            for q in self.subs:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self.subs.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self.lock:
            for m in self.backlog[-120:]:
                try:
                    q.put_nowait(m)
                except queue.Full:
                    break
            self.subs.append(q)
        return q

    def drop(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


class State:
    def __init__(self):
        self.bus = Bus()
        self.thread = None
        self.running = False
        self.label = None
        self.started = 0
        self.model = None
        self.test = None
        self.progress = (0, 0)
        self.error = None
        self.last_file = None
        self.cancel = threading.Event()

    def snapshot(self):
        return {
            "running": self.running, "label": self.label, "model": self.model,
            "test": self.test, "index": self.progress[0], "total": self.progress[1],
            "elapsed_s": round(time.time() - self.started, 1) if self.started else 0,
            "error": self.error, "last_file": self.last_file,
        }


S = State()


def sink(line, kind=None, data=None):
    """What the engine calls. A plain line, or a structured progress event."""
    if kind:
        if kind == "model":
            S.model = data.get("model")
            S.progress = (data.get("index", 0), data.get("total", 0))
        elif kind == "test":
            S.test = data.get("test")
        elif kind == "test_done":
            S.test = None
        S.bus.publish(kind, data)
        S.bus.publish("state", S.snapshot())
    else:
        S.bus.publish("line", {"line": line})


def run_suite(label, models, tests):
    """The worker. Mirrors what `main()` does for --model / --all, but driven
    by the browser's choices instead of argv."""
    S.running = True
    S.error = None
    S.started = time.time()
    S.label = label
    S.bus.publish("state", S.snapshot())
    try:
        tok = bench.key()
        cat = {m["id"]: m for m in bench.catalog(tok)}
        was = bench.running(tok)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        info = bench.api(bench.mgmt("/api/v1/sys/device_info"), tok, timeout=20)
        rec = {"label": label, "stamp": stamp, "build": info.get("tiiny_os"),
               "host": bench.HOST, "suite_version": 2,
               "bench_version": bench.VERSION,
               "connection": bench.where(),
               "firmware": bench.firmware(info),
               "models": []}
        path = bench.OUT / f"{stamp}-suite-{label}.json"
        S.last_file = path.name

        targets = [cat[m] for m in models if m in cat]
        touching = not (len(targets) == 1 and targets[0]["id"] in was)

        for i, meta in enumerate(targets, 1):
            if S.cancel.is_set():
                bench.say("\n  stopped")
                break
            model = meta["id"]
            sink(None, "model", {"model": model, "index": i, "total": len(targets)})
            bench.say(f"\n  [{i}/{len(targets)}] {model}")
            if touching:
                for other in bench.running(tok):
                    if other != model:
                        bench.say(f"    unloading {other} to make room")
                        bench.unload(tok, other)
                if not bench.load(tok, model):
                    rec["models"].append({"model": model, "error": "failed to load"})
                    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                    continue
            try:
                rec["models"].append(bench.suite(tok, model, tests, meta))
            finally:
                # After every model, always. A sweep is long and a box that
                # reboots at minute fifty should not cost the whole run.
                path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                S.bus.publish("saved", {"file": path.name,
                                        "models": len(rec["models"])})
            if touching:
                bench.unload(tok, model)

        if touching and was:
            bench.say("\n  putting the box back: " + ", ".join(was))
            for m in bench.running(tok):
                if m not in was:
                    bench.unload(tok, m)
            for m in was:
                bench.load(tok, m)

        import report
        report.build(bench.OUT, HERE / "report.html")
        bench.say("\n  report rebuilt")
        S.bus.publish("finished", {"file": path.name})
    except SystemExit as exc:
        # The one thing a run stops for that is not a fault: no key, or no box.
        # bench.key() and connect() both say what to do in a sentence, and that
        # sentence is the whole error. Without this clause the thread died
        # silently and the page just went quiet, which is the worst of both.
        S.error = str(exc)
        bench.say(f"\n  STOPPED {S.error}")
        S.bus.publish("failed", {"error": S.error})
    except Exception as exc:  # noqa: BLE001
        S.error = f"{type(exc).__name__}: {exc}"
        bench.say(f"\n  FAILED {S.error}")
        traceback.print_exc()
        S.bus.publish("failed", {"error": S.error})
    finally:
        S.running = False
        S.model = S.test = None
        S.cancel.clear()
        S.bus.publish("state", S.snapshot())


# Awards. Every one is derived from a measurement, so they update themselves
# and nobody has to defend a hand-assigned badge. Each is (key, label, what it
# says, how to score, whether lower wins).
AWARDS = [
    ("fastest",     "Fastest",        "highest sustained decode in its class",
     "decode_tok_s", False),
    ("per_unit",    "Best per unit",  "most tokens per second per NPU unit",
     "per_unit", False),
    ("feather",     "Featherweight",  "best of the models that leave room for another",
     "per_unit", False),
    ("deepreader",  "Deep reader",    "ingests a long prompt fastest",
     "prefill_peak", False),
    ("cheapthink",  "Cheap thinker",  "reasoning costs it the least wall time",
     "reasoning_tax", True),
    ("painter",     "Fastest brush",  "quickest 512 plate",
     "s_per_image", True),
    ("voice",       "Best voice",     "most audio per second of wall clock",
     "rtf", False),
]

# What you would actually load it FOR. Same idea, phrased as a job.
JOBS = [
    ("chat",      "a chat app",        "Text Generation",    "decode_tok_s", False),
    ("vision",    "reading screenshots", "Image-Text-to-Text", "decode_tok_s", False),
    ("documents", "long documents",    None,                 "prefill_peak", False),
    ("pictures",  "pictures",          "Text-to-Image",      "s_per_image", True),
    ("speaking",  "talking out loud",  "Text-to-Speech",     "rtf", False),
    ("search",    "search and recall", "Text Embedding",     "emb_per_s", False),
]


def fits(models):
    """Whether a set of models can be resident at once on THIS box.

    Answers three separate questions, because they fail differently: is each
    one installed, do they add up to under a hundred units, and is there room
    right now given what is already loaded."""
    try:
        tok = bench.key()
        cat = {m["id"]: m for m in bench.catalog(tok)}
        live = bench.running(tok)
    except SystemExit as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    rows, missing, need = [], [], 0
    for m in models:
        c = cat.get(m)
        if not c:
            missing.append(m)
            rows.append({"model": m, "installed": False})
            continue
        u = c.get("npu_usage") or 0
        need += u
        rows.append({"model": m, "installed": True, "npu": u,
                     "loaded": m in live, "params": c.get("params"),
                     "type": c.get("type")})

    used_now = sum((cat.get(m, {}).get("npu_usage") or 0) for m in live)
    already = sum((cat.get(m, {}).get("npu_usage") or 0)
                  for m in models if m in live)
    # What starting this app would add on top of what is running.
    extra = need - already
    verdict = ("missing" if missing else
               "no" if need > 100 else
               "yes" if used_now + extra <= 100 else "after-unloading")
    return {
        "verdict": verdict,
        "models": rows,
        "missing": missing,
        "npu_needed": need,
        "npu_total": 100,
        "npu_used_now": used_now,
        "npu_free_now": 100 - used_now,
        "npu_extra_needed": max(0, extra),
        "loaded_now": live,
        "note": {
            "missing": "Some of these are not installed on this box yet.",
            "no": f"These want {need} units and the box has 100. They cannot all be resident.",
            "yes": "Fits, and there is room for it right now.",
            "after-unloading": ("Fits in the budget, but something loaded now has to come "
                                "out first."),
        }[verdict],
    }


# (figure, the results block it is found in), for every class whose suite is a
# single test. The multi-test classes are pulled apart above by hand, because
# their figures come from different blocks of the same record.
HEADLINE = sorted({(bench.CLASS_METRIC[c][0], names[0])
                   for c, names in bench.SUITES.items()
                   if len(names) == 1 and c in bench.CLASS_METRIC})


def leaderboard():
    """Best measured figures per model, grouped by what kind of model it is.

    One ranked list would be a lie: a speech model's real-time factor and a
    text model's tokens per second are not the same axis. Each class is scored
    on the number its users actually care about, and the awards are all derived
    from measurements so none of them has to be argued about."""
    import report as rep
    best = {}
    for r in rep.load(bench.OUT):
        m = r.get("model")
        if not m:
            continue
        res = r.get("results") or {}
        blank = {"model": m, "runs": 0, "last": "", "decode_tok_s": None,
                 "prefill_peak": None, "agg_peak": None, "reasoning_tax": None,
                 "ttft_s": None, "dim": None}
        # Every headline figure any class is ranked by, so a class added to
        # bench.SUITES appears here without an edit. The three that were
        # written out by hand went stale the moment a class was added, which is
        # how four of the box's nine classes came to rank as blank.
        blank.update({key: None for key, _, _ in bench.CLASS_METRIC.values()})
        e = best.setdefault(m, blank)
        e["runs"] += 1
        e["last"] = max(e["last"], r.get("stamp") or "")

        run = ((res.get("sustained") or {}) or {}).get("run") or {}
        if run.get("decode_tok_s"):
            e["decode_tok_s"] = max(e["decode_tok_s"] or 0, run["decode_tok_s"])
        pf = [x for x in (res.get("prefill") or []) if x.get("prefill_tok_s")]
        if pf:
            e["prefill_peak"] = max(e["prefill_peak"] or 0,
                                    max(x["prefill_tok_s"] for x in pf))
            first = min(pf, key=lambda x: x.get("prompt_tokens") or 0)
            if first.get("ttft_s"):
                e["ttft_s"] = min(e["ttft_s"] or 9e9, first["ttft_s"])
        cc = res.get("concurrency") or []
        if cc:
            e["agg_peak"] = max(e["agg_peak"] or 0,
                                max(c["aggregate_tok_s"] for c in cc))
        th = res.get("thinking") or {}
        if th.get("on") and th.get("off") and th["off"].get("wall_s"):
            e["reasoning_tax"] = round(th["on"]["wall_s"] / th["off"]["wall_s"], 2)
        # Which block of a record holds a class's figure: the class's own
        # single test, read straight off the registry rather than listed again.
        for key, src in HEADLINE:
            blk = res.get(src) or {}
            if blk.get(key) is not None:
                cur = e[key]
                lo = key in bench.LOWER_IS_BETTER
                e[key] = blk[key] if cur is None else (
                    min(cur, blk[key]) if lo else max(cur, blk[key]))
            if src == "embed" and blk.get("dim"):
                e["dim"] = blk["dim"]

    # A leaderboard is built out of result files on disk and needs no key at
    # all. The key is only for the extra columns, so no key means fewer columns
    # rather than no answer. SystemExit is listed because that is what
    # bench.key() raises, and it is not an Exception: catching only Exception
    # here dropped the connection, and the dashboard died with it.
    try:
        tok = bench.key()
        cat = {m["id"]: m for m in bench.catalog(tok)}
        live = bench.running(tok)
    except (Exception, SystemExit):  # noqa: BLE001
        cat, live = {}, []

    rows = []
    for m, e in best.items():
        c = cat.get(m, {})
        e["params"] = c.get("params")
        e["npu"] = c.get("npu_usage")
        e["size_gb"] = round((c.get("total_size") or 0) / 1e9, 1) or None
        e["type"] = c.get("type") or "unknown"
        e["loaded"] = m in live
        e["per_unit"] = (round(e["decode_tok_s"] / e["npu"], 3)
                         if e["decode_tok_s"] and e["npu"] else None)
        key, unit, meaning = bench.CLASS_METRIC.get(e["type"], (None, "", ""))
        e["metric_key"], e["metric_unit"], e["metric_means"] = key, unit, meaning
        e["score"] = e.get(key) if key else None
        e["awards"] = []
        rows.append(e)

    def win(pool, field, lower):
        vals = [r for r in pool if r.get(field) is not None]
        if not vals:
            return None
        return min(vals, key=lambda r: r[field]) if lower else max(vals, key=lambda r: r[field])

    # Awards, scoped to a class where the metric only exists inside one.
    by_class = {}
    for r in rows:
        by_class.setdefault(r["type"], []).append(r)
    for key, label, blurb, field, lower in AWARDS:
        pool = rows
        if key == "feather":
            pool = [r for r in rows if (r.get("npu") or 999) <= 32]
        if field in ("decode_tok_s", "per_unit", "prefill_peak", "reasoning_tax"):
            # these only mean something between text-ish models
            pool = [r for r in pool if r["type"] in
                    ("Text Generation", "Image-Text-to-Text")]
        w = win(pool, field, lower)
        if w and not any(a["key"] == key for a in w["awards"]):
            w["awards"].append({"key": key, "label": label, "why": blurb})

    jobs = []
    for key, label, cls, field, lower in JOBS:
        pool = [r for r in rows if cls is None or r["type"] == cls]
        w = win(pool, field, lower)
        if w:
            jobs.append({"key": key, "job": label, "model": w["model"],
                         "value": w.get(field), "field": field,
                         "npu": w.get("npu"), "type": w["type"]})

    # Sort inside each class by that class's own metric.
    for cls, pool in by_class.items():
        lower = bench.CLASS_METRIC.get(cls, (None,))[0] in bench.LOWER_IS_BETTER
        pool.sort(key=lambda r: (r["score"] is None,
                                 (r["score"] or 0) * (1 if lower else -1), r["model"]))
        for i, r in enumerate(pool, 1):
            r["rank"] = i

    order = ["Text Generation", "Image-Text-to-Text", "Text-to-Image",
             "Text-to-Speech", "Text Embedding"]
    classes = [{"type": c,
                "metric": bench.CLASS_METRIC.get(c, (None, "", ""))[1],
                "means": bench.CLASS_METRIC.get(c, (None, "", ""))[2],
                "lower": bench.CLASS_METRIC.get(c, (None,))[0] in bench.LOWER_IS_BETTER,
                "rows": by_class[c]}
               for c in order + [k for k in by_class if k not in order]
               if c in by_class]

    # Every class on the box, measured or not, so the gaps are visible.
    coverage = []
    for c in sorted({(m.get("type") or "unknown") for m in cat.values()}):
        installed = [m for m in cat.values() if m.get("type") == c]
        coverage.append({"type": c, "installed": len(installed),
                         "measured": len(by_class.get(c, [])),
                         "testable": c in bench.SUITES})

    return {"classes": classes, "jobs": jobs, "coverage": coverage,
            "npu_total": 100,
            "npu_used": sum((cat.get(m, {}).get("npu_usage") or 0) for m in live),
            "loaded": live, "catalog": list(cat.values())}


# ---------------------------------------------------------------- chat page
# The Chat page is the AINode Pocket 0.1.3 chat tab adapted to one Tiiny.
# Pocket puts one endpoint in front of a fleet, so wherever it iterates devices
# this iterates the single box TiinyBench is pointed at and the device chooser
# collapses to a label. Nothing else is dropped.

# What the page asks for when it is not told otherwise. A reasoning model spends
# this budget on its chain of thought before the answer, so a small number here
# is how you get an empty reply and a "length" stop reason.
DEFAULT_MAX_TOKENS = 700

# The last line of a relayed stream. The stats event goes in front of it, never
# after: every SSE client stops reading at [DONE], and one that stopped would
# never see the numbers.
DONE_LINE = b"data: [DONE]"

# One inference at a time is what the hardware does. Two browser tabs sending at
# once collide with device error 150004, so they queue here instead.
CHAT_LOCK = threading.Lock()

# Who answered, for the stats bar. identify() is two HTTP calls, and the answer
# changes about as often as the box is renamed.
_WHO = {"id": None, "name": None, "at": 0.0}


def who(ttl=60.0):
    """(device id, device name) for the box this app is pointed at.

    Both fall back to the address, and the address falls back to a phrase,
    because a stats bar reading "on  \u00b7 Ornith-1.0-35B" with a hole where
    the box should be is worse than one that says it does not know the name.
    """
    if not _WHO["name"] or time.time() - _WHO["at"] > ttl:
        try:
            info = bench.identify() or {}
        except Exception:  # noqa: BLE001
            info = {}
        _WHO["id"] = info.get("serial") or bench.HOST or "this Tiiny"
        _WHO["name"] = info.get("name") or bench.HOST or "this Tiiny"
        _WHO["at"] = time.time()
    return _WHO["id"], _WHO["name"]


def io_capabilities(wants, gives):
    """The device's input and output words for a model, as lists.

    The device answers with one word a side, not with lists: "Text" and "Text"
    for a chat model, "Text" and "Vector" for the embedding model. They are
    wrapped rather than reshaped, so nothing is invented and a caller has one
    shape to read. A record carrying neither gets null, because an empty list
    would say this model takes nothing, which is a different claim.
    """
    def listed(value):
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    ins, outs = listed(wants), listed(gives)
    if not ins and not outs:
        return None
    return {"input": ins, "output": outs}


def catalog_url(model_id):
    """A link out for the model card, or None.

    No device field carries a URL: the only links on the installed record and on
    the vendor catalogue are icon paths on the device itself. Model ids here are
    Hugging Face repository paths, though, so one owner and one name is a page
    that exists. This is a derivation from the shape of the id and not a field
    anybody sent, which is why anything else gets null rather than a guess.
    """
    parts = str(model_id or "").split("/")
    if len(parts) != 2 or not all(part.strip() and " " not in part for part in parts):
        return None
    return "https://huggingface.co/%s" % model_id


def split_done(blob):
    """(everything before the [DONE] line, the [DONE] line and what follows).

    The sentinel counts only as a whole line of its own. Ask a model about
    streaming and it writes "data: [DONE]" into its answer, where it reaches
    this function inside a frame's JSON string: a match anywhere in the bytes
    cut that frame in half, so the browser got something it could not parse, the
    answer stopped mid-sentence with nothing saying why, and the gateway's last
    chunk, the one carrying timings and usage, was never read. A JSON string
    cannot hold a raw newline, so a newline in this blob is always a frame
    boundary and anchoring to one is enough.
    """
    index = 0
    while True:
        index = blob.find(DONE_LINE, index)
        if index < 0:
            return blob, None
        starts_line = index == 0 or blob[index - 1:index] == b"\n"
        ends_line = blob[index + len(DONE_LINE):index + len(DONE_LINE) + 1] in (
            b"", b"\n", b"\r")
        if starts_line and ends_line:
            return blob[:index], blob[index:]
        index += len(DONE_LINE)


def sse_error(message):
    blob = json.dumps({"error": {"message": message, "type": "device_error"}})
    return ("data: %s\n\ndata: [DONE]\n\n" % blob).encode()


class ChatWatch:
    """Reads the frames going past on their way to the browser.

    The stream is relayed exactly as the device sent it, so the only way to know
    what was in it is to read the bytes on the way through. Three things are
    wanted: when the first token arrived by this machine's clock, why generation
    stopped, and the timings and usage blocks the gateway puts in the last chunk
    of a stream that asked for them.
    """

    def __init__(self, started):
        self.started = started
        self.ttft_ms = None
        self.finish_reason = None
        self.timings = {}
        self.usage = {}

    def read(self, blob):
        for line in blob.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                frame = json.loads(data)
            except ValueError:
                continue
            if not isinstance(frame, dict):
                continue
            if isinstance(frame.get("timings"), dict):
                self.timings = frame["timings"]
            if isinstance(frame.get("usage"), dict):
                self.usage = frame["usage"]
            # The last chunk of a stream carries the numbers and an empty
            # choices list, so nothing here may assume there is a choices[0].
            choices = frame.get("choices") or []
            first = choices[0] if choices and isinstance(choices[0], dict) else {}
            if first.get("finish_reason"):
                self.finish_reason = first["finish_reason"]
            delta = first.get("delta") or {}
            if self.ttft_ms is None and (delta.get("content")
                                         or delta.get("reasoning_content")):
                # A reasoning token counts as the first token. It is a predicted
                # token like any other and it is what the benchmark's ttft_s
                # measures; waiting for the answer instead would report nothing
                # at all for a thinking model that spent its whole budget
                # thinking, which is the common case with a modest cap.
                self.ttft_ms = round((time.time() - self.started) * 1000, 1)


def chat_relay(tok, request, device_name):
    """Yield the device's stream as valid server-sent events.

    Two things happen on the way through. A server-sent event ends with a blank
    line and the device sends one data: line per frame; that blank is put back,
    because what left here without it was one long frame only a lenient parser
    could read. And the device's own [DONE] is dropped, because the caller
    appends one after the stats event: a client stops reading at the first, so a
    relayed pair would put the numbers somewhere nobody looks.
    """
    saw_content = False
    for attempt in range(bench.BUSY_RETRIES):
        busy = False
        try:
            for line in bench.chat_lines(tok, request):
                stripped = line.strip()
                if not stripped:
                    continue
                if not saw_content and stripped.startswith("{"):
                    # A busy refusal can arrive as a bare JSON body rather than
                    # as an SSE frame.
                    try:
                        probe = json.loads(stripped)
                    except ValueError:
                        probe = None
                    if isinstance(probe, dict) and probe.get("code") == bench.BUSY_CODE:
                        busy = True
                        break
                saw_content = True
                if stripped == "data: [DONE]":
                    continue
                if stripped.startswith("data:"):
                    yield (stripped + "\n\n").encode()
                else:
                    # A field line that is not data: (event:, id:) belongs to
                    # the frame that follows it and stays single spaced.
                    yield (stripped + "\n").encode()
        except bench.DeviceError as exc:
            yield sse_error(
                "%s: %s. The device gateway closes a single request at about 220 "
                "seconds; the tokens already streamed above are real."
                % (device_name, exc))
            return
        if not busy:
            yield b"data: [DONE]\n\n"
            return
        if saw_content or attempt == bench.BUSY_RETRIES - 1:
            yield sse_error("%s reported error 150004 (busy) through %d attempts. "
                            "Something outside this app is using the device; 150004 "
                            "means busy, not a bad request."
                            % (device_name, attempt + 1))
            return
        time.sleep(bench.BUSY_BACKOFF * (attempt + 1))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # ---- helpers --------------------------------------------------------
    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _file(self, path: pathlib.Path, ctype):
        if not path.exists():
            return self._send(404, "not here", "text/plain")
        self._send(200, path.read_bytes(), ctype)

    # ---- GET ------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(HERE / "static" / "app.html", "text/html; charset=utf-8")
        if p == "/report":
            r = HERE / "report.html"
            if not r.exists():
                import report
                report.build(bench.OUT, r)
            return self._file(r, "text/html; charset=utf-8")
        if p == "/api/state":
            return self._json(S.snapshot())
        if p == "/api/device":
            # Everything the settings panel needs to describe the connection,
            # and nothing that would leak the key back out to the page.
            key, how = "", "none"
            try:
                key = bench.key()
                how = bench.KEY_SOURCE or "saved"
            except SystemExit:
                pass
            w = bench.where()
            return self._json({
                "host": w["host"], "port": w["gateway_port"],
                "candidates": list(bench.PROXY_HOSTS),
                "device": bench.identify(),
                "source": w["source"],
                "plane": w["plane"],
                "serial": w["serial"],
                "transport": w["gateway_transport"],
                "vhost": w["gateway_vhost"],
                "key_set": bool(key),
                "key_hint": (key[:4] + "\u2026" + key[-4:]) if key else "",
                "key_source": how,
                "config": str(bench.CONFIG),
            })
        if p == "/api/manage":
            try:
                tok = bench.key()
            except SystemExit as e:
                return self._json({"error": str(e)}, 400)
            free, total = bench.npu_free(tok)
            st = bench.storage(tok)
            # The vendor catalogue is read off the box every time this route is
            # called; nothing about it is pinned or kept in this repo. Say when
            # it was read, and say so out loud when the read failed, because a
            # silent empty list reads as an empty store.
            online, online_error = [], None
            read_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                online = bench.online(tok)
            except bench.DeviceError as exc:
                online_error = str(exc)
            return self._json({
                "installed": bench.catalog(tok),
                "online": online,
                "online_error": online_error,
                "online_read_at": read_at,
                "running": bench.running(tok),
                "npu": {"free": free, "total": total},
                "storage": {"total": st.get("total_bytes"),
                            "used": st.get("used_bytes"),
                            "free": st.get("remaining_bytes")},
                "telemetry": bench.telemetry(tok),
                "busy": dict(MGMT),
            })
        if p == "/api/manage/progress":
            mid = urllib.parse.parse_qs(u.query).get("model", [""])[0]
            if not mid:
                return self._json({"error": "which model?"}, 400)
            try:
                tok = bench.key()
            except SystemExit as e:
                return self._json({"error": str(e)}, 400)
            return self._json(bench.download_progress(tok, mid))
        if p == "/api/catalog":
            try:
                tok = bench.key()
                return self._json({"models": bench.catalog(tok),
                                   "running": bench.running(tok),
                                   "host": bench.HOST})
            except SystemExit as e:
                return self._json({"error": str(e)}, 503)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 503)
        if p.startswith("/assets/"):
            name = p[len("/assets/"):]
            if "/" in name or ".." in name:
                return self._send(404, "no", "text/plain")
            f = HERE / "assets" / name
            ctype = ("image/svg+xml" if name.endswith(".svg")
                     else "image/png" if name.endswith(".png") else "application/octet-stream")
            return self._file(f, ctype)
        if p == "/api/fit":
            # What tiinyapp.farm asks on an app's behalf: "will this run on
            # MY box?" The farm cannot know that - it does not know what is
            # installed or what is already loaded. This does.
            q = urllib.parse.parse_qs(u.query)
            want = [m for m in (q.get("models") or [""])[0].split(",") if m]
            return self._json(fits(want))
        if p == "/api/leaderboard":
            return self._json(leaderboard())
        if p == "/api/runs":
            runs = []
            for f in sorted(bench.OUT.glob("*.json"), reverse=True):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                ms = ([m.get("model") for m in d.get("models", [])]
                      or ([d.get("model")] if d.get("model") else []))
                runs.append({"file": f.name, "label": d.get("label"),
                             "stamp": d.get("stamp"), "models": [m for m in ms if m]})
            return self._json({"runs": runs})
        if p == "/api/model_card":
            mid = urllib.parse.parse_qs(u.query).get("model", [""])[0]
            if not mid:
                return self._json({"error": {"message": "model is required"}}, 400)
            try:
                card = self.model_card(mid)
            except SystemExit as e:
                return self._json({"error": {"message": str(e)}}, 400)
            except bench.DeviceError as e:
                return self._json({"error": {"message": str(e)}}, 502)
            if card is None:
                return self._json({"error": {
                    "message": "this device has no model called %r" % mid}}, 404)
            return self._json(card)
        if p == "/api/instances":
            try:
                return self._json(self.instances())
            except SystemExit as e:
                return self._json({"error": {"message": str(e)}}, 400)
        if p == "/events":
            return self._sse()
        return self._send(404, "no such path", "text/plain")

    def _sse(self):
        q = S.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self._sse_send("state", S.snapshot())
            last = time.time()
            while True:
                try:
                    kind, data = q.get(timeout=1.0)
                    self._sse_send(kind, data)
                except queue.Empty:
                    if time.time() - last > 15:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            S.bus.drop(q)

    def _sse_send(self, kind, data):
        self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    # ---- chat -----------------------------------------------------------
    @staticmethod
    def chat_request(body):
        """The completion to send on, or a sentence saying what is wrong.

        The page could call /v1/chat/completions on the device directly, but
        then every number on the screen would be a guess made in the browser by
        a clock that never saw the device. This is the same completion with the
        controls the page owns folded in.
        """
        model_id = (body.get("model") or "").strip()
        if not model_id:
            return "field 'model' is required"
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return "field 'messages' must be a non-empty array"
        try:
            max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
        except (TypeError, ValueError):
            return "field 'max_tokens' must be a number"
        out = list(messages)
        system = (body.get("system") or "").strip()
        first = out[0] if isinstance(out[0], dict) else {}
        if system and first.get("role") != "system":
            out.insert(0, {"role": "system", "content": system})
        request = {
            "model": model_id, "messages": out, "max_tokens": max_tokens,
            # chat_template_kwargs is the only knob that turns reasoning on and
            # off on this gateway. Its own OpenAPI document declares a top level
            # enable_thinking as well, with thinking_enabled, reasoning_effort
            # and thinking_budget_tokens beside it, and the runtime ignores all
            # of them: measured, a request sending enable_thinking false at the
            # top level still got a chain of thought back.
            "chat_template_kwargs": {"enable_thinking": bool(body.get("thinking"))}}
        temperature = body.get("temperature")
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            request["temperature"] = float(temperature)
        if body.get("stream"):
            request["stream"] = True
            # Without this a stream carries no timings and no usage at all, and
            # the stats bar would have nothing but the browser's own clock.
            request["stream_options"] = {"include_usage": True}
        return request

    def refuse_chat(self, tok, model_id):
        """Why this model cannot answer, or None. Asked before the device is."""
        rows = bench.catalog_raw(tok)
        row = None
        for candidate in rows:
            if candidate.get("id") == model_id:
                row = candidate
                break
        live = bench.running(tok)
        chatty = [m.get("id") for m in rows
                  if m.get("id") in live
                  and bench.can_chat(m.get("type"), bench.capability_list(m))]
        if row is None:
            return (404, "this device has no model called %r. The Models page "
                         "lists what is installed." % model_id)
        if not bench.can_chat(row.get("type"), bench.capability_list(row)):
            phrase = bench.type_phrase(row.get("type"))
            head = ("%s is %s and cannot chat." % (model_id, phrase) if phrase
                    else "%s is not a chat model." % model_id)
            tail = ("Loaded chat models: %s." % ", ".join(chatty) if chatty else
                    "No chat model is loaded right now. Load one on the Models "
                    "page first.")
            return (400, "%s %s" % (head, tail))
        if model_id not in live:
            tail = (" Loaded chat models: %s." % ", ".join(chatty)) if chatty else ""
            return (503, "%s is installed but not loaded. Nothing auto-loads on "
                         "this hardware and this page does not load a model "
                         "behind your back to serve a chat, because a 35B takes "
                         "tens of seconds to come up.%s" % (model_id, tail))
        return None

    def page_chat(self, body):
        """POST /api/chat: a completion with the numbers attached."""
        if S.running:
            return self._json({"error": {
                "message": "a benchmark is running; a chat now would change what "
                           "it is measuring"}}, 409)
        request = self.chat_request(body)
        if isinstance(request, str):
            return self._json({"error": {"message": request}}, 400)
        model_id = request["model"]
        try:
            tok = bench.key()
        except SystemExit as e:
            return self._json({"error": {"message": str(e)}}, 400)
        try:
            refusal = self.refuse_chat(tok, model_id)
        except bench.DeviceError as e:
            return self._json({"error": {"message": str(e)}}, 502)
        if refusal:
            return self._json({"error": {"message": refusal[1]}}, refusal[0])

        dev_id, dev_name = who()
        device = {"id": dev_id, "name": dev_name}
        started = time.time()

        if not request.get("stream"):
            with CHAT_LOCK:
                try:
                    status, payload = bench.chat_once(tok, request)
                except bench.DeviceError as e:
                    return self._json({"error": {"message":
                        "%s: %s. The device gateway closes a single request at "
                        "about 220 seconds; ask for less in one call, or tick "
                        "Stream so partial output survives." % (dev_name, e)}}, 502)
            if status != 200 or not isinstance(payload, dict):
                # Anything the device itself refused is a 502 whatever status it
                # used, so this app's own 400, 404, 409 and 503 keep meaning what
                # they say about the request rather than about the box. The
                # device's own sentence is what the page shows.
                message = "the device refused the completion"
                if isinstance(payload, dict):
                    err = payload.get("error")
                    if isinstance(err, dict):
                        message = err.get("message") or message
                    elif payload.get("message"):
                        message = payload["message"]
                return self._json({"error": {"message": "%s: %s" % (dev_name, message)}},
                                  502)
            choices = payload.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            payload["stats"] = bench.chat_stats(
                payload.get("timings"), payload.get("usage"),
                total_ms=(time.time() - started) * 1000, ttft_ms=None,
                finish_reason=choice.get("finish_reason"),
                device=device, model=model_id)
            return self._json(payload)

        watch = ChatWatch(started)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with CHAT_LOCK:
            try:
                finished = False
                for blob in chat_relay(tok, request, dev_name):
                    if finished:
                        continue
                    head, tail = split_done(blob)
                    if head:
                        watch.read(head)
                        self.wfile.write(head)
                        self.wfile.flush()
                    if tail is None:
                        continue
                    stats = bench.chat_stats(
                        watch.timings, watch.usage,
                        total_ms=(time.time() - started) * 1000,
                        ttft_ms=watch.ttft_ms, finish_reason=watch.finish_reason,
                        device=device, model=model_id)
                    self.wfile.write(
                        ("event: stats\ndata: %s\n\n" % json.dumps(stats)).encode())
                    self.wfile.write(DONE_LINE + b"\n\n")
                    self.wfile.flush()
                    finished = True
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        return None

    # ---- the model card and the instances rail ---------------------------
    def model_card(self, model_id):
        """What the card beside the conversation shows, or None if it is not here.

        Read from the device's own installed record rather than from the seven
        columns catalog() keeps, because input, output and the description live
        on that record and the shorter one throws them away.
        """
        tok = bench.key()
        entry = None
        for row in bench.catalog_raw(tok):
            if row.get("id") == model_id:
                entry = row
                break
        if entry is None:
            return None
        _, dev_name = who()
        loaded_on = []
        detail = bench.running_detail(tok)
        for inst in ((detail.get("instances") or {}).get("running") or []):
            if isinstance(inst, dict) and inst.get("model_id") == model_id:
                loaded_on.append({"device_name": dev_name,
                                  "status": inst.get("status") or "running"})
        if not loaded_on and model_id in (detail.get("running") or []):
            loaded_on.append({"device_name": dev_name, "status": "running"})
        return {"model": model_id,
                "name": entry.get("display_name") or entry.get("name") or model_id,
                "type": entry.get("type") or None,
                "params": entry.get("params") or None,
                "size_bytes": entry.get("total_size") or entry.get("size") or None,
                "npu_usage": entry.get("npu_usage"),
                "capabilities": io_capabilities(entry.get("input"), entry.get("output")),
                "loaded_on": loaded_on,
                "catalog_url": catalog_url(model_id),
                "can_chat": bench.can_chat(entry.get("type"),
                                           bench.capability_list(entry)),
                "desc": entry.get("desc") or None}

    def instances(self):
        """Every loaded model on the box, read live.

        Not through any cache: the load panel polls this to watch a model go
        from loading to running, and a cache would hide the move it exists to
        show. `reachable` is answered by whether these two calls came back just
        now.
        """
        tok = bench.key()
        dev_id, dev_name = who()
        record = {"device_id": dev_id, "device_name": dev_name,
                  "npu_used": None, "npu_total": None, "reachable": False}
        rows = []
        try:
            units = bench.npu_status(tok)
            detail = bench.running_detail(tok)
        except bench.DeviceError:
            return {"instances": rows, "devices": [record]}
        if not units:
            return {"instances": rows, "devices": [record]}
        record["npu_used"] = units.get("npu_used") or 0
        record["npu_total"] = units.get("npu_total") or 0
        record["reachable"] = True
        budget = {row.get("model_id"): row for row in (units.get("models") or [])
                  if isinstance(row, dict)}
        for inst in ((detail.get("instances") or {}).get("running") or []):
            if not isinstance(inst, dict):
                continue
            model_id = inst.get("model_id")
            spare = budget.get(model_id) or {}
            rows.append({"device_id": dev_id, "device_name": dev_name,
                         "model": model_id,
                         "npu_usage": inst.get("npu_usage") or spare.get("npu_usage"),
                         "status": inst.get("status") or spare.get("status")
                                   or "running",
                         "instance_id": inst.get("instance_id")
                                        or spare.get("instance_id")})
        return {"instances": rows, "devices": [record]}

    def refuse_load(self, tok, model_id, chat_required=True):
        """Why this model cannot be loaded, or None.

        The NPU budget is checked here rather than left to the device because
        the device does not refuse: a start that does not fit is accepted with
        the same 200 as any other, sits in npu/status as loading, and then
        vanishes with no error anywhere. This is the only place anybody gets
        told. It is not a guarantee either, because the subtraction is not the
        whole constraint on that hardware, which is why the panel polls after.
        """
        _, dev_name = who()
        row = None
        for candidate in bench.catalog_raw(tok):
            if candidate.get("id") == model_id:
                row = candidate
                break
        if row is None:
            return ("%s is not installed on %s. Download it on the Models page "
                    "first; nothing is fetched from the catalogue on demand."
                    % (model_id, dev_name))
        if chat_required and not bench.can_chat(row.get("type"),
                                                bench.capability_list(row)):
            phrase = bench.type_phrase(row.get("type"))
            what = ("is %s and cannot chat" % phrase) if phrase \
                else "is not a chat model"
            return ("%s %s, so loading it here would not give you anything to "
                    "talk to." % (model_id, what))
        units = bench.npu_status(tok)
        resident = [m.get("model_id") for m in (units.get("models") or [])
                    if isinstance(m, dict)]
        if model_id in resident:
            # Already there. The rail shows it, which is a better answer than
            # an error.
            return None
        total = units.get("npu_total") or 0
        used = units.get("npu_used") or 0
        cost = row.get("npu_usage") or 0
        if total and used + cost > total:
            return ("%s needs %d NPU units and only %d of %d are free on %s. The "
                    "device accepts a load that does not fit and then rolls it "
                    "back without saying so, so it is refused here instead."
                    % (model_id, cost, total - used, total, dev_name))
        return None

    def instance_action(self, action, body):
        model_id = (body.get("model") or "").strip()
        if not model_id:
            return self._json({"error": {"message": "model is required"}}, 400)
        if S.running:
            return self._json({"error": {
                "message": "a benchmark is running; loading a model now would "
                           "change what it is measuring"}}, 409)
        try:
            tok = bench.key()
        except SystemExit as e:
            return self._json({"error": {"message": str(e)}}, 400)
        enc = urllib.parse.quote(model_id, safe="")
        try:
            if action == "load":
                refusal = self.refuse_load(tok, model_id)
                if refusal:
                    return self._json({"error": {"message": refusal}}, 400)
                r = bench.api(bench.gw(f"/api/v1/models/{enc}/start"),
                              tok, body={}, timeout=120)
            else:
                if model_id not in bench.running(tok):
                    _, dev_name = who()
                    return self._json({"error": {
                        "message": "%s is not loaded on %s." % (model_id, dev_name)}},
                        400)
                r = bench.api(bench.gw(f"/api/v1/models/{enc}/stop"),
                              tok, body={}, timeout=180)
        except bench.DeviceError as e:
            return self._json({"error": {"message": str(e)}}, 502)
        if isinstance(r, dict) and "_error" in r:
            return self._json({"error": {"message": r["_error"]}}, 502)
        # 202 because the device has only accepted the job. Whether the model
        # comes up is answered by polling /api/instances, and by a chat that
        # works, not by this reply.
        return self._json({"ok": True}, 202)

    # ---- POST -----------------------------------------------------------
    def do_POST(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or "{}")
        except ValueError:
            body = {}
        if u.path == "/api/run":
            if S.running:
                return self._json({"error": "a run is already going"}, 409)
            models = body.get("models") or []
            tests = [t for t in (body.get("tests") or list(bench.TESTS))
                     if t in bench.TESTS]
            label = (body.get("label") or "run").strip()[:48] or "run"
            if not models:
                return self._json({"error": "pick at least one model"}, 400)
            if not tests:
                return self._json({"error": "pick at least one test"}, 400)
            S.thread = threading.Thread(target=run_suite, args=(label, models, tests),
                                        daemon=True)
            S.thread.start()
            return self._json({"ok": True, "label": label,
                               "models": len(models), "tests": tests})
        if u.path == "/api/device":
            host = (body.get("host") or "").strip()
            newkey = (body.get("key") or "").strip()
            if host:
                if not bench.reachable(host):
                    return self._json(
                        {"error": f"nothing answering at {host}. The gateway should "
                                  f"return 401 on /v1/models, on its own port or on "
                                  f"port 80 as p8800.api.tiiny; check the address."}, 400)
                bench.save_config(host=host, plane="given")
                bench.connect(host=host)
            elif body.get("rediscover"):
                w, err = bench.connect_soft(rescan=True)
                if not w:
                    return self._json({"error": err}, 404)
            if newkey:
                # Check it against the device before saving, so a typo is caught
                # here rather than surfacing as a confusing failure mid-run.
                probe = bench.api(bench.gw("/api/v1/models/running"), newkey, timeout=20)
                if "_error" in probe:
                    return self._json({"error": "the device rejected that key"}, 400)
                bench.save_config(key=newkey)
            w = bench.where()
            return self._json({"ok": True, "host": w["host"],
                               "port": w["gateway_port"],
                               "plane": w["plane"], "source": w["source"],
                               "transport": w["gateway_transport"],
                               "device": bench.identify()})
        if u.path == "/api/manage":
            if S.running:
                return self._json(
                    {"error": "a benchmark is running; loading a model now would "
                              "change what it is measuring"}, 409)
            action = (body.get("action") or "").strip()
            mid = (body.get("model") or "").strip()
            try:
                tok = bench.key()
            except SystemExit as e:
                return self._json({"error": str(e)}, 400)
            if action == "unload_all":
                bench.unload_all(tok)
                return self._json({"ok": True})
            if not mid:
                return self._json({"error": "which model?"}, 400)
            if action == "unload":
                bench.unload(tok, mid)
                return self._json({"ok": True})
            if action == "delete":
                r = bench.delete_model(tok, mid)
                if isinstance(r, dict) and "_error" in r:
                    return self._json({"error": r["_error"]}, 400)
                return self._json({"ok": True})
            if action == "download":
                r = bench.download(tok, mid)
                if isinstance(r, dict) and "_error" in r:
                    return self._json({"error": r["_error"]}, 400)
                MGMT[mid] = "downloading"
                return self._json({"ok": True, "watch": True})
            if action == "load":
                # The same guard the Chat page's load panel uses, minus the chat
                # requirement: this page loads speech, image and embedding models
                # on purpose. One guard rather than two, so the two doors onto
                # the device's start cannot drift apart and the budget refusal
                # is given here as well.
                try:
                    refusal = self.refuse_load(tok, mid, chat_required=False)
                except bench.DeviceError as e:
                    return self._json({"error": str(e)}, 502)
                if refusal:
                    return self._json({"error": refusal}, 400)
                # Loading a 35B model takes minutes. Do it on a thread and let
                # the page poll, rather than holding a request open that long.
                if mid in MGMT:
                    return self._json({"error": f"already {MGMT[mid]}"}, 409)
                MGMT[mid] = "loading"

                def _load(m=mid, t=tok):
                    try:
                        bench.load(t, m)
                    finally:
                        MGMT.pop(m, None)
                threading.Thread(target=_load, daemon=True).start()
                return self._json({"ok": True, "watch": True})
            return self._json({"error": f"unknown action {action!r}"}, 400)
        if u.path == "/api/chat":
            return self.page_chat(body)
        if u.path in ("/api/instances/load", "/api/instances/unload"):
            return self.instance_action(u.path.rsplit("/", 1)[1], body)
        if u.path == "/api/stop":
            # Cooperative: the current model finishes and is saved, then the
            # sweep stops. Killing mid-request would throw away a measurement
            # the box already paid for.
            S.cancel.set()
            return self._json({"ok": True, "note": "stopping after this model"})
        if u.path == "/api/report":
            import report
            report.build(bench.OUT, HERE / "report.html")
            return self._json({"ok": True})
        return self._send(404, "no such path", "text/plain")


def run(port=8425, host=None, serial=None, rescan=False):
    bench.OUT.mkdir(exist_ok=True)
    bench.SINK = sink
    # A box that cannot be found is not a reason to refuse to start: the
    # settings panel exists so somebody can type the address in.
    w, err = bench.connect_soft(host=host, serial=serial, rescan=rescan)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"\n  TiinyBench {bench.VERSION}  http://127.0.0.1:{port}/")
    if w:
        print(f"  device      {w['host']}  ({w['plane']} plane, "
              f"{w['gateway_transport']}, found by {w['source']})")
    else:
        print("  device      not found yet. Open the app and set the address.")
        print("              " + err.replace("\n", "\n              "))
    print(f"  results     {bench.OUT}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")
    return 0
