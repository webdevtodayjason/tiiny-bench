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
        info = bench.api(f"http://{bench.HOST}/api/v1/sys/device_info", tok, timeout=20)
        rec = {"label": label, "stamp": stamp, "build": info.get("tiiny_os"),
               "host": bench.HOST, "suite_version": 2, "models": []}
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
                    path.write_text(json.dumps(rec, indent=2))
                    continue
            try:
                rec["models"].append(bench.suite(tok, model, tests, meta))
            finally:
                # After every model, always. A sweep is long and a box that
                # reboots at minute fifty should not cost the whole run.
                path.write_text(json.dumps(rec, indent=2))
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


def leaderboard():
    """Best measured numbers per model, across every run on disk.

    Ranked by decode rate, but the column that matters on this box is tok/s per
    NPU unit: you have 100 units and the device runs one inference at a time,
    so the real question is never "which is fastest" but "which earns its
    place in the budget"."""
    import report as rep
    best = {}
    for r in rep.load(bench.OUT):
        m = r.get("model")
        if not m:
            continue
        res = r.get("results") or {}
        run = ((res.get("sustained") or {}) or {}).get("run") or {}
        dec = run.get("decode_tok_s")
        pf = [x.get("prefill_tok_s") or 0 for x in (res.get("prefill") or [])]
        cc = res.get("concurrency") or []
        th = res.get("thinking") or {}
        tax = None
        if th.get("on") and th.get("off") and th["off"].get("wall_s"):
            tax = round(th["on"]["wall_s"] / th["off"]["wall_s"], 2)
        e = best.setdefault(m, {"model": m, "runs": 0, "decode_tok_s": None,
                                "prefill_peak": None, "agg_peak": None,
                                "reasoning_tax": None, "last": ""})
        e["runs"] += 1
        e["last"] = max(e["last"], r.get("stamp") or "")
        if dec and (e["decode_tok_s"] is None or dec > e["decode_tok_s"]):
            e["decode_tok_s"] = dec
        if pf and (e["prefill_peak"] is None or max(pf) > e["prefill_peak"]):
            e["prefill_peak"] = max(pf)
        if cc:
            a = max(c["aggregate_tok_s"] for c in cc)
            if e["agg_peak"] is None or a > e["agg_peak"]:
                e["agg_peak"] = a
        if tax is not None:
            e["reasoning_tax"] = tax

    # Fold in what the box says each model costs to keep loaded.
    try:
        tok = bench.key()
        cat = {m["id"]: m for m in bench.catalog(tok)}
        live = bench.running(tok)
    except Exception:  # noqa: BLE001
        cat, live = {}, []
    rows = []
    for m, e in best.items():
        c = cat.get(m, {})
        e["params"] = c.get("params")
        e["npu"] = c.get("npu_usage")
        e["size_gb"] = round((c.get("total_size") or 0) / 1e9, 1) or None
        e["type"] = c.get("type")
        e["loaded"] = m in live
        e["per_unit"] = (round(e["decode_tok_s"] / e["npu"], 3)
                         if e["decode_tok_s"] and e["npu"] else None)
        rows.append(e)
    rows.sort(key=lambda r: (-(r["decode_tok_s"] or 0), r["model"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"rows": rows, "npu_total": 100,
            "npu_used": sum((cat.get(m, {}).get("npu_usage") or 0) for m in live),
            "loaded": live,
            "catalog": list(cat.values())}


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
        if p == "/api/leaderboard":
            return self._json(leaderboard())
        if p == "/api/runs":
            runs = []
            for f in sorted(bench.OUT.glob("*.json"), reverse=True):
                try:
                    d = json.loads(f.read_text())
                except ValueError:
                    continue
                ms = ([m.get("model") for m in d.get("models", [])]
                      or ([d.get("model")] if d.get("model") else []))
                runs.append({"file": f.name, "label": d.get("label"),
                             "stamp": d.get("stamp"), "models": [m for m in ms if m]})
            return self._json({"runs": runs})
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


def run(port=8425):
    bench.OUT.mkdir(exist_ok=True)
    bench.SINK = sink
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"\n  TiinyBench  http://127.0.0.1:{port}/")
    print(f"  device      {bench.HOST}")
    print(f"  results     {bench.OUT}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")
    return 0
