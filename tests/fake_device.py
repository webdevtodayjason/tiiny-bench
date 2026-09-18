"""A fake Tiiny, on an ephemeral port, for the Chat page's tests.

Ported from AINode Pocket 0.1.3 (pocket/fake.py), trimmed to the surface the
Chat page touches: the installed catalogue, what is running, the NPU budget,
start and stop, and chat completions streamed and not. Every shape here was
recorded off live firmware in that file and is reproduced rather than invented,
because a fake that answers more politely than the hardware is worse than none.

The two behaviours these tests exist for are both real and both counterintuitive:
a start that does not fit the NPU budget is accepted with the same 200 as one
that does and is then rolled back in silence, and reasoning is charged against
max_tokens before the answer is.
"""
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Unit costs and types are the measured ones. "main" is the device's own word
# for the chat runtime; every other capability is a model that will never answer
# a chat completion.
CATALOG = [
    {"id": "deepreinforce-ai/Ornith-1.0-35B", "type": "Image-Text-to-Text",
     "params": "35B", "size": 18_000_000_000, "npu_usage": 50,
     "capabilities": ["main"]},
    {"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo", "type": "Text Generation",
     "params": "30B-A3B", "size": 15_200_000_000, "npu_usage": 45,
     "capabilities": ["main"]},
    {"id": "openai/gpt-oss-20b", "type": "Text Generation",
     "params": "20B", "size": 12_000_000_000, "npu_usage": 30,
     "capabilities": ["main"]},
    {"id": "zai-org/GLM-4.7-Flash", "type": "Text Generation",
     "params": "9B", "size": 6_000_000_000, "npu_usage": 12,
     "capabilities": ["main"]},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "type": "Text-to-Speech",
     "params": "1.7B", "size": 2_400_000_000, "npu_usage": 7,
     "capabilities": ["voice"]},
    {"id": "Qwen/Qwen3-Embedding-0.6B", "type": "Text Embedding",
     "params": "0.6B", "size": 900_000_000, "npu_usage": 1,
     "capabilities": ["embedding"]},
]

# What a model eats and what it produces. The device sends these as two scalar
# words per model rather than as lists.
IO_BY_TYPE = {
    "text generation": ("Text", "Text"),
    "image-text-to-text": ("Image, Text", "Text"),
    "text-to-speech": ("Text", "Audio"),
    "text embedding": ("Text", "Vector"),
}

DEFAULT_INSTALLED = [row["id"] for row in CATALOG]
DEFAULT_LOADED = ["deepreinforce-ai/Ornith-1.0-35B",
                  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                  "Qwen/Qwen3-Embedding-0.6B"]

# Where this fake stops of its own accord, standing in for a model reaching the
# end of what it had to say. Asking for more than this gets a "stop"; asking for
# less gets a "length", the same way the real gateway answers.
TOKEN_CAP = 400
# Measured prompt-processing rate on real firmware.
PREFILL_TOK_S = 28.2
# How many polls of npu/status a model spends in "loading" before it is either
# running or gone. Real firmware takes tens of seconds; two polls is the same
# shape at a speed a test can wait for.
LOAD_POLLS = 2

SAMPLE = (
    "Memory bandwidth sets the ceiling here. The accelerator reads the active "
    "weights once per token, so tokens per second is bandwidth divided by bytes "
    "read per token, and no amount of scheduling changes that arithmetic. "
    "Adding a second caller does not add throughput, because the two requests "
    "are not batched: they are queued, and the queue is in the runtime rather "
    "than in the API layer, so it cannot be tuned away from the outside. ")
REASONING = ("Checking the bandwidth math first. Bytes read per token divided "
             "into memory bandwidth is the ceiling here, so that is the number "
             "to give them. ")


class FakeState:
    def __init__(self, installed=None, loaded=None, npu_total=100):
        self.serial = "TNYF26090000000001Q"
        self.name = "tiiny-fake"
        self.npu_total = npu_total
        self.installed = list(DEFAULT_INSTALLED if installed is None else installed)
        self.loaded = list(DEFAULT_LOADED if loaded is None else loaded)
        # Models asked for and still coming up. A pending model is already in
        # `loaded`, because the device reserves its units the moment it accepts
        # the start, but it reports "loading" and will not answer an inference.
        self.pending = {}
        self.guard = threading.Lock()
        self.chat_bodies = []
        # A reason phrase to fail the vendor catalogue read with, or None to
        # answer it. The store is a separate service behind the box and it does
        # go away on its own, which is the case the Models page has to be
        # honest about.
        self.online_fails = None

    def row(self, model_id):
        for row in CATALOG:
            if row["id"] == model_id:
                return row
        return {"id": model_id, "type": "Text Generation", "params": "?",
                "size": 0, "npu_usage": 1, "capabilities": ["main"]}

    def cost(self, model_id):
        return self.row(model_id)["npu_usage"]

    def units_used(self):
        return sum(self.cost(m) for m in self.loaded)

    def status_of(self, model_id):
        return "loading" if model_id in self.pending else "running"

    def io(self, model_id):
        return IO_BY_TYPE.get(str(self.row(model_id)["type"]).strip().lower(),
                              ("Text", "Text"))

    def advance_loads(self):
        """Move every pending load one step. Called by the npu/status poll.

        A start that does not fit the NPU budget comes back with the same 200
        "start loading" as one that does, shows up as loading, and then
        vanishes. No error is returned anywhere, and polling npu/status is the
        only way to tell the two apart, so the poll is what moves this on.
        """
        for model_id in list(self.pending):
            job = self.pending[model_id]
            job["polls"] -= 1
            if job["polls"] > 0:
                continue
            self.pending.pop(model_id, None)
            if job["rollback"] and model_id in self.loaded:
                self.loaded.remove(model_id)

    # ------------------------------------------------------------- payloads
    def models_payload(self):
        data = []
        for model_id in self.installed:
            row = self.row(model_id)
            short = model_id.split("/")[-1]
            wants, gives = self.io(model_id)
            data.append({
                "id": model_id, "model_id": model_id, "fullname": model_id,
                "name": short, "display_name": short, "object": "model",
                "params": row["params"], "type": row["type"],
                "size": row["size"], "total_size": row["size"],
                "npu_usage": row["npu_usage"],
                "capabilities": list(row["capabilities"]),
                "status": "downloaded", "input": wants, "output": gives,
                "desc": "%s, %s parameters, %d NPU units on this device."
                        % (short, row["params"], row["npu_usage"])})
        return {"object": "list", "data": data}

    def running_payload(self):
        instances = []
        for offset, model_id in enumerate(self.loaded):
            row = self.row(model_id)
            wants, gives = self.io(model_id)
            status = self.status_of(model_id)
            instances.append({"model_id": model_id, "port": 9098 + offset,
                              "npu_usage": self.cost(model_id),
                              "type": row["type"], "status": status,
                              "display_name": model_id.split("/")[-1],
                              "input": wants, "output": gives,
                              "instance_id": "fake-%d" % (abs(hash(model_id)) % 10000)})
        return {"running": list(self.loaded), "instances": {"running": instances}}

    def units_payload(self):
        with self.guard:
            self.advance_loads()
            loaded = list(self.loaded)
            models = [{"model_id": m, "npu_usage": self.cost(m),
                       "status": self.status_of(m)} for m in loaded]
        used = sum(self.cost(m) for m in loaded)
        return {"npu_total": self.npu_total, "npu_used": used,
                "npu_available": max(0, self.npu_total - used), "models": models}

    def catalogue_payload(self):
        """The vendor catalogue: everything downloadable, installed or not."""
        out = []
        for row in CATALOG:
            short = row["id"].split("/")[-1]
            out.append({"model_id": row["id"], "id": row["id"], "name": short,
                        "fullname": row["id"], "display_name": short,
                        "type": row["type"], "params": row["params"],
                        "size": row["size"], "npu_usage": row["npu_usage"],
                        "status": "downloaded" if row["id"] in self.installed
                                  else "not_downloaded"})
        return out

    def device_info_payload(self):
        return {"device_name": self.name, "device_model_name": "Tiiny AI Pocket Lab",
                "tiiny_os": "1.0.0", "sn": self.serial, "ram": "80 GB",
                "storage": "1 TB"}


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self):
        return self.server.state

    def log_message(self, *a):
        pass

    def _send(self, status, payload):
        blob = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except ValueError:
            return {}

    def _authed(self):
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer ") and auth[7:].strip():
            return True
        self._send(401, {"code": 401, "msg": "Unauthorized"})
        return False

    @staticmethod
    def _model_from(path, prefix, suffix=""):
        """Pull a URL-encoded model id out of a path.

        A fake that quietly accepted an unencoded slash would hide the single
        most common real bug, so an id arriving with a raw slash is rejected.
        """
        rest = path[len(prefix):]
        if suffix:
            if not rest.endswith(suffix):
                return None
            rest = rest[:-len(suffix)]
        if not rest or "/" in rest:
            return None
        return urllib.parse.unquote(rest)

    def do_GET(self):  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        state = self.state
        if path == "/api/v1/sys/device_info":
            return self._send(200, state.device_info_payload())
        if not self._authed():
            return None
        if path in ("/api/v1/models", "/api/v1/models/"):
            return self._send(200, state.models_payload())
        if path == "/api/v1/models/running":
            return self._send(200, state.running_payload())
        if path == "/api/v1/models/npu/status":
            return self._send(200, state.units_payload())
        if path == "/api/v1/models/online_models":
            if state.online_fails:
                # The reason phrase, because that is what urllib puts in the
                # HTTPError the caller reads.
                self.send_response(503, state.online_fails)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            return self._send(200, state.catalogue_payload())
        if path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": m, "object": "model"} for m in state.loaded]})
        return self._send(404, {"code": 404, "msg": "Not Found"})

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if not self._authed():
            return None
        if path == "/v1/chat/completions":
            return self._chat(self._body())
        prefix = "/api/v1/models/"
        for suffix, handler in (("/start", self._start), ("/stop", self._stop)):
            if path.startswith(prefix) and path.endswith(suffix):
                model_id = self._model_from(path, prefix, suffix)
                if model_id is None:
                    return self._send(404, {"code": 404, "msg": "Not Found"})
                return handler(model_id)
        return self._send(404, {"code": 404, "msg": "Not Found"})

    # ----------------------------------------------------------- lifecycle
    def _start(self, model_id):
        state = self.state
        with state.guard:
            if model_id not in state.installed:
                return self._send(400, {"code": 400, "msg": "Error starting model.",
                                        "detail": "%s is not downloaded" % model_id})
            if model_id in state.loaded:
                return self._send(200, {"message": "%s already running" % model_id,
                                        "progress": 100})
            # A start that does not fit the remaining budget is accepted with
            # this same 200, reserves its units, sits in npu/status as
            # "loading", and then disappears. No error is ever returned.
            over = state.units_used() + state.cost(model_id) > state.npu_total
            state.loaded.append(model_id)
            state.pending[model_id] = {"polls": LOAD_POLLS, "rollback": over}
        return self._send(200, {"message": "start loading %s" % model_id, "progress": 0})

    def _stop(self, model_id):
        state = self.state
        with state.guard:
            if model_id not in state.loaded:
                return self._send(400, {"code": 400, "msg": "Error stopping model.",
                                        "detail": "%s is not running" % model_id})
            state.loaded.remove(model_id)
            state.pending.pop(model_id, None)
        return self._send(200, {"removed_container_ids": ["fake"]})

    # ----------------------------------------------------------- inference
    def _chat(self, body):
        state = self.state
        model_id = body.get("model")
        with state.guard:
            state.chat_bodies.append(body)
        if model_id not in state.loaded or model_id in state.pending:
            # The recorded not-loaded shape. Models never auto-load, and one
            # that is still coming up refuses inference too.
            return self._send(404, {"error": {
                "code": 404, "message": '"%s" is not loaded.' % model_id,
                "type": "model_not_found"}})
        if body.get("stream"):
            return self._chat_stream(body, model_id)
        return self._chat_once(body, model_id)

    @staticmethod
    def thinking(body):
        """Whether this request asked for a chain of thought.

        Only chat_template_kwargs.enable_thinking is read, because that is the
        only knob the runtime honours: the gateway's own OpenAPI document
        declares a top level enable_thinking as well, and a request sending it
        gets reasoning back anyway.
        """
        kwargs = body.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
            return bool(kwargs["enable_thinking"])
        return False

    def plan(self, body):
        """(reasoning tokens, answer tokens, hit the cap).

        Reasoning is charged against max_tokens before the answer is, which is
        why a reasoning model with a small cap answers nothing at all, and why
        the fake spends the budget in the same order: a stats bar that counted
        only the answer would never add up to what the device charged.
        """
        want = max(1, int(body.get("max_tokens") or 64))
        budget = min(want, TOKEN_CAP)
        reason = []
        if self.thinking(body):
            reason = [word + " " for word in REASONING.split()][:budget]
        words = SAMPLE.split()
        answer = [words[i % len(words)] + " "
                  for i in range(max(0, budget - len(reason)))]
        return reason, answer, len(reason) + len(answer) >= want

    def timings(self, body, reason, answer, elapsed):
        prompt_n = self.prompt_tokens(body)
        predicted_n = max(1, len(reason) + len(answer))
        elapsed = max(elapsed, 0.001)
        return {"cache_n": 0,
                "prompt_n": prompt_n,
                "prompt_ms": round(prompt_n / PREFILL_TOK_S * 1000, 1),
                "prompt_per_second": PREFILL_TOK_S,
                "prompt_per_token_ms": round(1000 / PREFILL_TOK_S, 4),
                "predicted_n": predicted_n,
                "predicted_ms": round(elapsed * 1000, 3),
                "predicted_per_second": round(predicted_n / elapsed, 2),
                "predicted_per_token_ms": round(elapsed * 1000 / predicted_n, 2)}

    @staticmethod
    def usage(timings):
        prompt_n, predicted_n = timings["prompt_n"], timings["predicted_n"]
        return {"prompt_tokens": prompt_n, "completion_tokens": predicted_n,
                "total_tokens": prompt_n + predicted_n,
                "prompt_tokens_details": {"cached_tokens": timings["cache_n"]}}

    @staticmethod
    def prompt_tokens(body):
        """Roughly four characters per token, counted off the real prompt.

        A fixed number here would make a prompt-length measurement draw a flat
        line, which looks like a broken measurement rather than a fake one.
        """
        chars = 0
        for message in body.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                chars += len(content)
        return max(1, chars // 4)

    def _chat_once(self, body, model_id):
        reason, answer, hit_cap = self.plan(body)
        started = time.time()
        timings = self.timings(body, reason, answer, time.time() - started)
        message = {"role": "assistant", "content": "".join(answer).strip()}
        if reason:
            message["reasoning_content"] = "".join(reason).strip()
        return self._send(200, {
            "id": "chatcmpl-fake", "object": "chat.completion",
            "created": int(started), "model": model_id,
            # "length" when the answer ran into max_tokens, which on a reasoning
            # model with a modest cap is the ordinary case and not a failure.
            "choices": [{"index": 0, "finish_reason": "length" if hit_cap else "stop",
                         "message": message}],
            "usage": self.usage(timings), "timings": timings})

    def _sse(self, payload):
        self.wfile.write(("data: %s\n\n" % json.dumps(payload)).encode())
        self.wfile.flush()

    def _chat_stream(self, body, model_id):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        created = int(time.time())

        def frame(delta, finish=None):
            return {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                    "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        reason, answer, hit_cap = self.plan(body)
        options = body.get("stream_options")
        want_usage = bool(isinstance(options, dict) and options.get("include_usage"))
        try:
            # The opening frame carries a null content, not an empty string.
            # It is the shape that catches a renderer appending the delta
            # without checking it.
            self._sse(frame({"role": "assistant", "content": None}))
            started = time.time()
            for token in reason:
                self._sse(frame({"reasoning_content": token}))
            for token in answer:
                self._sse(frame({"content": token}))
            self._sse(frame({}, finish="length" if hit_cap else "stop"))
            if want_usage:
                # With stream_options.include_usage the gateway adds one last
                # chunk carrying timings and usage, and its choices list is
                # empty. Without it there is no such chunk at all, which is why
                # a caller that wants the numbers has to ask.
                timings = self.timings(body, reason, answer, time.time() - started)
                self._sse({"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                           "created": created, "model": model_id, "choices": [],
                           "timings": timings, "usage": self.usage(timings)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True
        return None


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state, host="127.0.0.1", port=0):
        self.state = state
        super().__init__((host, port), FakeHandler)


class FakeDevice:
    """A fake Tiiny on an ephemeral port."""

    def __init__(self, **kwargs):
        self.state = FakeState(**kwargs)
        self.server = FakeServer(self.state)
        self.port = self.server.server_address[1]
        self.host = "127.0.0.1"
        self.thread = None

    @property
    def serial(self):
        return self.state.serial

    @property
    def name(self):
        return self.state.name

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
