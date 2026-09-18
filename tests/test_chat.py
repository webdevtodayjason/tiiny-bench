"""The Chat page's own API: the numbers, the framing, the card and the refusals.

Everything here runs against the fake device in tests/fake_device.py, offline,
on stdlib unittest:

    /usr/bin/python3 -m unittest discover -s tests -t . -v

The numbers a chat reports are the device's own timings and usage read through
the benchmark's derivation, so these tests care as much about the two agreeing
as about the shapes being right. Ported from AINode Pocket 0.1.3
(tests/test_chat.py), which is where the measured pair below came from.
"""
import json
import os
import re
import sys
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Both, so this runs the way CI runs it (discover -s tests, modules loaded as
# top level) and the way a person runs it (discover -s tests -t ., loaded as
# tests.*). ROOT is for bench and serve, HERE is for the fake beside this file.
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bench  # noqa: E402
import serve  # noqa: E402
from fake_device import FakeDevice  # noqa: E402

# One completion measured on Jason's Tiiny on 2026-09-16, Qwen/Qwen3-8B through
# POST /v1/chat/completions. Every field below came off the wire; nothing here is
# a rounded-off invention, because the point of this pair is to prove the
# derivation against what the hardware actually sends.
MEASURED_TIMINGS = {"cache_n": 1, "predicted_ms": 439.307, "predicted_n": 10,
                    "predicted_per_second": 22.763124648594264,
                    "predicted_per_token_ms": 43.9307,
                    "prompt_ms": 122.471, "prompt_n": 18,
                    "prompt_per_second": 146.97356925312928,
                    "prompt_per_token_ms": 6.803944444444444}
MEASURED_USAGE = {"completion_tokens": 10, "prompt_tokens": 19,
                  "prompt_tokens_details": {"cached_tokens": 1},
                  "total_tokens": 29}

STAT_KEYS = ("ttft_ms", "prefill_ms", "decode_tok_s", "prefill_tok_s",
             "prompt_tokens", "out_tokens", "cached_tokens", "total_ms",
             "finish_reason", "device", "model")

CHAT_MODEL = "deepreinforce-ai/Ornith-1.0-35B"


class ServerCase(unittest.TestCase):
    """One fake Tiiny and one TiinyBench web app, both on ephemeral ports."""

    loaded = None
    installed = None

    def setUp(self):
        self.fake = FakeDevice(loaded=self.loaded, installed=self.installed).start()
        self.addCleanup(self.fake.stop)

        saved = (bench.HOST, dict(bench.SERVICES), dict(bench.TRANSPORT),
                 bench.PORT_OVERRIDE, dict(bench.DISCO_SEEN),
                 os.environ.get("TIINY_KEY"))

        def restore():
            bench.HOST, bench.PORT_OVERRIDE = saved[0], saved[3]
            bench.SERVICES.clear(), bench.SERVICES.update(saved[1])
            bench.TRANSPORT.clear(), bench.TRANSPORT.update(saved[2])
            bench.DISCO_SEEN.clear(), bench.DISCO_SEEN.update(saved[4])
            if saved[5] is None:
                os.environ.pop("TIINY_KEY", None)
            else:
                os.environ["TIINY_KEY"] = saved[5]
        self.addCleanup(restore)
        # The fake serves all three surfaces on one ephemeral port, so every
        # service is pointed at it with no vhost name: _attempts then has one
        # url to try and the port-then-vhost walk is out of the way. Real
        # firmware is what the discovery tests exercise.
        bench.HOST, bench.PORT_OVERRIDE = self.fake.host, None
        bench.SERVICES.clear()
        bench.SERVICES.update({name: (self.fake.port, None)
                               for name in ("gateway", "openai", "mgmt")})
        bench.TRANSPORT.clear()
        bench.DISCO_SEEN.clear()
        os.environ["TIINY_KEY"] = "test-key"
        # who() caches the device name for a minute; every case points at a new
        # port, so the cache has to go with it.
        serve._WHO.update({"id": None, "name": None, "at": 0.0})

        self.app = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        self.app.daemon_threads = True
        thread = threading.Thread(target=self.app.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.app.server_close)
        self.addCleanup(self.app.shutdown)
        self.base = "http://127.0.0.1:%d" % self.app.server_address[1]

    # ----------------------------------------------------------- helpers
    def request(self, path, method="GET", body=None):
        """(status, decoded JSON or text) from the web app."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw, status = resp.read().decode(), resp.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read().decode(), exc.code
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw

    def raw_stream(self, body):
        """The whole body of a streamed /api/chat, bytes as the browser sees it."""
        req = urllib.request.Request(self.base + "/api/chat", method="POST",
                                     data=json.dumps(body).encode())
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.assertIn("text/event-stream", resp.headers.get("Content-Type"))
            return resp.read().decode()

    @staticmethod
    def read_stream(blob):
        """(assembled answer, reasoning, the stats event, how many frames were junk)."""
        text, reasoning, stats, junk, event = "", "", None, 0, None
        for line in blob.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                frame = json.loads(data)
            except ValueError:
                junk += 1
                continue
            if event == "stats":
                stats, event = frame, None
                continue
            for choice in frame.get("choices") or []:
                delta = choice.get("delta") or {}
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
        return text, reasoning, stats, junk

    def status_of(self, model_id):
        """What the rail says about one model right now, or None."""
        _, payload = self.request("/api/instances")
        for row in payload["instances"]:
            if row["model"] == model_id:
                return row["status"]
        return None


class TestStatsDerivation(unittest.TestCase):
    """One derivation, three callers.

    The chat bar, a saved TiinyBench result and AINode Pocket's own chat bar all
    report the same request from the same two gateway blocks, so they read the
    same function. Two copies of this arithmetic would drift the first time the
    gateway renamed a field, and then the page and the saved result would
    disagree about a number somebody is about to quote.
    """

    def test_the_captured_pair_derives_the_benchmarks_numbers(self):
        self.assertEqual(bench.derive_stats(MEASURED_TIMINGS, MEASURED_USAGE), {
            "prompt_tokens": 19,          # usage wins over timings.prompt_n
            "out_tokens": 10,
            "prefill_tok_s": 146.97,
            "decode_tok_s": 22.76,
            "prefill_ms": 122.5,
            # Prefill plus one token of decode, in seconds. This is the only
            # answer available when the whole reply arrives at once.
            "ttft_s": 0.166,
            "cached_tokens": 1})

    def test_wall_time_is_added_only_when_the_caller_measured_one(self):
        self.assertNotIn("wall_s", bench.derive_stats(MEASURED_TIMINGS, MEASURED_USAGE))
        self.assertEqual(
            bench.derive_stats(MEASURED_TIMINGS, MEASURED_USAGE, 1.23456)["wall_s"],
            1.235)

    def test_empty_blocks_derive_zeroes_rather_than_raising(self):
        """A device that answered without a timings block is not a crash."""
        stats = bench.derive_stats(None, None)
        self.assertEqual(stats["out_tokens"], 0)
        self.assertEqual(stats["decode_tok_s"], 0)
        self.assertEqual(stats["cached_tokens"], 0)

    def test_the_chat_stats_object_carries_the_same_numbers_through(self):
        derived = bench.derive_stats(MEASURED_TIMINGS, MEASURED_USAGE)
        stats = bench.chat_stats(
            MEASURED_TIMINGS, MEASURED_USAGE, total_ms=812.4, ttft_ms=None,
            finish_reason="stop", device={"id": "TNY1", "name": "tiiny"},
            model="Qwen/Qwen3-8B")
        for field in ("prefill_ms", "decode_tok_s", "prefill_tok_s",
                      "prompt_tokens", "out_tokens", "cached_tokens"):
            self.assertEqual(stats[field], derived[field], field)
        # Nothing streamed, so there was no first token to time here and the
        # device's own answer is used, in milliseconds.
        self.assertEqual(stats["ttft_ms"], 166.0)
        self.assertEqual(stats["total_ms"], 812.4)
        self.assertEqual(stats["device"]["name"], "tiiny")
        for field in STAT_KEYS:
            self.assertIn(field, stats)

    def test_a_block_the_device_never_sent_reports_nothing_not_zero(self):
        """A stream that died before the last chunk has no numbers to show.

        The benchmark's derivation answers a missing block with zeroes, because
        a saved row wants a number in every column. On the chat bar a zero is a
        claim, and "out 0" beside a turn that really streamed tokens is exactly
        the kind of invented figure this page exists to not print. The two wall
        clock figures are this machine's own and survive.
        """
        stats = bench.chat_stats(
            None, None, total_ms=168.3, ttft_ms=812.0, finish_reason=None,
            device={"id": "TNY1", "name": "tiiny"}, model="Qwen/Qwen3-8B")
        for field in ("prefill_ms", "decode_tok_s", "prefill_tok_s",
                      "prompt_tokens", "out_tokens", "cached_tokens"):
            self.assertIsNone(stats[field], field)
        self.assertEqual(stats["ttft_ms"], 812.0)
        self.assertEqual(stats["total_ms"], 168.3)

    def test_nothing_streamed_and_nothing_measured_leaves_ttft_null_too(self):
        """Prefill plus one token is an answer only when the device sent one."""
        stats = bench.chat_stats(
            None, None, total_ms=20.0, ttft_ms=None, finish_reason=None,
            device={"id": "TNY1", "name": "tiiny"}, model="Qwen/Qwen3-8B")
        self.assertIsNone(stats["ttft_ms"])

    def test_usage_without_timings_keeps_the_counts_it_really_has(self):
        """Half a report is not an excuse to throw the half that arrived away."""
        stats = bench.chat_stats(
            None, MEASURED_USAGE, total_ms=20.0, ttft_ms=100.0, finish_reason="stop",
            device={"id": "TNY1", "name": "tiiny"}, model="Qwen/Qwen3-8B")
        self.assertEqual(stats["prompt_tokens"], 19)
        self.assertEqual(stats["out_tokens"], 10)
        self.assertEqual(stats["cached_tokens"], 1)
        self.assertIsNone(stats["decode_tok_s"])
        self.assertIsNone(stats["prefill_ms"])

    def test_the_suites_own_chat_reads_the_same_derivation(self):
        """A figure on the page and the same figure in a saved run agree."""
        derived = bench.derive_stats(MEASURED_TIMINGS, MEASURED_USAGE, 1.5)
        self.assertEqual(set(derived),
                         {"prompt_tokens", "out_tokens", "prefill_tok_s",
                          "decode_tok_s", "prefill_ms", "ttft_s", "cached_tokens",
                          "wall_s"})


class TestSentinel(unittest.TestCase):
    """Where the relay is allowed to think a stream has ended.

    The relay sees one line at a time and a JSON string cannot hold a raw
    newline, so every newline in these bytes is a frame boundary. That is what
    makes anchoring the sentinel to one enough.
    """

    def test_the_sentinel_inside_an_answer_is_just_text(self):
        frame = b'data: {"choices":[{"delta":{"content":"data: [DONE]"}}]}\n'
        self.assertEqual(serve.split_done(frame), (frame, None))

    def test_the_sentinel_on_its_own_line_still_ends_the_stream(self):
        head, tail = serve.split_done(b"data: [DONE]\n\n")
        self.assertEqual(head, b"")
        self.assertEqual(tail, b"data: [DONE]\n\n")

    def test_the_error_frame_and_its_sentinel_arrive_in_one_blob(self):
        blob = serve.sse_error("the device gave up")
        head, tail = serve.split_done(blob)
        self.assertIn(b"the device gave up", head)
        self.assertEqual(tail, b"data: [DONE]\n\n")

    def test_a_frame_quoting_the_sentinel_before_a_real_one_keeps_both(self):
        blob = (b'data: {"choices":[{"delta":{"content":"data: [DONE]"}}]}\n'
                b"data: [DONE]\n\n")
        head, tail = serve.split_done(blob)
        self.assertTrue(head.endswith(b'}}]}\n'))
        self.assertEqual(tail, b"data: [DONE]\n\n")


class TestChatRequest(unittest.TestCase):
    """What actually goes on the wire to the device."""

    def test_only_chat_template_kwargs_turns_thinking_on(self):
        """The gateway ignores every other knob. Measured, not assumed.

        Its own OpenAPI document declares a top level enable_thinking as well,
        with thinking_enabled, reasoning_effort and thinking_budget_tokens
        beside it, and the runtime honours none of them: a request sending
        enable_thinking false at the top level still got a chain of thought
        back. This is what Pocket sends, and it is what works.
        """
        on = serve.Handler.chat_request({
            "model": "m", "thinking": True,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(on["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("enable_thinking", on)
        off = serve.Handler.chat_request({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})

    def test_the_request_asks_for_usage_only_when_it_streams(self):
        streamed = serve.Handler.chat_request({
            "model": "m", "stream": True, "temperature": 0.4,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(streamed["stream_options"], {"include_usage": True})
        self.assertEqual(streamed["temperature"], 0.4)
        plain = serve.Handler.chat_request({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("stream_options", plain)
        self.assertNotIn("stream", plain)
        self.assertEqual(plain["max_tokens"], serve.DEFAULT_MAX_TOKENS)

    def test_a_system_prompt_is_sent_ahead_of_the_conversation(self):
        built = serve.Handler.chat_request({
            "model": "m", "system": "you are terse",
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(built["messages"][0],
                         {"role": "system", "content": "you are terse"})

    def test_a_system_message_already_there_is_not_doubled(self):
        built = serve.Handler.chat_request({
            "model": "m", "system": "second",
            "messages": [{"role": "system", "content": "first"},
                         {"role": "user", "content": "hi"}]})
        self.assertEqual(len(built["messages"]), 2)
        self.assertEqual(built["messages"][0]["content"], "first")

    def test_the_two_required_fields_are_named_when_they_are_missing(self):
        self.assertIn("model", serve.Handler.chat_request({"messages": [{}]}))
        self.assertIn("messages", serve.Handler.chat_request({"model": "m"}))

    def test_the_catalogue_link_is_derived_and_only_when_it_can_be(self):
        """No device field carries a URL.

        Neither the installed record nor the vendor catalogue has one: their
        only links are icon paths on the device itself. Model ids here are
        Hugging Face repository paths, so a link can be derived from one, and
        anything not shaped like one gets null rather than a guess at a URL.
        """
        self.assertEqual(serve.catalog_url("Qwen/Qwen3-8B"),
                         "https://huggingface.co/Qwen/Qwen3-8B")
        self.assertIsNone(serve.catalog_url("localmodel"))
        self.assertIsNone(serve.catalog_url("a/b/c"))
        self.assertIsNone(serve.catalog_url(""))

    def test_a_record_with_neither_input_nor_output_gets_null(self):
        """Null and an empty list are different claims, so only one is made."""
        self.assertIsNone(serve.io_capabilities(None, None))
        self.assertEqual(serve.io_capabilities("Text", "Vector"),
                         {"input": ["Text"], "output": ["Vector"]})
        self.assertEqual(serve.io_capabilities(["Text", "Image"], None),
                         {"input": ["Text", "Image"], "output": []})


class TestChatRoute(ServerCase):
    def test_a_streamed_chat_ends_with_a_stats_event(self):
        blob = self.raw_stream({"model": CHAT_MODEL, "stream": True, "max_tokens": 24,
                                "messages": [{"role": "user", "content": "hi"}]})
        text, _, stats, junk = self.read_stream(blob)
        self.assertEqual(junk, 0, "a frame was cut in half")
        self.assertTrue(text)
        self.assertIsNotNone(stats, "no event: stats arrived before [DONE]")
        for field in STAT_KEYS:
            self.assertIn(field, stats)
        self.assertEqual(stats["model"], CHAT_MODEL)
        self.assertEqual(stats["device"]["name"], self.fake.name)
        self.assertEqual(stats["device"]["id"], self.fake.serial)
        self.assertEqual(stats["out_tokens"], 24)
        self.assertEqual(stats["finish_reason"], "length")
        self.assertGreater(stats["decode_tok_s"], 0)
        self.assertGreater(stats["prefill_tok_s"], 0)
        self.assertIsNotNone(stats["ttft_ms"])
        self.assertGreaterEqual(stats["total_ms"], stats["ttft_ms"])

    def test_the_stats_event_comes_before_done_not_after(self):
        """Every SSE client stops reading at [DONE].

        Appending the numbers after that line would put them somewhere no
        ordinary reader ever looks, so they go in front of it.
        """
        blob = self.raw_stream({"model": CHAT_MODEL, "stream": True, "max_tokens": 8,
                                "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("event: stats", blob)
        self.assertLess(blob.index("event: stats"), blob.index("data: [DONE]"))
        self.assertTrue(blob.rstrip().endswith("data: [DONE]"))
        # Exactly one of each. The device ends its own stream with a [DONE] and
        # the relay appends one; relaying both would put the numbers in front of
        # a line that had already told the browser to stop reading.
        self.assertEqual(blob.count("event: stats"), 1)
        self.assertEqual(blob.count("data: [DONE]"), 1)

    def test_every_frame_gets_its_blank_line_back(self):
        """A server-sent event ends with a blank line.

        The device sends one data: line per frame and then the blank, and
        dropping the blank left one long frame that only a lenient parser could
        read. No two data: lines may be adjacent.
        """
        blob = self.raw_stream({"model": CHAT_MODEL, "stream": True, "max_tokens": 12,
                                "messages": [{"role": "user", "content": "hi"}]})
        lines = blob.split("\n")
        for i, line in enumerate(lines[:-1]):
            if line.startswith("data:"):
                self.assertEqual(lines[i + 1], "",
                                 "frame %d has no blank line after it" % i)

    def test_a_non_streamed_chat_carries_stats_beside_the_answer(self):
        status, payload = self.request("/api/chat", "POST", {
            "model": CHAT_MODEL, "max_tokens": 24,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertTrue(payload["choices"][0]["message"]["content"])
        stats = payload["stats"]
        for field in STAT_KEYS:
            self.assertIn(field, stats)
        self.assertEqual(stats["device"]["name"], self.fake.name)
        self.assertEqual(stats["out_tokens"], 24)
        self.assertEqual(stats["prompt_tokens"], payload["usage"]["prompt_tokens"])
        self.assertEqual(stats["decode_tok_s"],
                         bench.derive_stats(payload["timings"],
                                            payload["usage"])["decode_tok_s"])

    def test_the_thinking_toggle_is_what_asks_for_reasoning(self):
        """Off means none is requested, not that some is requested and hidden."""
        blob = self.raw_stream({"model": CHAT_MODEL, "stream": True, "max_tokens": 40,
                                "thinking": False,
                                "messages": [{"role": "user", "content": "hi"}]})
        _, reasoning, _, _ = self.read_stream(blob)
        self.assertEqual(reasoning, "")
        self.assertEqual(self.fake.state.chat_bodies[-1]["chat_template_kwargs"],
                         {"enable_thinking": False})

        blob = self.raw_stream({"model": CHAT_MODEL, "stream": True, "max_tokens": 40,
                                "thinking": True,
                                "messages": [{"role": "user", "content": "hi"}]})
        text, reasoning, stats, _ = self.read_stream(blob)
        self.assertTrue(reasoning)
        self.assertTrue(text)
        self.assertEqual(self.fake.state.chat_bodies[-1]["chat_template_kwargs"],
                         {"enable_thinking": True})
        # The chain of thought is spent out of the same budget the answer needs,
        # so it is counted in the tokens out.
        self.assertEqual(stats["out_tokens"], 40)

    def test_a_chat_without_a_model_is_a_400(self):
        status, payload = self.request("/api/chat", "POST", {
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        self.assertIn("model", payload["error"]["message"])

    def test_a_model_that_cannot_chat_is_refused_before_the_device_is_asked(self):
        status, payload = self.request("/api/chat", "POST", {
            "model": "Qwen/Qwen3-Embedding-0.6B", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        self.assertIn("cannot chat", payload["error"]["message"])
        self.assertEqual(self.fake.state.chat_bodies, [])

    def test_a_model_that_is_installed_but_not_loaded_is_a_503_that_says_so(self):
        status, payload = self.request("/api/chat", "POST", {
            "model": "zai-org/GLM-4.7-Flash", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 503)
        self.assertIn("not loaded", payload["error"]["message"])
        self.assertIn(CHAT_MODEL, payload["error"]["message"])

    def test_a_model_nobody_has_is_a_404(self):
        status, payload = self.request("/api/chat", "POST", {
            "model": "nobody/has-this",
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 404)
        self.assertIn("no model called", payload["error"]["message"])


class TestRelayEdges(ServerCase):
    """What the relay does with frames the fake device cannot produce.

    The fake answers like the hardware and it never fails, so these two live
    where the device call is replaced for the length of one request: a model
    that quotes the SSE sentinel in its own answer, and a stream that dies
    before the chunk carrying timings and usage. Both are about the same thing,
    which is what the page reports when something goes wrong.
    """

    def relay(self, lines, fail=None):
        """The raw body the browser receives for a device that sends `lines`."""
        real = bench.chat_lines
        self.addCleanup(setattr, bench, "chat_lines", real)

        def stub(tok, body, timeout=600):
            for line in lines:
                yield line
            if fail:
                raise bench.DeviceError(fail)

        bench.chat_lines = stub
        return self.raw_stream({"model": CHAT_MODEL, "stream": True,
                                "max_tokens": 50,
                                "messages": [{"role": "user", "content": "x"}]})

    @staticmethod
    def content(piece, finish=None):
        return "data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": piece},
                          "finish_reason": finish}]})

    def test_a_model_quoting_the_sentinel_is_relayed_whole(self):
        """"data: [DONE]" is what a model writes when you ask it about streaming.

        It arrives inside a frame's JSON, and cutting the relay at it broke
        three things in one go: the browser got a frame it could not parse, the
        answer stopped mid-sentence with nothing saying why, and the gateway's
        last chunk was never read, so the numbers came back as zeroes.
        """
        lines = [self.content("An SSE stream ends with "), "",
                 self.content("data: [DONE]"), "",
                 self.content(" and then closes.", finish="stop"), "",
                 "data: " + json.dumps({"choices": [], "timings": MEASURED_TIMINGS,
                                        "usage": MEASURED_USAGE}), "",
                 "data: [DONE]", ""]
        blob = self.relay(lines)
        text, _, stats, junk = self.read_stream(blob)
        self.assertEqual(text, "An SSE stream ends with data: [DONE] and then closes.")
        self.assertEqual(junk, 0, "a frame was cut in half")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["out_tokens"], 10)
        self.assertEqual(stats["prompt_tokens"], 19)
        self.assertEqual(stats["finish_reason"], "stop")
        self.assertGreater(stats["decode_tok_s"], 0)
        # Still exactly one real sentinel, in the order that puts the numbers
        # where a reader that stops at it will still see them.
        self.assertEqual(blob.count("data: [DONE]"), 2)  # one quoted, one real
        self.assertEqual(blob.count("event: stats"), 1)
        self.assertLess(blob.index("event: stats"), blob.rindex("data: [DONE]"))
        self.assertTrue(blob.rstrip().endswith("data: [DONE]"))

    def test_a_stream_that_failed_reports_nothing_rather_than_zeroes(self):
        """The 220 second cap is the common way for a stream to end badly.

        The tokens that already streamed are real, and the numbers for them
        never arrive. Saying "out 0" there would contradict the error sitting
        directly above it.
        """
        lines = [self.content("An SSE stream ends with "), "",
                 self.content("a sentinel."), ""]
        blob = self.relay(lines, fail="timed out after 220 s")
        text, _, stats, junk = self.read_stream(blob)
        self.assertEqual(junk, 0)
        self.assertEqual(text, "An SSE stream ends with a sentinel.")
        self.assertIn("timed out after 220 s", blob)
        self.assertIn("already streamed above are real", blob)
        self.assertIsNotNone(stats, "the stats event is still appended")
        for field in ("prefill_ms", "decode_tok_s", "prefill_tok_s",
                      "prompt_tokens", "out_tokens", "cached_tokens"):
            self.assertIsNone(stats[field], field)
        self.assertIsNone(stats["finish_reason"])
        # The two figures this machine measured itself are real and stay.
        self.assertIsNotNone(stats["ttft_ms"])
        self.assertGreater(stats["total_ms"], 0)
        self.assertEqual(stats["device"]["name"], self.fake.name)


class TestModelCard(ServerCase):
    def card(self, model_id):
        return self.request("/api/model_card?model="
                            + urllib.parse.quote(model_id, safe=""))

    def test_the_card_for_a_loaded_chat_model(self):
        status, card = self.card(CHAT_MODEL)
        self.assertEqual(status, 200)
        self.assertEqual(card["model"], CHAT_MODEL)
        self.assertEqual(card["name"], "Ornith-1.0-35B")
        self.assertEqual(card["type"], "Image-Text-to-Text")
        self.assertEqual(card["params"], "35B")
        self.assertEqual(card["size_bytes"], 18_000_000_000)
        self.assertEqual(card["npu_usage"], 50)
        self.assertIs(card["can_chat"], True)
        self.assertTrue(card["desc"])
        self.assertEqual(card["catalog_url"],
                         "https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B")
        # The device answers with one word a side, not a list, so the words are
        # wrapped rather than reshaped into something it never said.
        self.assertEqual(card["capabilities"]["output"], ["Text"])
        self.assertTrue(card["capabilities"]["input"])
        self.assertEqual(len(card["loaded_on"]), 1)
        self.assertEqual(card["loaded_on"][0]["device_name"], self.fake.name)
        self.assertEqual(card["loaded_on"][0]["status"], "running")

    def test_an_installed_model_that_is_not_loaded_says_so_with_an_empty_list(self):
        status, card = self.card("zai-org/GLM-4.7-Flash")
        self.assertEqual(status, 200)
        self.assertEqual(card["loaded_on"], [])
        self.assertIs(card["can_chat"], True)

    def test_a_model_that_cannot_chat_says_so(self):
        status, card = self.card("Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(status, 200)
        self.assertIs(card["can_chat"], False)
        self.assertEqual(card["capabilities"]["output"], ["Vector"])

    def test_a_model_this_box_does_not_have_is_a_404(self):
        status, payload = self.card("nobody/has-this")
        self.assertEqual(status, 404)
        self.assertIn("no model called", payload["error"]["message"])

    def test_the_model_is_required(self):
        status, payload = self.request("/api/model_card")
        self.assertEqual(status, 400)
        self.assertIn("model is required", payload["error"]["message"])


class TestInstances(ServerCase):
    def test_the_rail_lists_every_loaded_model_with_its_units_and_status(self):
        status, payload = self.request("/api/instances")
        self.assertEqual(status, 200)
        for row in payload["instances"]:
            for field in ("device_id", "device_name", "model", "npu_usage",
                          "status", "instance_id"):
                self.assertIn(field, row)
        models = {row["model"] for row in payload["instances"]}
        # A speech model loaded next to a chat model is the ordinary case. The
        # rail shows all three, because all three are spending NPU units.
        self.assertIn("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", models)
        self.assertIn("Qwen/Qwen3-Embedding-0.6B", models)
        self.assertIn(CHAT_MODEL, models)
        self.assertEqual({row["status"] for row in payload["instances"]}, {"running"})

    def test_the_one_device_reports_its_budget(self):
        _, payload = self.request("/api/instances")
        self.assertEqual(len(payload["devices"]), 1)
        device = payload["devices"][0]
        self.assertIs(device["reachable"], True)
        self.assertEqual(device["npu_total"], 100)
        self.assertEqual(device["npu_used"], 58)   # 50 + 7 + 1
        self.assertEqual(device["device_name"], self.fake.name)

    def test_a_device_that_does_not_answer_keeps_its_row(self):
        """The rail says unreachable rather than dropping the box off the page."""
        self.fake.stop()
        status, payload = self.request("/api/instances")
        self.assertEqual(status, 200)
        device = payload["devices"][0]
        self.assertIs(device["reachable"], False)
        self.assertIsNone(device["npu_total"])
        self.assertEqual(payload["instances"], [])


class TestLoadAndUnload(ServerCase):
    def test_a_load_is_accepted_then_comes_up_then_answers(self):
        """Running in npu/status is not proof on its own.

        A load is believable when the status says running and a chat actually
        comes back, which is why this does both.
        """
        model = "openai/gpt-oss-20b"      # 30 units, and 42 are free
        status, payload = self.request("/api/instances/load", "POST", {"model": model})
        self.assertEqual(status, 202)
        self.assertIs(payload["ok"], True)
        self.assertEqual(self.status_of(model), "loading")
        state = None
        for _ in range(8):
            state = self.status_of(model)
            if state == "running":
                break
        self.assertEqual(state, "running")
        status, reply = self.request("/api/chat", "POST", {
            "model": model, "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertTrue(reply["choices"][0]["message"]["content"])

    def test_a_model_still_coming_up_will_not_answer_yet(self):
        model = "openai/gpt-oss-20b"
        self.request("/api/instances/load", "POST", {"model": model})
        # Hold it in the loading state for the rest of this test. Every status
        # read advances a pending load by one, and a 35B takes tens of seconds
        # on real hardware rather than two polls.
        self.fake.state.pending[model]["polls"] = 500
        self.assertEqual(self.status_of(model), "loading")
        status, _ = self.request("/api/chat", "POST", {
            "model": model, "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 502, "the device refuses an instance still coming up")

    def test_the_device_takes_an_oversized_load_and_rolls_it_back_in_silence(self):
        """Measured on real hardware, and the reason the refusal above exists.

        A start that does not fit answers with the same 200 as one that does,
        the model shows as loading, and then it vanishes. Nothing is returned
        anywhere, which is why the app has to do the subtraction itself. This
        talks to the device directly, going round the refusal, to hold the
        behaviour the refusal is protecting against.
        """
        model = "Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo"   # 45 units, 42 free
        enc = urllib.parse.quote(model, safe="")
        accepted = bench.api(bench.gw("/api/v1/models/%s/start" % enc),
                             "test-key", body={}, timeout=10)
        self.assertNotIn("_error", accepted, "the device refused, which it does not do")
        self.assertEqual(self.status_of(model), "loading")
        for _ in range(6):
            if self.status_of(model) is None:
                break
        self.assertIsNone(self.status_of(model),
                          "the rolled back instance is still in the rail")

    def test_unload_takes_it_back_off(self):
        status, payload = self.request("/api/instances/unload", "POST",
                                       {"model": CHAT_MODEL})
        self.assertEqual(status, 202)
        self.assertIs(payload["ok"], True)
        _, rail = self.request("/api/instances")
        self.assertNotIn(CHAT_MODEL, [row["model"] for row in rail["instances"]])

    def test_unloading_something_that_is_not_loaded_is_a_400(self):
        status, payload = self.request("/api/instances/unload", "POST",
                                       {"model": "zai-org/GLM-4.7-Flash"})
        self.assertEqual(status, 400)
        self.assertIn("is not loaded on", payload["error"]["message"])

    def test_the_model_field_is_required(self):
        status, payload = self.request("/api/instances/load", "POST", {})
        self.assertEqual(status, 400)
        self.assertIn("model is required", payload["error"]["message"])

    # ---- the three refusals, word for word ------------------------------
    def test_a_model_that_is_not_installed_is_refused(self):
        status, payload = self.request("/api/instances/load", "POST",
                                       {"model": "nobody/has-this"})
        self.assertEqual(status, 400)
        self.assertEqual(
            payload["error"]["message"],
            "nobody/has-this is not installed on %s. Download it on the Models "
            "page first; nothing is fetched from the catalogue on demand."
            % self.fake.name)

    def test_a_model_that_cannot_chat_is_refused(self):
        status, payload = self.request("/api/instances/load", "POST",
                                       {"model": "Qwen/Qwen3-Embedding-0.6B"})
        self.assertEqual(status, 400)
        self.assertEqual(
            payload["error"]["message"],
            "Qwen/Qwen3-Embedding-0.6B is an embedding model and cannot chat, so "
            "loading it here would not give you anything to talk to.")

    def test_the_models_page_load_shares_the_same_guard(self):
        """Two doors onto the device's start, one set of reasons for refusing.

        The Models page loads speech and embedding models on purpose, so it does
        not ask for a chat model, but it gets the same not-installed and
        over-budget answers. Two guards would drift, and the one that drifted
        would be the one that lets a silent rollback through.
        """
        status, payload = self.request("/api/manage", "POST",
                                       {"action": "load", "model": "nobody/has-this"})
        self.assertEqual(status, 400)
        self.assertIn("is not installed on", payload["error"])
        status, payload = self.request(
            "/api/manage", "POST",
            {"action": "load", "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo"})
        self.assertEqual(status, 400)
        self.assertIn("rolls it back", payload["error"])
        status, _ = self.request("/api/manage", "POST",
                                 {"action": "load", "model": "zai-org/GLM-4.7-Flash"})
        self.assertEqual(status, 200, "a 12 unit chat model fits in the 42 free")

    def test_the_models_page_may_load_something_that_cannot_chat(self):
        """An embedding model is a fine thing to load from there."""
        self.fake.state.loaded.remove("Qwen/Qwen3-Embedding-0.6B")
        status, _ = self.request("/api/manage", "POST",
                                 {"action": "load",
                                  "model": "Qwen/Qwen3-Embedding-0.6B"})
        self.assertEqual(status, 200)

    def test_a_load_that_does_not_fit_the_budget_is_refused_here(self):
        """The device would take it and then roll it back without saying so.

        Refusing it here is the only place anybody is told.
        """
        status, payload = self.request(
            "/api/instances/load", "POST",
            {"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo"})
        self.assertEqual(status, 400)
        self.assertEqual(
            payload["error"]["message"],
            "Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo needs 45 NPU units and only "
            "42 of 100 are free on %s. The device accepts a load that does not "
            "fit and then rolls it back without saying so, so it is refused here "
            "instead." % self.fake.name)
        self.assertNotIn("Qwen/Qwen3-Coder-30B-A3B-Instruct-Turbo",
                         self.fake.state.loaded)


class TestChatPage(unittest.TestCase):
    """The page is served, and it asks for nothing off this machine.

    A farm app that pulled a script or a font from a CDN would stop working the
    moment the device is off the internet, which is most of the time.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "static", "app.html"), encoding="utf-8") as handle:
            cls.body = handle.read()

    def test_every_element_the_chat_page_needs_is_there(self):
        for element in ("p-chat", "model-card", "conversation-list", "new-chat",
                        "chat-model", "chat-temp", "chat-max-tokens", "chat-system",
                        "chat-stream", "chat-thinking", "chat-turns", "chat-form",
                        "chat-input", "chat-send", "instances-list",
                        "instances-devices", "instances-count", "load-device",
                        "load-model", "load-hint", "load-btn"):
            self.assertIn('id="%s"' % element, self.body, element)

    def test_nothing_is_fetched_from_anywhere_but_this_app(self):
        """The one link off the box is the vendor catalogue in the footer.

        The model card's catalogue link is not in this markup at all: it is
        built at runtime from the card's catalog_url, which is derived from the
        model id. Every other src and href is a path on this app.
        """
        allowed = {"https://github.com/webdevtodayjason/tiiny-sdk-docs"}
        found = re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', self.body)
        self.assertTrue(found, "the page has no assets at all, which is suspicious")
        for value in found:
            if value in allowed:
                continue
            self.assertFalse(value.lower().startswith(("http", "//")),
                             "%s points off this machine" % value)
        self.assertNotIn("cdn.", self.body)
        self.assertNotIn("fonts.googleapis", self.body)

    def test_the_page_reads_the_servers_own_field_names(self):
        """The bar is only honest if it reads the names the server sends."""
        for field in ("ttft_ms", "decode_tok_s", "total_ms", "prompt_tokens",
                      "out_tokens", "cached_tokens", "finish_reason", "npu_used",
                      "npu_total", "loaded_on", "can_chat", "catalog_url",
                      "reasoning_content"):
            self.assertIn(field, self.body, field)

    def test_the_whole_request_is_never_labelled_thinking_time(self):
        """The thinking clock stops at the first word of the answer.

        Only a stream sees that boundary. When the reply arrives in one piece
        nothing does, and stamping the wall clock there reported prefill plus
        reasoning plus the entire answer as "thinking", so the same model and
        the same question gave two different durations depending on whether
        Stream was ticked. There is no JS runner in this repo, so what is held
        here is the guard itself: every place the page stamps that clock is
        either the streamed delta handler or gated on the turn having streamed.
        """
        guards = re.findall(r"if \(([^()]*think_s === null[^()]*)\)", self.body)
        self.assertTrue(guards, "nothing stamps think_s any more; check this test")
        for guard in guards:
            if "live.streamed" in guard:
                continue
            self.assertIn("!live.msg.content", guard,
                          "a wall clock stamp that is neither streamed nor guarded")

    def test_a_thinking_time_too_short_to_print_is_not_printed(self):
        """Rounded to a tenth, a sub-50 ms pause reads "thinking, 0.0 s".

        That is a measurement of nothing dressed as a measurement, and the label
        already has an honest form for a duration nobody timed: the bare word.
        """
        self.assertIn("Number(msg.think_s) >= 0.05", self.body)

    def test_model_output_never_reaches_the_dom_unescaped(self):
        """Exactly one innerHTML in the chat renderer, fed by renderMarkdown.

        renderMarkdown escapes everything the model wrote before a single tag
        is assembled, so this is the one door and it is guarded.
        """
        self.assertIn("box.__body.innerHTML = renderMarkdown(msg.content);", self.body)
        self.assertIn("think.__body.textContent = msg.reasoning;", self.body)


if __name__ == "__main__":
    unittest.main()
