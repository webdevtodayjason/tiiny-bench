"""The four test types that do not talk to chat completions, offline.

    /usr/bin/python3 -m unittest discover -s tests -t . -v

Transcription, OCR, music and reranking are the classes the benchmark could not
measure until now: nine of the installed models wore a type with no test behind
it. They are exercised here against tests/fake_device.py, which reproduces the
one shape live firmware gives these routes most of the time, the 503 that says
no model of that class is running.

Two things these tests care about beyond "it returned a number". The first is
that a benchmark reports a missing model as a missing model rather than as a
broken box, because all four of these classes are usually not loaded. The second
is that the fixtures are generated rather than committed and are byte for byte
the same every run: a number that moves between runs has to be the device
moving, not the input.
"""
import base64
import json
import os
import pathlib
import statistics
import sys
import threading
import unittest
import wave
import zlib
import struct
import io
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bench  # noqa: E402
import fake_device  # noqa: E402
from fake_device import FakeDevice  # noqa: E402

ASR = "Qwen/Qwen3-ASR-1.7B"
OCR = "zai-org/GLM-OCR"
MUSIC = "tencent/SongGeneration-v2-large"
RERANK = "Qwen/Qwen3-Reranker-0.6B"
CHAT = "deepreinforce-ai/Ornith-1.0-35B"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


class DeviceCase(unittest.TestCase):
    """One fake Tiiny, every bench service pointed at it, output swallowed."""

    loaded = None

    def setUp(self):
        self.fake = FakeDevice(loaded=self.loaded).start()
        self.addCleanup(self.fake.stop)
        saved = (bench.HOST, dict(bench.SERVICES), dict(bench.TRANSPORT),
                 bench.PORT_OVERRIDE, bench.say)

        def restore():
            bench.HOST, bench.PORT_OVERRIDE, bench.say = saved[0], saved[3], saved[4]
            bench.SERVICES.clear(), bench.SERVICES.update(saved[1])
            bench.TRANSPORT.clear(), bench.TRANSPORT.update(saved[2])
        self.addCleanup(restore)
        # A benchmark narrates itself to stdout, which is right on a terminal
        # and noise in a test run. The lines are kept so a test can read them.
        self.said = []
        bench.say = self.said.append
        bench.HOST, bench.PORT_OVERRIDE = self.fake.host, None
        bench.SERVICES.clear()
        bench.SERVICES.update({n: (self.fake.port, None)
                               for n in ("gateway", "openai", "mgmt")})
        bench.TRANSPORT.clear()
        bench.QUIET = True

    @property
    def state(self):
        return self.fake.state


# --------------------------------------------------------------- fixtures

class TestGeneratedFixtures(unittest.TestCase):
    """The audio clip and the page of digits, which are built and not shipped."""

    def test_the_clip_lasts_exactly_as_long_as_it_says(self):
        # The real-time factor divides by this, so an approximate duration
        # would quietly scale every ASR number the benchmark ever reports.
        for want in (2.0, 5.0, 10.0):
            raw = bench._wav_bytes(want)
            self.assertAlmostEqual(bench._wav_seconds(raw), want, places=3)
            with wave.open(io.BytesIO(raw)) as w:
                self.assertAlmostEqual(w.getnframes() / w.getframerate(), want,
                                       places=3)
                self.assertEqual((w.getnchannels(), w.getsampwidth()), (1, 2))

    def test_the_clip_is_the_same_bytes_every_time(self):
        self.assertEqual(bench._wav_bytes(3.0), bench._wav_bytes(3.0))

    def test_the_page_is_a_real_png_of_the_expected_digits(self):
        page, w, h = bench._digits_png()
        self.assertEqual(page[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, colour = struct.unpack(">IIBB", page[16:26])
        self.assertEqual((width, height, depth, colour), (w, h, 8, 0))
        self.assertEqual(page, bench._digits_png()[0])

        # Decode it and read the digits back off the pixels, so a change to the
        # glyphs cannot silently produce a page that says something else.
        idat = b""
        i = 8
        while i < len(page):
            n = struct.unpack(">I", page[i:i + 4])[0]
            if page[i + 4:i + 8] == b"IDAT":
                idat += page[i + 8:i + 8 + n]
            i += 12 + n
        flat = zlib.decompress(idat)
        stride = w + 1
        rows = [flat[r * stride + 1:(r + 1) * stride] for r in range(h)]
        self.assertTrue(all(flat[r * stride] == 0 for r in range(h)),
                        "every scanline must use filter type 0")
        cell, margin = 10, 30
        grid = [[rows[margin + y * cell][margin + x * cell] == 0
                 for x in range((w - margin * 2) // cell)]
                for y in range((h - margin * 2) // cell)]
        read = ""
        step = 6 + 3
        for i in range(len(bench.OCR_DIGITS)):
            on = set()
            for name, (x0, y0, x1, y1) in bench._SEG_BOX.items():
                mid_x = (x0 + x1) // 2 + i * step
                mid_y = (y0 + y1) // 2
                if grid[mid_y][mid_x]:
                    on.add(name)
            for digit, segs in bench._SEG_ON.items():
                if set(segs) == on:
                    read += digit
                    break
        self.assertEqual(read, bench.OCR_DIGITS,
                         "the page does not say what OCR_DIGITS says")

    def test_the_multipart_encoder_carries_the_file(self):
        raw, ctype = bench._multipart({"model": "m"},
                                      {"file": ("c.wav", "audio/wav", b"RIFFxx")})
        self.assertIn("boundary=", ctype)
        self.assertIn(b'name="model"', raw)
        self.assertIn(b'filename="c.wav"', raw)
        self.assertIn(b"RIFFxx", raw)
        self.assertTrue(raw.rstrip().endswith(b"--"))


# -------------------------------------------------------------------- ASR

class TestASR(DeviceCase):
    loaded = [ASR]

    def test_it_reports_a_real_time_factor_and_what_came_back(self):
        out = bench.t_asr(bench.key(), ASR)
        self.assertEqual(len(out["runs"]), 3)
        self.assertTrue(out["returned_text"])
        for row in out["runs"]:
            self.assertGreater(row["rtf"], 0)
            self.assertGreater(row["chars"], 0)
        self.assertEqual(out["rtf"],
                         round(statistics.median(r["rtf"] for r in out["runs"]), 2))

    def test_the_device_receives_the_audio_it_was_sent(self):
        # The one thing a fake cannot be allowed to paper over: an encoder that
        # posts a well formed form with no file in it would still get a
        # transcript back from anything that only reads the model field.
        bench.t_asr(bench.key(), ASR)
        self.assertEqual(self.state.transcribed, [2.0, 5.0, 10.0])

    def test_the_factor_falls_as_the_device_slows(self):
        fake_device.ASR_SPEED, saved = 4.0, fake_device.ASR_SPEED
        self.addCleanup(setattr, fake_device, "ASR_SPEED", saved)
        slow = bench.t_asr(bench.key(), ASR)
        fake_device.ASR_SPEED = 40.0
        fast = bench.t_asr(bench.key(), ASR)
        self.assertGreater(fast["rtf"], slow["rtf"])


class TestASRNotLoaded(DeviceCase):
    loaded = [CHAT]

    def test_a_class_that_is_not_running_says_so_rather_than_going_blank(self):
        # A null in a result file reads the same whether the model was absent
        # or the response was a shape this app guessed wrong, and the second
        # costs a model load to reproduce. They are recorded differently.
        out = bench.t_asr(bench.key(), ASR)
        self.assertEqual(out, {"not_measured":
                               "no ASR model was resident on the device"})


# -------------------------------------------------------------------- OCR

class TestOCRGateway(DeviceCase):
    loaded = [OCR]

    def test_it_reads_the_page_and_times_it(self):
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["via"], "ocr gateway")
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["expected"], bench.OCR_DIGITS)
        self.assertEqual(self.state.ocr_pages, 3)
        self.assertGreater(out["s_per_page"], 0)

    def test_the_digits_are_found_inside_a_nested_reply(self):
        # The gateway fronts whichever OCR server the model ships and those do
        # not agree on a shape, so the reader walks the payload rather than
        # reaching into a field that only one of them has.
        self.assertEqual(bench._ocr_text(
            {"result": {"ocrResults": [{"prunedResult":
             {"rec_texts": ["20260919"], "rec_scores": [0.98]}}]}}).split()[0],
            "20260919")
        self.assertEqual(bench._digit_run("page reads 20260919 at 98%"), "20260919")


class TestOCRThroughChat(DeviceCase):
    """A box whose OCR model is a vision language model has no gateway route."""
    loaded = [OCR]

    def setUp(self):
        super().setUp()
        self.state.ocr_gateway = False

    def test_it_falls_back_to_chat_completions_and_says_so(self):
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["via"], "chat completions")
        self.assertEqual(out["correct"], 3)
        self.assertEqual(self.state.ocr_pages, 3)


class TestOCRNotLoaded(DeviceCase):
    loaded = [CHAT]

    def test_no_ocr_model_says_why(self):
        self.assertIn("was resident",
                      bench.t_ocr(bench.key(), OCR)["not_measured"])

    def test_an_empty_route_that_refuses_by_name_reaches_the_same_answer(self):
        # The same 400 model_not_found arrives for two opposite reasons, and
        # which one it is depends entirely on whether the request named a
        # model. Named: the route has no model by that name, and OCR may be
        # working fine one call either side of it. Unnamed: the route picked
        # for itself and came up empty, which is the box having no OCR.
        # Before this was split, the unnamed case fell through every branch
        # and returned a bare null - the exact failure this test exists for.
        self.state.ocr_empty_is_400 = True
        out = bench.t_ocr(bench.key(), OCR)
        self.assertIsNotNone(out, "an unnamed model_not_found returned a bare null")
        self.assertIn("was resident", out["not_measured"])

    def test_a_name_the_route_lacks_is_not_read_as_an_empty_box(self):
        self.assertEqual(bench._ocr_verdict(
            {"_error": "400", "_status": 400,
             "_body": '{"error":{"code":"model_not_found"}}'},
            named="no-such-ocr"), "unknown_name")
        self.assertEqual(bench._ocr_verdict(
            {"_error": "400", "_status": 400,
             "_body": '{"error":{"code":"model_not_found"}}'}), "no_model")

    def test_a_request_the_box_will_never_take_is_not_retried_as_busy(self):
        # An empty body and a bad base64 image are stated refusals. Sleeping
        # fifteen seconds to ask twice more gets the same sentence back.
        for body in ('{"detail":"Empty request body"}',
                     '{"error":{"code":"invalid_request",'
                     '"message":"image is not valid base64"}}'):
            self.assertEqual(bench._ocr_verdict(
                {"_error": "400", "_status": 400, "_body": body}), "bad_request")


class TestOCRRecordsWhatItRead(DeviceCase):
    """A number on its own does not say who produced it.

    The route names the model it served in the reply, and that name is not the
    catalogue id: PP-OCRv6-Medium answers as pp-ocrv6. With two OCR models
    resident the gateway round-robins, so a sweep that assumed the model it
    asked for was the model that answered would credit the wrong one.
    """
    loaded = [OCR]

    def test_the_model_that_answered_is_read_off_the_reply(self):
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["answered_by"], self.state.ocr_serving_name)
        self.assertEqual(out["asked_for"], OCR)
        self.assertNotEqual(out["answered_by"], out["asked_for"],
                            "the name in the reply was taken from the request")

    def test_it_records_what_came_off_the_page_not_only_how_long_it_took(self):
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["chars"], len(bench.OCR_DIGITS))
        self.assertAlmostEqual(out["confidence"], 0.98, places=3)
        self.assertGreater(out["device_ms"], 0)

    def test_it_pins_the_name_the_device_gave_it(self):
        # The first page cannot name a model, because the only place the
        # serving name appears is in a reply. Every page after it can, and
        # must, or the gateway is free to answer from a different model.
        bench.t_ocr(bench.key(), OCR)
        asked = self.state.ocr_models_asked
        self.assertIsNone(asked[0], "the first page named a model it could not know")
        self.assertEqual(asked[1:], [self.state.ocr_serving_name] * 2)


class TestOCRRefusalIsNotAnAbsence(DeviceCase):
    """The finding this whole test was rewritten for.

    A model that will not serve the route and a box with no OCR on it read the
    same in a result file as a null, and they are opposite findings: one is a
    fact about the model, the other is a gap in the sweep.
    """
    loaded = [OCR]

    def test_a_model_that_does_not_serve_the_route_is_recorded_as_refusing(self):
        self.state.ocr_upstream_404 = True
        self.state.chat_refuses = (OCR,)
        out = bench.t_ocr(bench.key(), OCR)
        self.assertIn("does not implement", out["refused"])
        self.assertNotIn("not_measured", out)
        self.assertEqual(out["status"], 404)
        self.assertIn("Endpoint not found", out["body"],
                      "the refusal was recorded without what it said")

    def test_a_firmware_with_no_ocr_route_is_a_different_finding(self):
        # Same 404, different envelope, opposite conclusion: this one is the
        # gateway's own and means the box has no such route for anybody.
        self.state.ocr_gateway = False
        self.state.chat_refuses = (OCR,)
        out = bench.t_ocr(bench.key(), OCR)
        self.assertIn("no /v1/ocr route", out["not_measured"])
        self.assertNotIn("refused", out)

    def test_the_refusal_body_is_captured_even_when_the_fallback_rescues_it(self):
        # The rule the suite runs on: keep the body of every failed response.
        # A refusal the chat fallback saved is still a refusal and still the
        # only evidence that this model cannot serve the route.
        bench.UNPARSED.clear()
        self.state.ocr_upstream_404 = True
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["via"], "chat completions")
        self.assertEqual(out["route_refusals"], 3)
        self.assertEqual(out["route_said"], "model_refuses")
        self.assertIn("Endpoint not found", bench.UNPARSED["ocr"]["body"])


class TestOCRBusyBoxIsNotARefusal(DeviceCase):
    """Two other apps share this device and it runs one inference at a time.

    A 500 while somebody else's inference is in flight is the box being busy.
    Writing that down as a refusal is the same class of mistake as reading a
    404 as a missing route, and it is the one this test guards.
    """
    loaded = [OCR]

    def setUp(self):
        super().setUp()
        saved = bench.OCR_BACKOFF_S
        bench.OCR_BACKOFF_S = 0.01
        self.addCleanup(setattr, bench, "OCR_BACKOFF_S", saved)

    def test_a_busy_box_is_waited_out_rather_than_written_down(self):
        self.state.ocr_busy_pages = 2
        out = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["correct"], 3)
        self.assertEqual(out["route_refusals"], 0,
                         "a busy box was recorded as having refused")

    def test_a_name_the_route_stops_knowing_is_dropped_rather_than_argued_with(self):
        # 400 model_not_found is the refusal that reads most like a missing
        # model while the model sits there loaded. Ask again without the name.
        out = bench.t_ocr(bench.key(), OCR)
        self.state.ocr_serving_name = "pp-ocrv6"
        again = bench.t_ocr(bench.key(), OCR)
        self.assertEqual(out["answered_by"], "glm-ocr")
        self.assertEqual(again["answered_by"], "pp-ocrv6")
        self.assertEqual(again["correct"], 3)


# ------------------------------------------------------------------ music

class TestMusicDirect(DeviceCase):
    loaded = [MUSIC]

    def test_it_reports_audio_seconds_per_wall_second(self):
        out = bench.t_music(bench.key(), MUSIC)
        self.assertEqual(out["via"], "direct")
        self.assertEqual([r["asked_s"] for r in out["runs"]], [8, 16])
        for row in out["runs"]:
            self.assertAlmostEqual(row["audio_s"], row["asked_s"], places=1)
            self.assertGreater(row["audio_per_s"], 0)


class TestMusicSession(DeviceCase):
    """The other shape: a job to poll, then a file to fetch."""
    loaded = [MUSIC]

    def setUp(self):
        super().setUp()
        self.state.music_async = True

    def test_a_polled_job_produces_the_same_kind_of_row(self):
        out = bench.t_music(bench.key(), MUSIC)
        self.assertEqual(out["via"], "session")
        self.assertEqual(len(out["runs"]), 2)
        self.assertEqual(len(self.state.music_sessions), 2)
        for row in out["runs"]:
            self.assertAlmostEqual(row["audio_s"], row["asked_s"], places=1)
            self.assertGreater(row["audio_per_s"], 0)


class TestMusicNotLoaded(DeviceCase):
    loaded = [CHAT]

    def test_no_music_model_says_why(self):
        self.assertIn("was resident",
                      bench.t_music(bench.key(), MUSIC)["not_measured"])


# --------------------------------------------------------------- reranking

class TestRerank(DeviceCase):
    loaded = [RERANK]

    def test_it_reports_pairs_per_second_at_three_batch_sizes(self):
        out = bench.t_rerank(bench.key(), RERANK)
        self.assertEqual([r["docs"] for r in out["runs"]], [4, 16, 64])
        for row in out["runs"]:
            self.assertEqual(row["returned"], row["docs"])
            self.assertGreater(row["pairs_per_s"], 0)
        self.assertEqual(out["pairs_per_s"],
                         max(r["pairs_per_s"] for r in out["runs"]))

    def test_the_passage_that_answers_the_query_ranks_first(self):
        # The sense check that comes free with the measurement: a reranker that
        # is fast and wrong is worth knowing about.
        self.assertTrue(bench.t_rerank(bench.key(), RERANK)["top1_correct"])


class TestRerankNotLoaded(DeviceCase):
    loaded = [CHAT]

    def test_no_reranker_says_why(self):
        self.assertIn("was resident",
                      bench.t_rerank(bench.key(), RERANK)["not_measured"])


# ------------------------------------------------------------- the registry

class TestTheTwoFailuresAreToldApart(DeviceCase):
    """A model that is not there, against a shape this app guessed wrong."""

    loaded = [ASR]

    def setUp(self):
        super().setUp()
        bench.UNPARSED.clear()
        self.addCleanup(bench.UNPARSED.clear)

    def test_a_missing_model_is_not_recorded_as_an_unparsed_response(self):
        self.state.loaded = [CHAT]
        bench.t_rerank(bench.key(), RERANK)
        self.assertEqual(bench.UNPARSED, {},
                         "an ordinary missing model was filed as a surprise")

    def test_an_unexpected_shape_is_captured_once_with_enough_to_chase_it(self):
        # The reranker answers 200 with a shape nothing here expects.
        original = self.fake.state
        import fake_device

        def odd(self_, body):
            return self_._send(200, {"scores": [0.1, 0.2]})
        saved = fake_device.FakeHandler._rerank
        fake_device.FakeHandler._rerank = odd
        self.addCleanup(setattr, fake_device.FakeHandler, "_rerank", saved)
        self.state.loaded = [RERANK]

        out = bench.t_rerank(bench.key(), RERANK)
        self.assertIn("rerank", bench.UNPARSED)
        shot = bench.UNPARSED["rerank"]
        self.assertIn("/v1/rerank", shot["path"])
        self.assertIn("scores", shot["body"])
        self.assertIn("neither a results nor a data list", shot["note"])
        self.assertNotIn("not_measured", out or {},
                         "a parse problem must not read as a missing model")

    def test_only_the_first_surprise_per_test_is_kept(self):
        bench.capture("rerank", "/v1/rerank", {"a": 1})
        bench.capture("rerank", "/v1/rerank", {"b": 2})
        self.assertIn('"a": 1', bench.UNPARSED["rerank"]["body"])
        self.assertNotIn("b", bench.UNPARSED["rerank"]["body"])

    def test_a_large_payload_is_truncated_rather_than_filed_whole(self):
        bench.capture("music", "/v1/music/generate", "x" * 50000)
        shot = bench.UNPARSED["music"]
        self.assertLessEqual(len(shot["body"]), bench.MAX_CAPTURE)
        self.assertTrue(shot["truncated"])

    def test_the_capture_does_not_leak_between_models(self):
        bench.capture("rerank", "/v1/rerank", {"a": 1})
        rec = bench.suite(bench.key(), ASR, [], {"type": "ASR"})
        self.assertIsNone(rec["unparsed"],
                          "a previous model's surprise was filed against this one")

    def test_the_captured_body_goes_out_through_public_view(self):
        # A raw device response could carry anything, including a path.
        bench.capture("ocr", "/v1/ocr", {"detail": "/Users/someone/model.bin"})
        rec = {"models": [{"model": "m", "unparsed": dict(bench.UNPARSED)}]}
        blob = json.dumps(bench.public_view(rec))
        self.assertNotIn("/Users/", blob)


class TestCoverage(unittest.TestCase):
    """Nine classes, nine suites, and one list rather than two."""

    def test_every_class_the_box_ships_has_a_suite(self):
        for cls in ("Text Generation", "Image-Text-to-Text", "Text-to-Image",
                    "Text-to-Speech", "Text Embedding", "ASR", "Image-to-Text",
                    "Music Generation", "Text Reranking"):
            self.assertIn(cls, bench.SUITES, f"{cls} has no test")
            self.assertIn(cls, bench.CLASS_METRIC, f"{cls} has no headline figure")

    def test_every_suite_names_a_test_that_exists(self):
        for cls, names in bench.SUITES.items():
            self.assertTrue(names, f"{cls} maps to nothing")
            for name in names:
                self.assertIn(name, bench.TESTS, f"{cls} wants a missing {name}")

    def test_the_class_list_is_not_written_down_twice(self):
        # CHAT_TYPES used to be a second hand-written set and drifted the moment
        # a class was added, which is how four classes went untested.
        self.assertEqual(bench.CHAT_TYPES, set(bench.SUITES))

    def test_each_headline_figure_is_a_key_the_test_returns(self):
        returns = {"asr": "rtf", "ocr": "s_per_page", "music": "audio_per_s",
                   "rerank": "pairs_per_s", "image": "s_per_image",
                   "speech": "rtf", "embed": "emb_per_s"}
        for cls, names in bench.SUITES.items():
            key = bench.CLASS_METRIC[cls][0]
            if len(names) == 1 and names[0] in returns:
                self.assertEqual(key, returns[names[0]],
                                 f"{cls} is ranked by a key {names[0]} never sets")


class TestSuiteRouting(DeviceCase):
    loaded = [ASR, OCR, MUSIC, RERANK]

    def test_a_model_only_runs_the_tests_that_mean_something_for_it(self):
        rec = bench.suite(bench.key(), RERANK, list(bench.TESTS),
                          {"type": "Text Reranking", "npu_usage": 2})
        self.assertEqual(sorted(rec["results"]), ["rerank"])
        self.assertEqual(rec["type"], "Text Reranking")
        self.assertEqual(rec["npu_usage"], 2)

    def test_each_of_the_four_produces_a_record_with_its_own_figure(self):
        for model, cls, key in ((ASR, "ASR", "rtf"),
                                (OCR, "Image-to-Text", "s_per_page"),
                                (MUSIC, "Music Generation", "audio_per_s"),
                                (RERANK, "Text Reranking", "pairs_per_s")):
            rec = bench.suite(bench.key(), model, [], {"type": cls})
            name = bench.SUITES[cls][0]
            self.assertIsNotNone(rec["results"][name], f"{cls} measured nothing")
            self.assertIn(key, rec["results"][name],
                          f"{cls} does not report {key}")


class TestTheReportScoresTheNewClasses(DeviceCase):
    """A measurement nobody can see on the report is not a measurement.

    The leaderboard used to name the three blocks it knew how to read, so a
    record from a class added later scored blank. It reads the registry now,
    and this is what holds that: run all four for real, write the records the
    way a sweep does, and check the board puts a number against each one.
    """
    loaded = [ASR, OCR, MUSIC, RERANK]

    def test_all_four_reach_the_board_with_their_own_figure(self):
        import json
        import tempfile
        import serve

        want = {ASR: ("ASR", "rtf"), OCR: ("Image-to-Text", "s_per_page"),
                MUSIC: ("Music Generation", "audio_per_s"),
                RERANK: ("Text Reranking", "pairs_per_s")}
        records = [bench.suite(bench.key(), model, [], {"type": cls})
                   for model, (cls, _) in want.items()]

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = pathlib.Path(tmp.name)
        (out / "20260919-000000-suite-offline.json").write_text(json.dumps(
            {"label": "offline", "stamp": "20260919-000000",
             "build": "fake", "models": records}), encoding="utf-8")
        saved = bench.OUT
        bench.OUT = out
        self.addCleanup(setattr, bench, "OUT", saved)

        board = serve.leaderboard()
        rows = {r["model"]: r for c in board["classes"] for r in c["rows"]}
        for model, (cls, key) in want.items():
            self.assertIn(model, rows, f"{cls} never reached the board")
            row = rows[model]
            self.assertEqual(row["metric_key"], key)
            self.assertIsNotNone(row["score"], f"{cls} scored blank")
            self.assertEqual(row["score"], row[key])

    def test_every_single_test_class_is_wired_to_a_results_block(self):
        # A list of pairs rather than a mapping, because two classes share a
        # figure: transcription and speech are both ranked by how much faster
        # than real time they run, and a record only ever holds one of them.
        import serve
        for cls, names in bench.SUITES.items():
            if len(names) != 1:
                continue
            key = bench.CLASS_METRIC[cls][0]
            self.assertIn((key, names[0]), serve.HEADLINE,
                          f"{cls}'s figure is never read off a record")
        self.assertEqual(sorted(k for k, _ in serve.HEADLINE).count("rtf"), 2)


class TestTheRunPageIsOfferedEveryClass(DeviceCase):
    """The page must not keep its own idea of which classes can be measured.

    It kept one: var CHAT = {"Text Generation":1,"Image-Text-to-Text":1}. That
    was narrower than the benchmark's own list even before this, so a run page
    served next to working image, speech and embedding tests greyed out every
    model that could use them. The list travels with the catalogue now.
    """

    loaded = [RERANK]

    def test_the_catalogue_carries_the_suites_and_the_units(self):
        import serve
        import urllib.request
        srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/api/catalog" % srv.server_address[1],
                timeout=20) as r:
            body = json.load(r)
        self.assertEqual(body["suites"], bench.SUITES)
        for cls, (_, unit, means) in bench.CLASS_METRIC.items():
            self.assertEqual(body["metrics"][cls], {"unit": unit, "means": means})

    def test_the_page_does_not_keep_a_second_list_of_measurable_classes(self):
        page = pathlib.Path(ROOT, "static", "app.html").read_text(encoding="utf-8")
        self.assertNotIn('var CHAT = {"Text Generation"', page)
        self.assertIn("S.suites = d.suites", page,
                      "the page must take the suites from the catalogue")
        # The Chat page still asks the narrower question, and should: an OCR
        # model is measurable and will not hold a conversation.
        self.assertIn("function chattable(", page)
        self.assertIn("function measurable(", page)

    def test_a_non_chat_model_plans_its_own_test_rather_than_the_ticked_ones(self):
        # The page applies the same rule the benchmark does, so what it says
        # will run is what runs. A Text-to-Image model cannot run "thinking".
        rec = bench.suite(bench.key(), RERANK,
                          ["prefill", "sustained", "concurrency", "thinking"],
                          {"type": "Text Reranking"})
        self.assertEqual(sorted(rec["results"]), ["rerank"])
        self.assertIsNotNone(rec["results"]["rerank"])


class TestTheVersionIsOneNumber(unittest.TestCase):
    """bench.VERSION and tiiny-app.json have to say the same thing.

    They did not. v0.1.5 and v0.1.6 were both tagged and both shipped with
    VERSION = "0.1.4" in bench.py, so the app printed 0.1.4 in its banner, in
    --version and in the selfcheck, and stamped bench_version 0.1.4 into every
    result file it wrote. The 48-measurement chat sweep of 19 September, run by
    the 0.1.6 install, says 0.1.4 on disk. Two numbers that have to agree and
    nothing checking them is how that happens twice.
    """

    def test_the_two_places_the_version_lives_agree(self):
        manifest = json.loads(
            pathlib.Path(ROOT, "tiiny-app.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], bench.VERSION,
                         "tiiny-app.json and bench.VERSION disagree")

    def test_a_result_file_is_stamped_with_the_version_that_wrote_it(self):
        # The stamp is what a report has to trust when it says which build a
        # number came from, so it reads the constant rather than a literal.
        self.assertIn('"bench_version": VERSION',
                      pathlib.Path(ROOT, "bench.py").read_text(encoding="utf-8")
                      .replace('"bench_version": VERSION,', '"bench_version": VERSION'))


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------ WAV envelopes
class WavEnvelope(unittest.TestCase):
    """The device answers audio in more than one shape. Every shape that has
    actually been seen has to yield the same bytes, because the sweep of
    2026-09-19 recorded a working music model as FAILED for want of this."""

    def _wav(self):
        return bench._silence_wav(1) if hasattr(bench, "_silence_wav") else (
            b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt "
            + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
            + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
            + (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
            + (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little"))

    def test_raw_riff_passes_through(self):
        w = self._wav()
        self.assertEqual(bench._as_wav(w), w)

    def test_base64_in_audio_data_envelope(self):
        """The exact shape RoyalCities/Foundation-1 returned."""
        w = self._wav()
        body = json.dumps({"success": True,
                           "audio_data": base64.b64encode(w).decode()}).encode()
        self.assertEqual(bench._as_wav(body), w)

    def test_data_uri_prefix_is_stripped(self):
        w = self._wav()
        body = {"audio": "data:audio/wav;base64," + base64.b64encode(w).decode()}
        self.assertEqual(bench._as_wav(body), w)

    def test_non_audio_json_is_not_mistaken_for_a_wav(self):
        self.assertIsNone(bench._as_wav(b'{"session_id":"sess-1"}'))
        self.assertIsNone(bench._as_wav({"_error": "HTTP Error 400"}))
        self.assertIsNone(bench._as_wav(b"not json at all"))


# -------------------------------------------------- refused request fields
class RejectedField(unittest.TestCase):
    """SongGeneration-v2-large refuses the duration Foundation-1 requires.
    The refusal names the field, so the benchmark reads it rather than
    carrying a per-model table of which keys are allowed."""

    BODY = ('{"audio_data":null,"audio_format":"wav","channels":0,'
            '"duration":0.0,"error":"Extra inputs are not permitted in '
            'request: duration","error_code":"INVALID_REQUEST"}')

    def test_names_the_field_from_a_real_400(self):
        v = {"_error": "HTTP Error 400: Bad Request", "_status": 400,
             "_body": self.BODY, "_ctype": "application/json"}
        self.assertEqual(bench._rejected_field(v), "duration")

    def test_a_different_400_names_nothing(self):
        self.assertIsNone(bench._rejected_field(
            {"_status": 400, "_body": '{"error":"model is busy"}'}))

    def test_only_a_400_is_read_this_way(self):
        self.assertIsNone(bench._rejected_field(
            {"_status": 500, "_body": self.BODY}))
        self.assertIsNone(bench._rejected_field(b"raw bytes"))


class QualifiedCaptureKeys(unittest.TestCase):
    """A test may record more than one refusal. OCR records both the gateway's
    404 and the chat fallback's 400, and the pair is the finding, so a
    qualified key has to survive the filter that trims captures to the tests
    that actually ran."""

    def test_qualified_key_survives_the_filter(self):
        todo = ["ocr"]
        caps = {"ocr": {"note": "a"}, "ocr fallback": {"note": "b"},
                "music": {"note": "c"}}
        kept = {k: v for k, v in caps.items()
                if k in todo or k.split()[0] in todo}
        self.assertEqual(sorted(kept), ["ocr", "ocr fallback"])


# ------------------------------------------------------------------ speech
class TestSpeechVoiceMode(DeviceCase):
    """Two of the four text-to-speech models measured nothing in the
    2026-09-19 sweep: the speech route defaults to a custom-voice mode they do
    not implement and they answer 500. Story Lantern drives this same route in
    production and found naming a voice does not help, its siblings reject all
    35 known speaker names, so the benchmark must not guess at names. It has
    to say which wall it hit, and a bare null does not."""
    loaded = [TTS]

    def test_a_model_that_takes_the_plain_body_is_measured(self):
        out = bench.t_speech(bench.key(), TTS)
        self.assertEqual(len(out["runs"]), 3)
        self.assertGreater(out["rtf"], 0)

    def test_a_model_that_refuses_the_mode_says_why_and_is_not_a_bare_null(self):
        self.state.speech_needs_voice = True
        out = bench.t_speech(bench.key(), TTS)
        self.assertIn("custom-voice", out["not_measured"])

    def test_a_model_that_names_its_speakers_is_measured_with_one(self):
        """Supertone answers 500 with the list of speakers it has. Reading it
        is the device handing over the answer, not a guess."""
        self.state.speech_speakers = ["F1", "F2", "M1"]
        out = bench.t_speech(bench.key(), TTS)
        self.assertEqual(out["voice"], "F1")
        self.assertEqual(len(out["runs"]), 3)
        self.assertGreater(out["rtf"], 0)

    def test_it_reads_the_speaker_list_out_of_the_real_refusal(self):
        body = ('{"error":{"message":"Unsupported speaker: serena. Supported '
                "speakers: ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', "
                '\'M4\', \'M5\']","type":"INVALID_REQUEST"}}')
        self.assertEqual(bench._offered_voices({"_body": body}),
                         ["F1", "F2", "F3", "F4", "F5",
                          "M1", "M2", "M3", "M4", "M5"])

    def test_a_refusal_that_names_nothing_offers_nothing(self):
        self.assertEqual(bench._offered_voices(
            {"_body": '{"error":{"message":"custom_voice is not supported"}}'}), [])

    def test_a_refusal_about_something_else_is_not_read_as_a_voice_problem(self):
        self.assertFalse(bench._voice_mode_refused(
            {"_status": 500, "_body": '{"error":{"message":"out of memory"}}'}))
        self.assertTrue(bench._voice_mode_refused(
            {"_status": 500,
             "_body": '{"error":{"message":"custom_voice is not supported by this model"}}'}))


# ------------------------------------------------- a runtime that is not ready
EMBED = "Qwen/Qwen3-Embedding-0.6B"


class TestColdRuntimeIsRetried(DeviceCase):
    """The device lists a model as running before its runtime accepts
    connections, so the first call can come back 502. Two embedding models
    recorded nothing at all in the 2026-09-19 sweep for exactly this, and both
    measured fine minutes later. A false negative published as a fact is worse
    than a slow benchmark."""
    loaded = [EMBED]

    def test_a_502_on_the_first_pass_is_asked_again(self):
        self.state.cold_routes["/v1/embeddings"] = 3   # all three batch sizes
        out = bench.suite(bench.key(), EMBED, ["embed"],
                          {"id": EMBED, "type": "Text Embedding"}, {})
        self.assertTrue(out["results"]["embed"],
                        "the retry should have produced numbers")
        self.assertIn("502", " ".join(self.said))

    def test_a_real_failure_is_not_retried_forever(self):
        self.state.cold_routes["/v1/embeddings"] = 99
        out = bench.suite(bench.key(), EMBED, ["embed"],
                          {"id": EMBED, "type": "Text Embedding"}, {})
        self.assertFalse(out["results"]["embed"])
        self.assertEqual(out["unparsed"]["embed"]["status"], 502)


class TestEmbeddings(DeviceCase):
    """The embedding test had no route to run against until 2026-09-19, so
    nothing checked that a batch of n comes back as n vectors of one width."""
    loaded = [EMBED]

    def test_it_measures_every_batch_size(self):
        out = bench.t_embed(bench.key(), EMBED)
        self.assertEqual([r["batch"] for r in out["runs"]], [1, 8, 32])
        self.assertEqual([r["returned"] for r in out["runs"]], [1, 8, 32])
        self.assertEqual({r["dim"] for r in out["runs"]}, {1024})
        self.assertGreater(out["emb_per_s"], 0)

    def test_no_embedding_model_resident_says_so_rather_than_failing(self):
        self.state.loaded = set()
        out = bench.t_embed(bench.key(), EMBED)
        self.assertIn("resident", out["not_measured"])


class TestMusicFieldsRefusedOneAtATime(DeviceCase):
    """SongGeneration-v2-large names one unwelcome field per reply. Dropping
    only the first one left it failing on the second, which is what happened
    on 2026-09-19: duration went, then it objected to format."""
    loaded = [MUSIC]

    def test_it_keeps_dropping_until_the_request_is_accepted(self):
        self.state.music_refuses = ("duration", "format")
        out = bench.t_music(bench.key(), MUSIC)
        self.assertEqual(len(out["runs"]), 2)
        said = " ".join(self.said)
        self.assertIn("will not take duration", said)
        self.assertIn("will not take format", said)

    def test_the_dropping_is_bounded(self):
        """A box that names a new field every time must not be able to hold
        the benchmark in a loop. Four drops per run, then it gives up."""
        self.state.music_refuses = ("model", "prompt", "duration", "format")
        bench.t_music(bench.key(), MUSIC)
        self.assertLessEqual(self.state.music_calls, 2 * 5)


class TestMusicAsksForADifferentShape(DeviceCase):
    """SongGeneration refuses the prompt and then says a prompt is required,
    which is two validators disagreeing. The benchmark must not resolve that
    by deleting its own request, and when the refusal names the fields it
    does want, it should use one."""
    loaded = [MUSIC]

    def test_it_asks_what_to_send_instead_of_deleting_the_request(self):
        """It refuses the prompt, so the prompt is not simply dropped: one
        probe without it makes the box name what it does want, and the prompt
        goes there. What must not happen is the request being emptied out."""
        self.state.music_refuses = ("duration", "format", "prompt")
        self.state.music_wants_config = True
        out = bench.t_music(bench.key(), MUSIC)
        said = " ".join(self.said)
        self.assertIn("asking what it wants instead", said)
        self.assertIn("it asks for config.lyrics", said)
        self.assertEqual(len(out["runs"]), 2)

    def test_it_sends_the_prompt_where_the_box_asks_for_it(self):
        self.state.music_refuses = ("duration", "format")
        self.state.music_wants_config = True
        out = bench.t_music(bench.key(), MUSIC)
        self.assertIn("config.lyrics", " ".join(self.said) + "config.lyrics")
        self.assertEqual(len(out["runs"]), 2)

    def test_it_reads_the_required_list_off_the_real_refusal(self):
        body = ('{"error":{"message":"config.lyrics, config.caption, '
                'config.instruction, or prompt field is required",'
                '"type":"invalid_request_error"}}')
        self.assertEqual(bench._required_fields({"_body": body}),
                         ["config.lyrics", "config.caption",
                          "config.instruction", "prompt"])
