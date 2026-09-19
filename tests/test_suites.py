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

    def test_a_class_that_is_not_running_is_reported_not_crashed(self):
        self.assertIsNone(bench.t_asr(bench.key(), ASR))


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

    def test_no_ocr_model_means_no_row(self):
        self.assertIsNone(bench.t_ocr(bench.key(), OCR))


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

    def test_no_music_model_means_no_row(self):
        self.assertIsNone(bench.t_music(bench.key(), MUSIC))


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

    def test_no_reranker_means_no_row(self):
        self.assertIsNone(bench.t_rerank(bench.key(), RERANK))


# ------------------------------------------------------------- the registry

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


if __name__ == "__main__":
    unittest.main()
