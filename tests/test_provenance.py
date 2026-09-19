"""The provenance envelope, and the one function that makes a record public.

    /usr/bin/python3 -m unittest discover -s tests -t . -v

A run taken without the envelope is a run that cannot be compared tomorrow.
These tests are mostly about the two rules that keep it honest.

Everything is recorded raw and local, and exactly one function produces the
shareable view, so the upload path and the report cannot disagree about what
is safe to show. That function is public_view and most of what is below is
trying to get something identifying past it.

A field that could not be read is null with a reason, never omitted and never
defaulted. A missing key reads as an oversight and a zero reads as a
measurement; neither is true of a thing nobody could measure.
"""
import json
import os
import re
import pathlib
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bench  # noqa: E402
import report  # noqa: E402
from fake_device import FakeDevice  # noqa: E402


class Box(unittest.TestCase):
    """A fake Tiiny with bench pointed at it."""

    def setUp(self):
        self.fake = FakeDevice().start()
        self.addCleanup(self.fake.stop)
        saved = (bench.HOST, dict(bench.SERVICES), dict(bench.TRANSPORT),
                 bench.PORT_OVERRIDE, os.environ.get("TIINY_KEY"))

        def restore():
            bench.HOST, bench.PORT_OVERRIDE = saved[0], saved[3]
            bench.SERVICES.clear(), bench.SERVICES.update(saved[1])
            bench.TRANSPORT.clear(), bench.TRANSPORT.update(saved[2])
            if saved[4] is None:
                os.environ.pop("TIINY_KEY", None)
            else:
                os.environ["TIINY_KEY"] = saved[4]
        self.addCleanup(restore)
        bench.HOST, bench.PORT_OVERRIDE = self.fake.host, None
        bench.SERVICES.clear()
        bench.SERVICES.update({n: (self.fake.port, None)
                               for n in ("gateway", "openai", "mgmt")})
        bench.TRANSPORT.clear()
        os.environ["TIINY_KEY"] = "test-key"
        self.cat = {m["id"]: m for m in bench.catalog("test-key")}

    def envelope(self):
        return bench.envelope("test-key", self.cat)


class TestItRecordsWhatChangesTheNumber(Box):
    def test_the_machine_driving_the_benchmark_is_recorded(self):
        # A benchmark driven from a Mac over USB and one driven from a Windows
        # box over Wi-Fi are not the same measurement.
        h = self.envelope()["host"]
        for key in ("os", "os_release", "arch", "python", "cpu_logical", "ram_bytes"):
            self.assertIsNotNone(h[key], f"host.{key} was not recorded")
        self.assertIn("machine_model", h, "a field must not be omitted")

    def test_the_wire_is_recorded_including_how_far_away_the_box_was(self):
        t = self.envelope()["transport"]
        for key in ("plane", "gateway_transport", "gateway_port", "gateway_vhost"):
            self.assertIn(key, t)
        self.assertIsNotNone(t["rtt_ms"], "the round trip was not measured")
        self.assertGreaterEqual(t["rtt_ms"], 0)

    def test_the_device_is_recorded_down_to_its_unit_budget(self):
        d = self.envelope()["device"]
        self.assertEqual(d["serial"], self.fake.state.serial)
        self.assertEqual(d["model"], "Tiiny AI Pocket Lab")
        self.assertEqual(d["tiiny_os"], "1.0.0")
        self.assertEqual(d["npu_units_total"], 100)

    def test_the_tool_records_its_own_version_and_commit(self):
        t = self.envelope()["tool"]
        self.assertEqual(t["bench_version"], bench.VERSION)
        self.assertEqual(t["envelope_schema"], bench.ENVELOPE_SCHEMA)
        self.assertIn("git_commit", t)

    def test_the_resident_set_is_recorded_with_each_models_units(self):
        # The condition that most decides the answer on a box with one
        # accelerator, and the one nothing recorded before.
        c = self.envelope()["conditions"]
        resident = {r["model"]: r["npu_usage"] for r in c["resident"]}
        self.assertEqual(resident.get("deepreinforce-ai/Ornith-1.0-35B"), 50)
        self.assertEqual(c["npu_units_used"], 58)
        self.assertEqual(c["npu_units_free"], 42)

    def test_what_could_not_be_measured_says_so_rather_than_saying_zero(self):
        c = self.envelope()["conditions"]
        self.assertIsNone(c["temperature_c"])
        self.assertIsNone(c["power_w"])
        self.assertIn("temperature_c", c["unavailable"])
        self.assertIn("power_w", c["unavailable"])

    def test_the_envelope_carries_a_schema_number(self):
        self.assertEqual(self.envelope()["schema"], bench.ENVELOPE_SCHEMA)


class TestTheDeviceLock(Box):
    def test_a_run_that_shared_the_box_records_who_it_shared_it_with(self):
        # Turnstile holds an advisory lock around a whole sectioned job. A run
        # that waited behind another app is not a clean run and nothing in a
        # result file would have shown it.
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock = pathlib.Path(tmp.name) / "tiiny.lock"
        lock.write_text(json.dumps({"owner": "warboard/enrich"}), encoding="utf-8")
        saved = bench.LOCK_PATHS
        bench.LOCK_PATHS = (str(lock),)
        self.addCleanup(setattr, bench, "LOCK_PATHS", saved)
        self.assertEqual(bench.conditions("test-key", self.cat)["device_lock_held_by"],
                         "warboard/enrich")

    def test_no_lock_is_recorded_as_nobody_rather_than_as_unknown(self):
        saved = bench.LOCK_PATHS
        bench.LOCK_PATHS = ("/nonexistent/tiiny.lock",)
        self.addCleanup(setattr, bench, "LOCK_PATHS", saved)
        self.assertIsNone(bench.conditions("test-key", self.cat)["device_lock_held_by"])

    def test_reading_the_lock_does_not_take_it(self):
        # A benchmark that fought for the lock would change what it measures.
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock = pathlib.Path(tmp.name) / "tiiny.lock"
        lock.write_text("someone-else", encoding="utf-8")
        saved = bench.LOCK_PATHS
        bench.LOCK_PATHS = (str(lock),)
        self.addCleanup(setattr, bench, "LOCK_PATHS", saved)
        bench.conditions("test-key", self.cat)
        self.assertEqual(lock.read_text(encoding="utf-8"), "someone-else",
                         "the lock file was written to")


class TestPublicView(Box):
    """Everything below is an attempt to get something identifying past it."""

    def record(self):
        return {"label": "sweep", "stamp": "20260919-120000",
                "host": "192.168.100.94",
                "provenance": self.envelope(),
                "models": [{"model": "vendor/m", "results": {"sustained": {
                    "run": {"decode_tok_s": 30.0}}}}]}

    def test_the_serial_becomes_a_hash_and_never_survives_raw(self):
        raw = self.record()
        pub = bench.public_view(raw)
        blob = json.dumps(pub)
        self.assertNotIn(self.fake.state.serial, blob, "the serial was published")
        self.assertEqual(pub["provenance"]["device"]["device_hash"],
                         bench.hash_serial(self.fake.state.serial))

    def test_the_hash_is_stable_so_runs_group_by_box(self):
        a = bench.hash_serial("TNYM26072400300011Q")
        b = bench.hash_serial("TNYM26072400300011Q")
        self.assertEqual(a, b)
        self.assertNotEqual(a, bench.hash_serial("TNYM26072400300012Q"))
        self.assertNotIn("TNYM", a)

    def test_the_address_and_the_vhost_do_not_survive(self):
        blob = json.dumps(bench.public_view(self.record()))
        for leak in ("192.168.100.94", "127.0.0.1", "api.tiiny", "tiiny-fake"):
            self.assertNotIn(leak, blob, f"{leak} was published")

    def test_a_home_path_anywhere_in_the_record_does_not_survive(self):
        raw = self.record()
        raw["provenance"]["conditions"]["device_lock_held_by"] = \
            "/Users/someone/code/thing"
        raw["provenance"]["tool"]["checkout"] = "~/code/tiiny-bench"
        blob = json.dumps(bench.public_view(raw))
        self.assertNotIn("/Users/", blob)
        self.assertNotIn("~/code", blob)

    def test_every_measured_number_survives_untouched(self):
        # This makes a record publishable, not smaller.
        pub = bench.public_view(self.record())
        self.assertEqual(
            pub["models"][0]["results"]["sustained"]["run"]["decode_tok_s"], 30.0)
        self.assertEqual(pub["provenance"]["conditions"]["npu_units_used"], 58)
        self.assertEqual(pub["provenance"]["device"]["npu_units_total"], 100)
        self.assertEqual(pub["provenance"]["host"]["arch"],
                         self.envelope()["host"]["arch"])

    def test_it_records_who_published_it_and_by_what_transform(self):
        pub = bench.public_view(self.record(), account="titanium")
        self.assertEqual(pub["published"]["account"], "titanium")
        self.assertEqual(pub["published"]["transform"], "public_view/1")
        self.assertEqual(pub["published"]["envelope_schema"], bench.ENVELOPE_SCHEMA)

    def test_it_does_not_mutate_the_record_it_was_given(self):
        raw = self.record()
        before = json.dumps(raw)
        bench.public_view(raw)
        self.assertEqual(json.dumps(raw), before, "the local record was changed")


class TestOlderFilesStillWork(unittest.TestCase):
    """Additive means additive: a file from last week has none of this."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        (self.dir / "20260901-000000-suite-old.json").write_text(json.dumps({
            "label": "old", "stamp": "20260901-000000",
            "models": [{"model": "vendor/old", "npu_usage": 30, "results": {
                "sustained": {"run": {"decode_tok_s": 20.0, "out_tokens": 1500,
                                      "wall_s": 75.0, "ttft_s": 0.5}}}}]}),
            encoding="utf-8")

    def test_a_file_with_no_envelope_still_loads(self):
        runs = report.load(self.dir)
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0].get("provenance"))

    def test_a_file_with_no_envelope_still_renders(self):
        dest = self.dir / "out.html"
        report.build(self.dir, dest)
        page = dest.read_text(encoding="utf-8")
        self.assertIn("20.0", page)

    def test_a_file_with_no_envelope_still_exports_as_markdown(self):
        self.assertIn("20.0", report.markdown(self.dir))

    def test_public_view_survives_a_record_that_has_no_envelope(self):
        raw = json.loads(
            (self.dir / "20260901-000000-suite-old.json").read_text(encoding="utf-8"))
        pub = bench.public_view(raw)
        self.assertEqual(
            pub["models"][0]["results"]["sustained"]["run"]["decode_tok_s"], 20.0)
        self.assertEqual(pub["published"]["transform"], "public_view/1")


if __name__ == "__main__":
    unittest.main()


class BothWritersAgree(unittest.TestCase):
    """The terminal and the web app each build their own result record, and
    for one release the web app's was missing the provenance envelope: a
    three-hour sweep driven from the browser could not say which OS, which
    Python or which commit measured it, while a one-model run from the
    terminal could. Nothing failed, because every test asked one writer or
    the other and never asked whether they matched.

    Read both source files rather than run a sweep: this is a question about
    the shape of the record, and the shape is visible in the literal.
    """

    @staticmethod
    def _keys(path, anchor):
        src = pathlib.Path(ROOT, path).read_text()
        i = src.index(anchor)
        chunk = src[i:i + 1400]
        # the dict literal ends at the line that closes it
        end = chunk.index('"models": []}')
        return set(re.findall(r'"([a-z_]+)":', chunk[:end]))

    def test_the_web_app_records_what_the_terminal_records(self):
        cli = self._keys("bench.py", '"label": a.label')
        web = self._keys("serve.py", '"label": label')
        missing = cli - web
        self.assertEqual(
            missing, set(),
            "the web app's result record is missing %s, which the terminal's "
            "records; a sweep driven from the browser would be less "
            "comparable than the same sweep driven from the shell"
            % sorted(missing))
