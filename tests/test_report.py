"""The report: what it says, what it must never say, and the three exports.

    /usr/bin/python3 -m unittest discover -s tests -t . -v

The report is about to feed three surfaces, the app's own page, a public
artifact and a results page on tiinybench.app, all off the same result files.
Two things follow from that and are what most of these tests are about. Every
rendered fact has to come from a result file rather than from the template, or
the three drift apart. And nothing that identifies whose box it was can reach
the output, because two of those three surfaces are public.
"""
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import bench  # noqa: E402  (imported for OUT, and by report)
import report  # noqa: E402

# Two models, enough to make every cross-model chart real, with numbers chosen
# so the arithmetic in the captions is checkable by hand: 30 tok/s on 30 units
# is exactly 1.00 per unit, 12 on 48 is exactly 0.25, and the ratio is 4.
def _model(name, decode, npu, stamp_tag=""):
    return {
        "model": name, "type": "Text Generation", "params": "9B",
        "npu_usage": npu, "total_size": 6_000_000_000, "elapsed_s": 120,
        "results": {
            "prefill": [
                {"prompt_tokens": 70, "ttft_s": 0.5, "prefill_tok_s": 140,
                 "decode_tok_s": decode, "out_tokens": 2},
                {"prompt_tokens": 6000, "ttft_s": 4.0, "prefill_tok_s": 1500,
                 "decode_tok_s": decode, "out_tokens": 2}],
            "sustained": {
                "run": {"out_tokens": 1500, "decode_tok_s": decode, "ttft_s": 0.5,
                        "wall_s": 60.0, "prompt_tokens": 45},
                "during": {"samples": 50, "npu_util_peak": 96.0,
                           "npu_util_median": 30.0, "npu_mem_peak_mb": 9800,
                           "npu_mem_total_mb": 9823}},
            "concurrency": [
                {"parallel": 1, "aggregate_tok_s": 20.0, "per_stream_tok_s": 22.0,
                 "wall_s": 7.0, "ok": 1, "failed": 0},
                {"parallel": 8, "aggregate_tok_s": 21.0, "per_stream_tok_s": 22.5,
                 "wall_s": 53.0, "ok": 8, "failed": 0}],
            "thinking": {
                "off": {"wall_s": 10.0, "out_tokens": 500, "decode_tok_s": 50.0,
                        "ttft_s": 0.4},
                "on": {"wall_s": 30.0, "out_tokens": 1500, "decode_tok_s": 50.0,
                       "ttft_s": 0.4}}}}


class Fixture(unittest.TestCase):
    """A results directory built here, so the numbers are known exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = pathlib.Path(self.tmp.name)
        (self.dir / "20260901-000000-suite-old.json").write_text(json.dumps({
            "label": "old", "stamp": "20260901-000000", "build": "0.1.30",
            "models": [_model("vendor/fast-9b", 99.0, 30)]}), encoding="utf-8")
        (self.dir / "20260919-000000-suite-new.json").write_text(json.dumps({
            "label": "sweep", "stamp": "20260919-000000", "build": "0.1.34",
            "models": [_model("vendor/fast-9b", 30.0, 30),
                       _model("vendor/slow-27b", 12.0, 48)]}), encoding="utf-8")

    def html(self):
        dest = self.dir / "out.html"
        report.build(self.dir, dest)
        return dest.read_text(encoding="utf-8")


class TestNewestWins(Fixture):
    def test_the_cross_model_charts_take_the_newest_run_not_the_best(self):
        # fast-9b measured 99 tok/s in September and 30 today. The report is
        # about what the box does now, so 30 is the number that appears.
        picks = {r["model"]: r for r in report.newest_per_model(report.load(self.dir))}
        self.assertEqual(picks["vendor/fast-9b"]["stamp"], "20260919-000000")
        self.assertEqual(
            picks["vendor/fast-9b"]["results"]["sustained"]["run"]["decode_tok_s"], 30.0)
        page = self.html()
        cross = page[page.index('id="compare"'):page.index('class="runsintro"')]
        self.assertIn("30.0 tok/s", cross)
        self.assertNotIn("99.0 tok/s", cross,
                         "a better old run must not be reached back for")

    def test_the_older_run_is_still_shown_in_its_own_block(self):
        # Newest-wins is about the comparison, not about hiding history.
        self.assertIn("99.0", self.html())


class TestComputedCaptions(Fixture):
    def test_the_efficiency_caption_does_the_arithmetic_from_the_data(self):
        page = self.html()
        # 30 on 30 units is 1.00; 12 on 48 is 0.25; the ratio is 4.0.
        self.assertIn("1.00 tok/s per unit", page)
        self.assertIn("0.25 tok/s per unit", page)
        self.assertIn("4.0x difference", page)

    def test_the_efficiency_caption_spends_the_budget_out_loud(self):
        # The ratio on its own is a fact about arithmetic. What decides a
        # choice is what is left of the box's hundred units afterwards.
        page = self.html()
        self.assertIn("occupies 30 of the 100 units and leaves 70", page)
        self.assertIn("occupies 48 and leaves 52", page)

    def test_a_queueing_box_is_named_as_queueing(self):
        # 21 over 20 is 1.05x, which is flat, and the caption has to say so
        # rather than reporting a 1.05x gain as headroom.
        page = self.html()
        self.assertIn("1.05x", page)
        self.assertIn("queueing them", page)
        self.assertNotIn("real headroom", page)

    def test_the_reasoning_caption_separates_thinking_from_slowing_down(self):
        page = self.html()
        self.assertIn("20.0s", page)          # 30s on against 10s off
        self.assertIn("1,000 tokens", page)   # 1500 against 500


class TestStableColour(unittest.TestCase):
    def test_a_model_keeps_its_colour_when_the_set_changes(self):
        few = report.model_colours(["a/one", "b/two"])
        many = report.model_colours(["a/one", "b/two", "c/three", "d/four"])
        self.assertEqual(few["a/one"], many["a/one"])
        self.assertEqual(few["b/two"], many["b/two"])

    def test_no_two_models_in_one_report_share_a_colour(self):
        ids = ["vendor/model-%d" % i for i in range(14)]
        cols = report.model_colours(ids)
        self.assertEqual(len(set(cols.values())), len(ids))

    def test_the_legend_lists_only_models_that_were_drawn(self):
        # A legend entry for a model that appears in no chart is a small lie.
        runs = report.load(pathlib.Path(ROOT, "bench-results"))
        if not runs:
            self.skipTest("no result files in this checkout")
        section = report.cross_model(runs)
        if not section:
            self.skipTest("not enough models to compare")
        named = set(re.findall(r'</i>([^<]+)</span>', section))
        for name in named:
            self.assertIn(name, section.split('class="legend"')[1],
                          f"{name} is in the legend")
            body = section[section.index("</div>", section.index("legend")):]
            self.assertIn(name, body, f"{name} is in the legend but in no chart")


class TestTheThreeExportsAgree(Fixture):
    def test_markdown_carries_the_same_headline_numbers_as_the_page(self):
        page = self.html()
        md = report.markdown(self.dir)
        for figure in ("30.0", "12.0", "1.00", "0.25"):
            self.assertIn(figure, md, f"{figure} missing from the markdown")
            self.assertIn(figure, page, f"{figure} missing from the page")

    def test_markdown_has_tables_and_no_ascii_art(self):
        md = report.markdown(self.dir)
        self.assertIn("| model | sustained tok/s | NPU units |", md)
        self.assertIn("|---|", md)
        for junk in ("█", "▄", "▏", "+---+", "*****"):
            self.assertNotIn(junk, md, "a chart was drawn in characters")

    def test_markdown_says_the_old_version_stamp_cannot_be_trusted(self):
        self.assertIn("not reliable", report.markdown(self.dir))

    def test_the_page_carries_a_print_stylesheet_with_real_page_breaks(self):
        page = self.html()
        self.assertIn("@media print", page)
        self.assertIn("break-before:page", page)
        self.assertIn("@page{margin", page)
        # A dark ground prints as a sheet of toner, so print repaints it.
        self.assertIn("--ground:#fff", page)


class TestNothingIdentifyingReachesTheOutput(Fixture):
    """Two of the three surfaces this feeds are public."""

    def test_no_owner_no_address_no_path_in_the_page(self):
        page = self.html()
        text = re.sub(r"<[^>]+>", " ", re.sub(r"data:[^\"')]+", "ASSET", page))
        self.checks(text)

    def test_no_owner_no_address_no_path_in_the_markdown(self):
        self.checks(report.markdown(self.dir))

    def checks(self, text):
        for label, pat in (
                ("a name", r"Jason"),
                ("first person", r"\b(we|We|our|Our|us|Us|I|my|My)\b"),
                ("a home path", r"/Users/|~/code"),
                ("an address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
                ("a serial", r"TNY[A-Z0-9]{8,}"),
                ("a key", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
                ("a vhost", r"[a-z0-9]+\.api\.tiiny")):
            found = re.findall(pat, text)
            self.assertFalse(found, f"the output contains {label}: {found[:3]}")

    def test_the_real_report_in_this_checkout_is_also_clean(self):
        out = pathlib.Path(ROOT, "bench-results")
        if not any(out.glob("*.json")):
            self.skipTest("no result files in this checkout")
        dest = self.dir / "real.html"
        report.build(out, dest)
        page = dest.read_text(encoding="utf-8")
        self.checks(re.sub(r"<[^>]+>", " ", re.sub(r"data:[^\"')]+", "ASSET", page)))


class TestChartsFit(unittest.TestCase):
    def test_a_horizontal_bar_leaves_room_for_its_value_and_its_note(self):
        # The value used to be drawn at the end of the bar and the note hard
        # right, and a long value label ran straight into it.
        rows = [("a-very-long-model-name-here", 0.95, "30 tok/s on 32u"),
                ("short", 0.21, "12 tok/s on 55u")]
        svg = report.hbars(rows, w=880, unit=" tok/s per unit", fmt="{:.2f}")
        width = float(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
        # Every text anchor has to start inside the drawing.
        for x in re.findall(r'<text x="([\d.]+)"', svg):
            self.assertLess(float(x), width, "a label starts outside the chart")
        bars = [float(x) + float(w) for x, w in
                re.findall(r'<rect x="([\d.]+)"[^>]*width="([\d.]+)"', svg)]
        longest = max(len("0.95 tok/s per unit  30 tok/s on 32u"),
                      len("0.21 tok/s per unit  12 tok/s on 55u"))
        self.assertLess(max(bars) + longest * 6.7, width + 1,
                        "the longest label does not fit after its bar")

    def test_the_longest_name_clears_the_start_of_its_own_bar(self):
        # The gutter was capped at 200px, which fitted every name but the
        # longest, and that one ran under its own bar with its tail
        # unreadable. Shrinking the font or truncating the name would hide the
        # thing a reader is scanning for, so the gutter grows instead.
        longest = "Qwen3-Coder-30B-A3B-Instruct-Turbo"
        rows = [(longest, 0.54, "29 tok/s on 55u"), ("Qwen3-8B", 0.74, "21 on 28u")]
        svg = report.hbars(rows, w=880, unit=" tok/s per unit", fmt="{:.2f}")
        gutter = float(re.search(r'<text x="([\d.]+)"[^>]*text-anchor="end"', svg).group(1))
        bar_x = float(re.search(r'<rect x="([\d.]+)"', svg).group(1))
        self.assertGreater(gutter, len(longest) * 7.2,
                           "the longest label does not fit in its gutter")
        self.assertGreaterEqual(bar_x, gutter, "a bar starts inside the name gutter")

    def test_the_rows_are_padded_the_same_top_and_bottom(self):
        rows = [("a", 1.0, ""), ("b", 2.0, ""), ("c", 3.0, "")]
        svg = report.hbars(rows, w=660)
        height = float(re.search(r'viewBox="0 0 \d+ ([\d.]+)', svg).group(1))
        bars = [(float(y), float(h)) for y, h in
                re.findall(r'<rect x="[\d.]+" y="([\d.]+)"[^>]*height="([\d.]+)"', svg)]
        top = bars[0][0]
        bottom = height - (bars[-1][0] + bars[-1][1])
        self.assertLess(abs(top - bottom), 3,
                        f"first row sits {top:.1f} from the top, last {bottom:.1f} "
                        f"from the bottom")

    def test_the_value_and_its_note_are_separated(self):
        # "0.95 tok/s per unit 30 tok/s on 32u" is three numbers with nothing
        # between them and parses as one figure at a glance.
        svg = report.hbars([("m", 0.95, "30 tok/s on 32u")], unit=" tok/s per unit",
                           fmt="{:.2f}")
        self.assertIn("&#183;", svg)

    def test_the_many_model_chart_thins_its_tick_labels(self):
        # Twelve models asked for twelve slightly different prompt lengths, and
        # a tick for each was an unreadable smear.
        xs = [70 + i for i in range(12)] + [6000 + i for i in range(12)]
        series = [("m%d" % i, [(x, 100.0) for x in xs], "#fff") for i in range(3)]
        svg = report.multi_line(series, "prompt tokens", "tok/s")
        ticks = re.findall(r'class="tick">([\d,]+)</text>', svg)
        self.assertLess(len(ticks), 8, f"{len(ticks)} tick labels is a smear")


if __name__ == "__main__":
    unittest.main()
