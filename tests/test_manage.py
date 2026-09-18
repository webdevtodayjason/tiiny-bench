"""The Models page's catalogue: read live, stamped, and honest when it is not.

Nothing about the vendor catalogue is pinned or kept in this repo. bench.online
reads it off the box on every call to /api/manage, and the question this file
answers is what the page says when that read fails: it used to come back as an
empty list, which the page printed as "0 in the catalogue", and an empty store
and a question nobody answered are not the same claim.

    /usr/bin/python3 -m unittest discover -s tests -t . -v

The fake device in tests/fake_device.py serves the catalogue, and its
`online_fails` switch is how the failing read is provoked.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bench  # noqa: E402
from fake_device import CATALOG  # noqa: E402
from test_chat import ServerCase  # noqa: E402


class TestCatalogueFreshness(ServerCase):
    # /api/manage also reads bench.telemetry, which addresses the device
    # without the gateway port. Against the fake that is a refused connection
    # and a row of nulls, which is fine here: nothing below reads it.

    def test_a_good_read_is_counted_and_stamped(self):
        status, payload = self.request("/api/manage")
        self.assertEqual(status, 200)
        self.assertIsNone(payload["online_error"])
        self.assertEqual(len(payload["online"]), len(CATALOG))
        # An ISO stamp with an offset, so the browser can print a local time
        # rather than guessing which clock it came off.
        self.assertRegex(payload["online_read_at"],
                         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")

    def test_a_failed_read_says_so_instead_of_reporting_an_empty_store(self):
        self.fake.state.online_fails = "the model store did not answer"
        status, payload = self.request("/api/manage")
        self.assertEqual(status, 200, "the rest of the page still works")
        self.assertEqual(payload["online"], [])
        self.assertIsNotNone(payload["online_error"])
        self.assertIn("the model store did not answer", payload["online_error"])
        # Everything read off the box itself is still there, so one dead
        # service does not blank the page.
        self.assertTrue(payload["installed"])
        self.assertTrue(payload["running"])
        self.assertEqual(payload["npu"]["total"], 100)

    def test_the_helper_raises_rather_than_returning_an_empty_list(self):
        """The silent zero was made here, so this is where it is held."""
        self.fake.state.online_fails = "the model store did not answer"
        with self.assertRaises(bench.DeviceError):
            bench.online("test-key")
        self.fake.state.online_fails = None
        self.assertEqual(len(bench.online("test-key")), len(CATALOG))


class TestCataloguePage(unittest.TestCase):
    """What the Models page does with the three fields."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(HERE)
        with open(os.path.join(root, "static", "app.html"), encoding="utf-8") as handle:
            cls.body = handle.read()

    def test_the_page_reads_all_three_fields(self):
        for field in ("online_error", "online_read_at", "mcatline"):
            self.assertIn(field, self.body, field)

    def test_a_failed_read_never_prints_a_count(self):
        """Both places that would have said zero say the read failed instead."""
        self.assertIn("the box did not answer for the catalogue: ", self.body)
        self.assertIn("catalogue unread", self.body)


if __name__ == "__main__":
    unittest.main()
