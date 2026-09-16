"""What discovery does, tested without a network.

Nothing here opens a socket to anything outside this machine and nothing here
broadcasts. GitHub's macOS runner refuses a broadcast outright (errno 65), and a
test that put a real datagram on a real interface would be testing whoever's
network the runner happens to be on. So the parser and the candidate lists are
what is tested, which is also where every bug in this area has been.
"""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bench  # noqa: E402


class Interfaces(unittest.TestCase):
    def test_returns_pairs_of_address_and_prefix(self):
        for addr, bits in bench.interfaces(refresh=True):
            self.assertEqual(addr.count("."), 3, addr)
            self.assertIn(bits, (24, 30), addr)

    def test_never_shells_out(self):
        # The 0.1.1 bug: `ip` then `ifconfig`, neither of which exists on
        # Windows, so a Windows tester got an empty candidate list and a scan
        # that could not see a Tiiny on the same network.
        self.assertNotIn("subprocess", bench.__dict__)
        self.assertNotIn("subprocess", pathlib.Path(bench.__file__).read_text()
                         .split("UUID_RE")[0])

    def test_loopback_is_never_a_candidate(self):
        for addr, _ in bench.interfaces(refresh=True):
            self.assertFalse(addr.startswith("127."), addr)

    def test_holds_says_no_to_an_address_this_host_cannot_have(self):
        self.assertFalse(bench._holds("203.0.113.7"))

    def test_holds_says_yes_to_loopback(self):
        self.assertTrue(bench._holds("127.0.0.1"))

    def test_cached_until_refreshed(self):
        first = bench.interfaces(refresh=True)
        self.assertIs(first, bench.interfaces())


class Candidates(unittest.TestCase):
    def setUp(self):
        self._real = bench._IFACES

    def tearDown(self):
        bench._IFACES = self._real

    def test_lan_candidates_is_the_whole_24_minus_this_host(self):
        bench._IFACES = [("192.168.4.20", 24)]
        c = bench.lan_candidates()
        self.assertEqual(len(c), 253)
        self.assertIn("192.168.4.1", c)
        self.assertIn("192.168.4.254", c)
        self.assertNotIn("192.168.4.20", c)

    def test_lan_candidates_skips_loopback_and_the_cable(self):
        bench._IFACES = [("127.0.0.1", 24), ("172.17.3.2", 30)]
        self.assertEqual(bench.lan_candidates(), [])

    def test_lan_candidates_covers_every_card(self):
        bench._IFACES = [("192.168.4.20", 24), ("10.1.1.5", 24)]
        c = bench.lan_candidates()
        self.assertIn("192.168.4.7", c)
        self.assertIn("10.1.1.7", c)

    def test_usb_peer_is_the_other_usable_address_in_the_30(self):
        bench._IFACES = [("172.17.5.2", 30)]
        self.assertEqual(bench.usb_peers(), ["172.17.5.1"])
        bench._IFACES = [("172.17.5.1", 30)]
        self.assertEqual(bench.usb_peers(), ["172.17.5.2"])

    def test_usb_peers_ignores_a_lan_address(self):
        bench._IFACES = [("192.168.4.20", 24)]
        self.assertEqual(bench.usb_peers(), [])

    def test_udp_targets_is_the_broadcast_plus_every_cable(self):
        bench._IFACES = [("172.17.5.2", 30), ("192.168.4.20", 24)]
        self.assertEqual(bench.udp_targets(), ["255.255.255.255", "172.17.5.1"])


class UdpParser(unittest.TestCase):
    def test_a_good_answer_becomes_a_record(self):
        payload = json.dumps({"serial_number": "TNY1", "device_name": "My Tiiny",
                              "transport": ["lan"]}).encode()
        self.assertEqual(
            bench.udp_record(payload, "192.168.4.9"),
            {"addr": "192.168.4.9", "serial": "TNY1", "name": "My Tiiny",
             "transport": ["lan"]})

    def test_without_a_serial_it_is_not_a_tiiny(self):
        self.assertIsNone(bench.udp_record(b'{"device_name":"something"}', "10.0.0.1"))

    def test_rubbish_on_the_port_is_ignored(self):
        self.assertIsNone(bench.udp_record(b"not json at all", "10.0.0.1"))
        self.assertIsNone(bench.udp_record(b"[1,2,3]", "10.0.0.1"))
        self.assertIsNone(bench.udp_record(b"\xff\xfe\x00", "10.0.0.1"))

    def test_the_token_is_the_one_the_responder_answers(self):
        self.assertEqual(bench.UDP_TOKEN, b"GADGET_DISCOVER_V1")
        self.assertEqual(bench.UDP_PORT, 39217)
        self.assertEqual(bench.DISCO_PORT, 39218)


class WhereWithNoBox(unittest.TestCase):
    def test_connect_soft_reports_instead_of_raising(self):
        w, err = bench.connect_soft(host="203.0.113.7")
        # An address that answers nothing must not raise out of connect_soft.
        self.assertTrue(w is None or isinstance(w, dict))
        if w is None:
            self.assertTrue(err)

    def test_the_not_found_message_names_every_probe(self):
        src = pathlib.Path(bench.__file__).read_text()
        self.assertIn("asked the responder on", src)
        self.assertIn("A box on another network hears none of that", src)


class PageLoads(unittest.TestCase):
    """The 0.1.1 field bug, as a test that does not need a browser.

    Every button on the page was dead because the script referenced $ above the
    line that defines it, so the whole <script> stopped on a ReferenceError
    before it ever asked /api/device. Nothing in the app can be used after that,
    which is what both testers reported.
    """

    def setUp(self):
        self.src = (pathlib.Path(bench.__file__).parent
                    / "static" / "app.html").read_text()

    def test_the_dollar_helper_is_defined_before_it_is_used(self):
        define = self.src.index("var $ = function(id)")
        first_use = self.src.index("<script>")
        body = self.src[first_use:define]
        self.assertNotIn("$('", body,
                         "something calls $ before the line that defines it")

    def test_one_scope_only(self):
        # Two scopes was how the first use ended up above the definition.
        self.assertEqual(self.src.count('var $ = function(id)'), 1)

    def test_the_panel_says_what_it_is_doing_before_it_knows(self):
        self.assertIn("Looking for your Tiiny", self.src)

    def test_a_missing_key_asks_for_one(self):
        self.assertIn("No key yet. Paste it here", self.src)

    def test_nothing_found_is_said_out_loud(self):
        self.assertIn("None answered", self.src)
        self.assertIn("A Tiiny on another network hears none of that", self.src)

    def test_nothing_found_still_offers_the_address_and_the_key(self):
        # The panel exists to be used when detection failed, so both boxes are
        # in the markup unconditionally rather than rendered on success.
        self.assertIn('id="connhost"', self.src)
        self.assertIn('id="connkey"', self.src)
        self.assertIn('id="credetect"', self.src)

    def test_routing_does_not_wait_on_the_catalog(self):
        # go() used to be inside catalog().then(...), so a slow or failed
        # request left every nav button inert.
        self.assertNotIn("catalog().then(function(){ return board(); }).then(",
                         self.src)


class NarrowScreen(unittest.TestCase):
    """390 wide, which is the width Jason checks every panel at."""

    def setUp(self):
        self.src = (pathlib.Path(bench.__file__).parent
                    / "static" / "app.html").read_text()

    def test_the_nav_strip_scrolls_inside_itself(self):
        # Without min-width:0 the nav grew to its content and took the page to
        # 1083px at a 390 viewport, so every panel sat behind a sideways scroll.
        block = self.src[self.src.index("@media (max-width:820px)"):]
        block = block[:block.index("}\n\n")]
        self.assertIn("min-width:0", block)
        self.assertIn("minmax(0,1fr)", block)
        self.assertIn("overflow-x:auto", block)


class NoEmDashes(unittest.TestCase):
    """House rule. Written as an escape so this file does not contain one."""

    EM = "\u2014"

    def test_nowhere_a_person_reads(self):
        here = pathlib.Path(bench.__file__).parent
        for f in ["bench.py", "serve.py", "report.py", "static/app.html",
                  "tiiny-app.json", "tests/test_discovery.py", "README.md",
                  ".gitignore", ".github/workflows/ci.yml"]:
            n = (here / f).read_text().count(self.EM)
            self.assertEqual(n, 0, f"{f} has {n} em dashes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
