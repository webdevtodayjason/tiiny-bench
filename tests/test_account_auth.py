"""account_auth_key(): the one call that gets the gateway key without TiinyOS.

See ~/code/tiiny/tools/README-unlock.md for how this route was found and
confirmed. This test only checks that bench.py parses the response the way
the real device answers -- it is not a re-test of the discovery itself.
"""
import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bench


class _Handler(BaseHTTPRequestHandler):
    reply = {"status": "ok", "data_state": "unlocked", "auth_key": "the-key"}
    status = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.last_body = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps(self.reply).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestAccountAuthKey(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.addr = "127.0.0.1:%d" % self.server.server_port
        import threading
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)

    def test_returns_the_auth_key_field(self):
        got = bench.account_auth_key(self.addr, "TNYM000", "correct horse")
        self.assertEqual(got, "the-key")

    def test_empty_auth_key_is_none_not_empty_string(self):
        _Handler.reply = {"status": "ok", "auth_key": ""}
        got = bench.account_auth_key(self.addr, "TNYM000", "wrong")
        self.assertIsNone(got)
        _Handler.reply = {"status": "ok", "data_state": "unlocked", "auth_key": "the-key"}

    def test_unreachable_host_is_none_not_a_crash(self):
        got = bench.account_auth_key("127.0.0.1:1", "TNYM000", "x")
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
