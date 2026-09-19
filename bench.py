"""TiinyBench - an independent benchmark suite for the Tiiny AI Pocket Lab.

    tiiny-bench --label turbo-16aug          benchmark whatever is loaded
    tiiny-bench --all                        sweep every text model on the box
    tiiny-bench --model Qwen/Qwen3-8B        benchmark one model by name
    tiiny-bench --only prefill,concurrency   run a subset of the tests
    tiiny-bench --catalog                    list what is installed
    tiiny-bench --report                     build the HTML report from results

Measures the things a spec sheet does not: how prefill scales with prompt
length, whether throughput holds over a long generation, what happens when more
than one person uses the box at once, and what a reasoning model's hidden tokens
actually cost. Records NPU utilisation and memory alongside every number,
because tokens/sec without the load context is not evidence.

BY DEFAULT NOTHING IS LOADED OR UNLOADED. The suite runs against whatever model
is already running and leaves the box exactly as it found it. `--all` and
`--model` are the two flags that change that, and they say so before they start.
"""
import argparse
import base64
import errno
import glob
import hashlib
import json
import pathlib
import platform
import re
import socket
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import os
import zlib

VERSION = "0.1.9"

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "bench-results"

# ------------------------------------------------------------------ finding it
#
# Firmware 1.0 changed how a Tiiny is addressed, twice over: the AI gateway no
# longer answers on its own port from another machine, and a box's LAN address
# is a DHCP lease that moves. So nothing here is a constant any more. The
# address is resolved at startup, in this order, and the source is always
# printed so a run can never quietly measure a box you did not mean:
#
#   TIINY_BASE / TIINY_KEY      what the farm CLI puts in the environment
#   ~/.tiinyapps/device.json    what `farm device` writes, {"base", "key"}
#   --host                      an address typed on the command line
#   the saved config below      whatever the last successful run found
#   a scan                      USB links first, then this host's own /24
#
# The scan is the only step that identifies a box rather than just finding an
# open port, because :39218/device.json carries the serial.

# Where a saved device and key live, so the scan runs once rather than on
# every invocation. The web UI writes this too.
CONFIG = pathlib.Path(
    os.environ.get("XDG_CONFIG_HOME") or (pathlib.Path.home() / ".config")
) / "tiiny-bench.json"

# What the farm hands an app when it gives it a box: {"base": ..., "key": ...}.
FARM_DEVICE = pathlib.Path.home() / ".tiinyapps" / "device.json"

# Unauthenticated device metadata. The only endpoint that answers identically
# on every firmware and both transports, and the only one carrying the serial.
DISCO_PORT = 39218
DISCO_PATH = "/device.json"

# A USB-attached Tiiny is a point-to-point /30 inside 172.17/16.
USB_NET = "172.17."

# The responder every Tiiny runs. One datagram to the broadcast address finds a
# box whose DHCP lease moved, and finds it without a 254-address sweep. Same
# port and same token as the farm CLI, because it is the same responder.
UDP_PORT = 39217
UDP_TOKEN = b"GADGET_DISCOVER_V1"

# Names the TiinyOS desktop app puts in /etc/resolver, pointed at a proxy on
# this machine. They work on a Mac running that app and nowhere else, so they
# are a shortcut tried after the scan finds nothing, not the primary route.
PROXY_HOSTS = ("tiiny", "openai.api.tiiny", "tiiny.local")

# ---------------------------------------------------------------- transport
#
# Firmware 1.0 moved the AI gateway behind port 80. It still binds
# 172.17.0.1:8800 for the container bridge, so a direct connect from another
# machine now gets ECONNREFUSED, and every service arrives on 80 where a router
# picks one out of the Host header.
#
# Both shapes are in the field, so both are supported: the service's own port
# is tried first and ONLY a refused connection moves it to the vhost. A timeout
# or a bad address must not, or a busy box looks like old firmware and the real
# fault gets buried under a second, less honest error.
#
#   service      own port   vhost on port 80
SERVICES = {
    "gateway": (8800, "p8800.api.tiiny"),   # models, NPU, inference
    "openai":  (8800, "openai.api.tiiny"),  # the OpenAI-compatible surface
    "mgmt":    (80,   None),                # device management, always on 80
}

# service -> "direct" or "vhost". Decided by the first call that gets an
# answer and reused for the rest of the run, so the dead port is probed once.
TRANSPORT = {}

HOST = ""          # resolved by connect()
PORT_OVERRIDE = None   # a port carried in TIINY_BASE, or TIINY_PORT
SOURCE = ""        # which step of the chain above answered
PLANE = ""         # "usb", "lan", "proxy" or "given"
DEVICE = {}        # what device.json said, when we got it from a scan
KEY_SOURCE = ""    # where key() found the key, set by key()


class Call:
    """A path on one of the device's services, not yet a URL.

    The URL cannot be fixed at the call site any more: which port and which
    headers reach a service depends on the firmware, and the only way to know
    is to try. So call sites name a service and a path and api() resolves it.
    """
    __slots__ = ("service", "path")

    def __init__(self, service, path):
        self.service = service
        self.path = path

    def __str__(self):
        return "%s%s" % (self.service, self.path)


def gw(path):
    """Models, NPU and inference."""
    return Call("gateway", path)


def oai(path):
    """The OpenAI-compatible surface."""
    return Call("openai", path)


def mgmt(path):
    """Device management: /api/v1/sys/*."""
    return Call("mgmt", path)


def _own_port(service):
    """The service's own port, before any vhost fallback."""
    port = SERVICES[service][0]
    if service != "mgmt" and PORT_OVERRIDE:
        return PORT_OVERRIDE
    return port


def _attempts(target):
    """(url, extra headers, mode) to try for a target, best first."""
    if isinstance(target, str):
        return [(target, {}, None)]
    vhost = SERVICES[target.service][1]
    known = TRANSPORT.get(target.service)
    modes = [known] if known else (["direct", "vhost"] if vhost else ["direct"])
    out = []
    for m in modes:
        if m == "vhost" and vhost:
            out.append(("http://%s:80%s" % (HOST, target.path), {"Host": vhost}, m))
        elif m == "direct":
            out.append(("http://%s:%d%s" % (HOST, _own_port(target.service),
                                            target.path), {}, m))
    return out


def _mark(target, mode):
    """Remember which transport reached a service, for the rest of the run."""
    if mode and not isinstance(target, str):
        TRANSPORT[target.service] = mode


def _refused(exc):
    """True only for "nothing is listening on that port".

    Deliberately narrow. This is the one condition that means the firmware has
    moved the service rather than that the address or the box is wrong.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return False                    # it answered, so the port is open
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ConnectionRefusedError):
        return True
    return getattr(reason, "errno", None) == errno.ECONNREFUSED


def _config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent or unreadable is just "nothing saved"
        return {}


def save_config(**kw):
    """Remember a host or a key. Written 0600: the key is a root credential."""
    cfg = _config()
    cfg.update({k: v for k, v in kw.items() if v is not None})
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    try:
        CONFIG.chmod(0o600)
    except OSError:
        pass
    return cfg


def _farm_device():
    """What `farm device` wrote, or an empty dict."""
    try:
        d = json.loads(FARM_DEVICE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _split_base(base):
    """A base URL or bare address as (address, explicit port or None).

    Accepts everything the farm and the environment actually contain:
    "1.2.3.4", "http://1.2.3.4", "http://1.2.3.4:8800/", a hostname, or a
    hostname with a port.
    """
    base = (base or "").strip()
    if not base:
        return None, None
    if "//" not in base:
        base = "http://" + base
    try:
        u = urllib.parse.urlsplit(base)
        return (u.hostname or None), u.port
    except ValueError:
        return None, None


# -------------------------------------------------------------- discovery
def device_json(addr, timeout=0.6):
    """What a box at this address says about itself, or None.

    Unauthenticated, and the response carries serial_number, which is what
    lets two addresses for the same box be recognised as one box.
    """
    try:
        with urllib.request.urlopen(
                "http://%s:%d%s" % (addr, DISCO_PORT, DISCO_PATH),
                timeout=timeout) as r:
            d = json.load(r)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict) or not d.get("serial_number"):
        return None
    return d


DISCO_SEEN = {}    # address -> its device.json body, or {} when it said nothing


def discovered(addr, timeout=1.5):
    """The device.json for an address, asked once and then remembered.

    A box handed over in the environment is never scanned for, so this file is
    the only unauthenticated description of it we get. The connection panel
    repaints on a timer and a box does not rename itself between repaints, so
    asking once per address is enough; connect() clears this, which is what the
    Detect again button runs.
    """
    if not addr:
        return {}
    if addr not in DISCO_SEEN:
        DISCO_SEEN[addr] = device_json(addr, timeout=timeout) or {}
    return DISCO_SEEN[addr]


def _holds(addr):
    """Whether this machine holds this IPv4 address.

    bind succeeds only on an address the host really has and sends nothing, so
    this asks the operating system directly and answers the same way on macOS,
    Linux and Windows. Nothing is granted or refused here: no packet leaves.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((addr, 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _usb_addresses():
    """This machine's own side of every USB cable, by bind over 172.17/16.

    A /30 holds a network address, two usable ones and a broadcast, and this
    machine takes one of the two usable ones, so probing those two in each block
    of four covers the whole /16 in 32768 binds, about a third of a second.
    """
    found = []
    for third in range(256):
        for block in range(0, 256, 4):
            for last in (block + 2, block + 1):
                addr = "%s%d.%d" % (USB_NET, third, last)
                if _holds(addr):
                    found.append(addr)
                    break
    return found


def _lan_addresses():
    """Every non-loopback IPv4 this host answers to, as far as the stdlib knows.

    Two sources because neither is complete on its own. A UDP socket connected
    to a documentation address names the interface this host would route from,
    which is the one a Tiiny on the network is almost always on, and connecting
    a datagram socket sends nothing. The host's own name adds the rest on a
    machine with more than one card. Deduped, order kept.
    """
    found = []

    def add(a):
        if (a and not a.startswith("127.") and not a.startswith(USB_NET)
                and a.count(".") == 3 and a not in found):
            found.append(a)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # TEST-NET-3, reserved for documentation, so nothing is contacted even
        # if a stack were to decide to send something.
        s.connect(("203.0.113.1", 9))
        add(s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET, socket.SOCK_DGRAM):
            add(info[4][0])
    except OSError:
        pass
    return found


_IFACES = None


def interfaces(refresh=False):
    """(address, prefix length) for every IPv4 this host holds.

    Stdlib only, and identical on every platform. The version that shipped in
    0.1.1 ran `ip` and then `ifconfig` and Windows has neither, so a Windows
    tester got an empty list, an empty candidate list, and a scan that found
    nothing while a Tiiny sat on the same network. Nothing here shells out.

    The prefix length is exact for a cable, which is a /30 by definition, and is
    taken as /24 for a LAN address because neither stdlib source reports a
    netmask. lan_candidates() only ever reads the first three octets, so a /24
    is the whole of what it needs; a host on a wider prefix still gets its own
    /24 swept, which is the part a benchmark has any business touching.

    Cached, because one scan asks twice and the cable probe is the expensive
    half. scan() refreshes it, so plugging a cable in and pressing Detect again
    finds the cable.
    """
    global _IFACES
    if _IFACES is None or refresh:
        _IFACES = ([(a, 30) for a in _usb_addresses()]
                   + [(a, 24) for a in _lan_addresses()])
    return _IFACES


def usb_peers():
    """The device end of every attached USB link.

    A Tiiny's USB interface is a point-to-point /30: four addresses, of which
    the box takes the first usable one and the host the second. So the peer is
    arithmetic rather than a guess, and somebody with four boxes plugged in has
    four of these on four separate /30s.
    """
    peers = []
    for addr, bits in interfaces():
        if bits != 30 or not addr.startswith(USB_NET):
            continue
        try:
            o = [int(x) for x in addr.split(".")]
        except ValueError:
            continue
        n = (o[0] << 24) | (o[1] << 16) | (o[2] << 8) | o[3]
        base = n & ~3
        for cand in (base + 1, base + 2):
            if cand != n:
                peers.append("%d.%d.%d.%d" % (
                    (cand >> 24) & 255, (cand >> 16) & 255,
                    (cand >> 8) & 255, cand & 255))
    return peers


def lan_candidates():
    """Every other address in the /24 around each of this host's LAN addresses.

    A /24 is 254 probes, about a second threaded, and it is the only thing that
    finds a box whose DHCP lease moved. Only the /24 the host sits in: sweeping
    a /16 to locate a Tiiny is not something a benchmark should do to
    somebody's network.
    """
    out, seen = [], set()
    for addr, bits in interfaces():
        if addr.startswith("127.") or addr.startswith(USB_NET) or bits >= 31:
            continue
        head = addr.rsplit(".", 1)[0]
        for i in range(1, 255):
            cand = "%s.%d" % (head, i)
            if cand != addr and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def udp_record(data, where):
    """The one device a datagram describes, or None.

    A pure function on bytes so the parser can be tested without a socket. A
    test that broadcast for real would be testing somebody's network, and
    GitHub's macOS runner refuses broadcast outright (errno 65), so the wire is
    exactly the part that is never exercised in CI.
    """
    try:
        d = json.loads(data.decode("utf-8", "replace"))
    except ValueError:
        return None
    if not isinstance(d, dict) or not d.get("serial_number"):
        return None
    return {"addr": where, "serial": d.get("serial_number"),
            "name": d.get("device_name"), "transport": d.get("transport")}


def udp_targets():
    """Who to ask: the whole network, then each cable.

    The broadcast address is what finds a box whose lease moved, and it is the
    only probe here that can find a box this host has no /24 in common with.
    Each cable is asked directly as well, because a point-to-point /30 does not
    carry a broadcast worth the name.
    """
    return ["255.255.255.255"] + usb_peers()


def udp_scan(timeout=1.2, grace=0.4):
    """Every Tiiny in earshot of one datagram each.

    One packet does what a 254-address sweep does, and does it for a box on a
    network this host is not sweeping. A host that forbids broadcast, which is
    what a CI runner does, is not an error: the sweep below still runs, so this
    can only ever add.
    """
    found = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return found
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass  # A host that forbids broadcast can still be asked directly.
        asked = False
        for t in udp_targets():
            try:
                sock.sendto(UDP_TOKEN, (t, UDP_PORT))
                asked = True
            except OSError:
                continue
        if not asked:
            return found
        seen = set()
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                sock.settimeout(max(0.05, left))
                data, where = sock.recvfrom(65535)
            except (socket.timeout, OSError):
                break
            rec = udp_record(data, where[0])
            if not rec or (where[0], rec["serial"]) in seen:
                continue
            seen.add((where[0], rec["serial"]))
            found.append(rec)
            # The rest of the budget was only ever for boxes that are not there.
            deadline = min(deadline, time.monotonic() + grace)
    finally:
        sock.close()
    return found


def scan(timeout=0.6, workers=64):
    """Every Tiiny this host can see, deduped by serial, USB first.

    Three ways, in the order a found address is worth trusting, which is the
    order the farm CLI uses because it is the same responder answering.

      the cable    a /30 peer, asked for :39218/device.json. Cannot move.
      a datagram   GADGET_DISCOVER_V1 to the broadcast address. This is the one
                   that finds a box whose DHCP lease moved, and the only one
                   that can find a box outside this host's own /24.
      the sweep    every address in this host's /24 on :39218.

    A box on Wi-Fi and USB at once answers on all three and they are one device.
    USB wins the tie: a /30 handed out by the cable cannot move, while a DHCP
    lease can and did.
    """
    interfaces(refresh=True)
    found = {}
    lock = threading.Lock()

    def keep(rec):
        cur = found.get(rec["serial"])
        if cur is None or (cur["plane"] == "lan" and rec["plane"] == "usb"):
            found[rec["serial"]] = rec

    def probe(addr, plane):
        d = device_json(addr, timeout)
        if not d:
            return
        rec = {"addr": addr, "plane": plane,
               "serial": d.get("serial_number"),
               "name": d.get("device_name"),
               "transport": d.get("transport")}
        with lock:
            keep(rec)

    peers = usb_peers()
    for plane, addrs in (("usb", peers), ("lan", lan_candidates())):
        if plane == "lan":
            # Between the cable and the sweep, because a datagram costs one
            # packet and the sweep costs 254, and an answer here can name an
            # address the sweep would never have reached.
            for rec in udp_scan():
                rec["plane"] = "usb" if rec["addr"] in peers else "lan"
                with lock:
                    keep(rec)
        for i in range(0, len(addrs), workers):
            ths = [threading.Thread(target=probe, args=(a, plane))
                   for a in addrs[i:i + workers]]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
    return sorted(found.values(), key=lambda r: (r["plane"] != "usb", r["addr"]))


def reachable(host, timeout=2.5):
    """Does the AI gateway answer at this address, either way round?

    401 counts as found: the gateway is listening and wants a key, which is
    exactly the thing we are looking for. Only 404 and a dead socket mean "not
    a Tiiny". Runs outside the transport cache on purpose, so probing a
    candidate cannot leave a note about a box we do not end up using.
    """
    vhost = SERVICES["gateway"][1]
    for url, extra in (
            ("http://%s:%d/v1/models" % (host, _own_port("gateway")), {}),
            ("http://%s:80/v1/models" % host, {"Host": vhost})):
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer probe", **extra})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 404:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def connect(host=None, serial=None, rescan=False, quiet=False):
    """Settle on a box and on how to reach it. Returns where().

    Everything that can go wrong here is worth a sentence rather than a
    traceback, so the failures raise SystemExit with the thing to do next.
    """
    global HOST, PORT_OVERRIDE, SOURCE, PLANE, DEVICE
    TRANSPORT.clear()
    DISCO_SEEN.clear()
    DEVICE = {}
    env_port = os.environ.get("TIINY_PORT")
    forced_port = int(env_port) if env_port and env_port.isdigit() else None
    # Set before anything probes, so a candidate is probed on the port the
    # caller says the gateway is on rather than on the built-in one.
    PORT_OVERRIDE = forced_port

    def settle(addr, port, source, plane, dev=None):
        global HOST, PORT_OVERRIDE, SOURCE, PLANE, DEVICE
        HOST, SOURCE, PLANE = addr, source, plane
        PORT_OVERRIDE = forced_port or (port if port and port != 80 else None)
        DEVICE = dev or {}
        if not DEVICE.get("serial"):
            # One cheap unauthenticated request, so a result file records which
            # box it measured even when the address came from the environment
            # rather than from a scan. Harmless when it does not answer.
            d = discovered(addr)
            if d.get("serial_number"):
                DEVICE = {"addr": addr, "plane": plane,
                          "serial": d.get("serial_number"),
                          "name": d.get("device_name"),
                          "transport": d.get("transport")}
        probe_gateway()
        return where()

    # TIINY_HOST is what this suite documented before the farm existed, and
    # people have it exported. It still works, it is just no longer the name.
    base, port = _split_base(os.environ.get("TIINY_BASE")
                             or os.environ.get("TIINY_HOST"))
    if base:
        var = "TIINY_BASE" if os.environ.get("TIINY_BASE") else "TIINY_HOST"
        if host and host != base and not quiet:
            say("  note: %s=%s is being used, not --host %s. "
                "Unset %s to use the flag." % (var, base, host, var))
        return settle(base, port, var, "given")

    base, port = _split_base(_farm_device().get("base"))
    if base:
        if host and host != base and not quiet:
            say("  note: %s is being used, not --host %s." % (FARM_DEVICE, host))
        return settle(base, port, str(FARM_DEVICE), "given")

    if host:
        addr, port = _split_base(host)
        if not addr:
            sys.exit("--host %r is not an address." % host)
        return settle(addr, port, "--host", "given")

    if not rescan and not serial:
        saved, port = _split_base(_config().get("host"))
        if saved and reachable(saved):
            return settle(saved, port, str(CONFIG), _config().get("plane") or "saved")

    boxes = scan()
    if serial:
        boxes = [b for b in boxes if b["serial"] == serial
                 or b["serial"].endswith(serial)]
        if not boxes:
            sys.exit("no Tiiny with serial %r answered. "
                     "Run without --serial to see what is here." % serial)
    if len(boxes) > 1:
        lines = ["", "  More than one Tiiny answered. Pick one:", ""]
        for b in boxes:
            lines.append("    %-16s %-5s %-24s %s" % (
                b["addr"], b["plane"], b["serial"], b["name"] or ""))
        lines += ["", "    tiiny-bench --serial %s ..." % boxes[0]["serial"],
                  "    tiiny-bench --host %s ..." % boxes[0]["addr"], ""]
        sys.exit("\n".join(lines))
    if boxes:
        b = boxes[0]
        save_config(host=b["addr"], plane=b["plane"])
        return settle(b["addr"], None, "scan", b["plane"], b)

    # Nothing on either plane. On a Mac running the TiinyOS app the proxy
    # names still work, and they cost one request each, so try them last.
    for name in PROXY_HOSTS:
        if reachable(name, timeout=1.5):
            save_config(host=name, plane="proxy")
            return settle(name, None, "proxy name", "proxy")

    sys.exit(
        "No Tiiny found.\n"
        "  Looked for a USB /30 peer on :%d, asked the responder on :%d, swept "
        "this host's own /24 on :%d, and tried %s.\n"
        "  A box on another network hears none of that: give it an address with "
        "--host, or set TIINY_BASE." % (
            DISCO_PORT, UDP_PORT, DISCO_PORT, ", ".join(PROXY_HOSTS)))


def connect_soft(**kw):
    """connect(), but a failure is reported rather than fatal.

    The web UI has to come up even with no box on the network, because asking
    for an address is one of the things it is there to do.
    """
    try:
        return connect(**kw), ""
    except SystemExit as exc:
        return None, str(exc)


def probe_gateway(timeout=3.0):
    """Settle which transport reaches the gateway before anything is measured.

    The key is deliberately bogus: a 401 proves the gateway is there, and we
    would rather learn which port serves it here than three tests into a run.
    """
    api(gw("/v1/models"), "probe", timeout=timeout)
    return TRANSPORT.get("gateway")


def round_trip(tok, n=7):
    """How long an empty-handed call to the gateway takes, end to end.

    Every measured number on this box includes the trip to it. A benchmark
    driven from this Mac over the USB tunnel and one driven from a laptop on
    the LAN are not comparable until the reader knows what that trip costs,
    and time to first token is where it shows up most. One cheap GET against
    the same host, port and vhost that inference uses, several times, median
    reported: the median rather than the mean because the first call after an
    idle period is always the slow one and it should not set the figure.
    """
    ms = []
    for i in range(n):
        t0 = time.time()
        r = api(gw("/api/v1/models/running"), tok, timeout=20)
        if isinstance(r, dict) and "_error" in r:
            continue
        ms.append((time.time() - t0) * 1000)
    if not ms:
        return {"probe": "/api/v1/models/running", "n": 0,
                "note": "the probe call did not answer, so nothing was timed"}
    ms.sort()
    return {"probe": "/api/v1/models/running", "n": len(ms),
            "median_ms": round(statistics.median(ms), 1),
            "min_ms": round(ms[0], 1), "max_ms": round(ms[-1], 1)}


def where():
    """One dict describing the connection, for the UI and the result file."""
    mode = TRANSPORT.get("gateway") or "unknown"
    return {
        "host": HOST,
        "source": SOURCE,
        "plane": PLANE,
        "serial": DEVICE.get("serial"),
        "gateway_transport": mode,
        "gateway_port": 80 if mode == "vhost" else _own_port("gateway"),
        "gateway_vhost": SERVICES["gateway"][1] if mode == "vhost" else None,
        "services": dict(TRANSPORT),
    }


def firmware(info):
    """The two version numbers out of a device_info response.

    TiinyOS and the device service ship on separate version lines and a bug
    report needs both, so both go in the result file rather than the one the
    report header happens to show.
    """
    if not isinstance(info, dict) or "_error" in info:
        return {}
    return {"tiiny_os": info.get("tiiny_os"), "version": info.get("version")}


def identify():
    """What this device says it is. Unauthenticated where it can be, so the
    web UI can show the user what it found before asking them for anything.

    Management lives on port 80 and a box handed over by the launcher is not
    always reachable there, so the discovery file on :39218 is asked for the
    same address and fills in anything management could not answer. Without it
    a perfectly healthy box reads as "found, unnamed" under a column of dashes.
    A present-but-empty field counted as answered before, which is the other
    way that column filled with dashes.
    """
    out = {}
    d = api(mgmt("/api/v1/sys/device_info"), "probe", timeout=5)
    if "_error" not in d:
        out = {"name": d.get("device_name"), "model": d.get("device_model_name"),
               "os": d.get("tiiny_os"), "service": d.get("version"),
               "ram": d.get("ram"), "storage": d.get("storage"),
               "serial": d.get("sn")}
    disco = discovered(HOST)
    if disco:
        if not out.get("name"):
            out["name"] = disco.get("device_name")
        out["serial"] = disco.get("serial_number") or out.get("serial")
        out["transport"] = disco.get("transport")
    return {k: v for k, v in out.items() if v}

# Everything the suite says out loud goes through say(). The CLI prints it;
# the server hands it a sink as well so the same words stream to the browser.
# One indirection, so the tests below never have to know which one is watching.
SINK = None


def say(line=""):
    print(line)
    if SINK:
        try:
            SINK(line)
        except Exception:
            pass


def emit(kind, **data):
    """Structured progress, for the web UI's benefit. The CLI ignores it."""
    if SINK:
        try:
            SINK(None, kind, data)
        except Exception:
            pass


IMAGE_PROMPTS = [
    "a lighthouse on a rocky shore at dusk, painterly",
    "a bowl of oranges on a wooden table, soft window light",
    "a fox asleep under a fern, children's book illustration",
]
SPEECH_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Engineers measured the throughput carefully and recorded every result, "
    "then compared it against the figure printed on the box.",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief.",
]

# A query whose right answer is known, so reranking is measured for speed and
# checked for sense in the same call. Document 2 is the one that answers it.
RERANK_QUERY = "How many requests can the NPU serve at the same time?"
RERANK_DOCS = [
    "The device draws under twenty watts at the wall while it is idle.",
    "Model weights are stored on the internal drive and loaded on demand.",
    "The runtime serves one inference at a time; further callers are queued "
    "rather than batched, so concurrency does not raise total throughput.",
    "Wi-Fi and USB both reach the gateway, which listens on port 80.",
]
RERANK_ANSWER = 2

# The digits printed on the generated OCR page. Eight of them, because a short
# run is easy to read back and a wrong digit is obvious.
OCR_DIGITS = "20260919"

MUSIC_PROMPT = "a slow acoustic guitar figure in a minor key, no vocals"

# Deterministic filler so prompt lengths are repeatable across runs.
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Engineers measured the throughput carefully and recorded every result. ")

# The suite talks to chat completions, so these are the model types it can
# actually exercise. Everything else on the box is listed but skipped, with the
# reason shown, rather than silently dropped.


UUID_RE = re.compile(
    rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _tiinyos_keys():
    """Every key-shaped string in the TiinyOS app's own storage, most used first.

    Both file types, because LevelDB writes the newest value to the .log and
    only later compacts it into a .ldb, so looking at one of them finds a key
    that is either stale or missing depending on which half you picked. Ranking
    by how often a UUID appears puts the key the app is actually using in front
    of the request ids and session ids that share its shape.
    """
    home = pathlib.Path.home()
    files = sorted(glob.glob(str(
        home / "Library/Application Support/TiinyOS/Local Storage/leveldb/*.ldb")))
    files += sorted(glob.glob(str(
        home / "Library/Application Support/TiinyOS/Local Storage/leveldb/*.log")))
    if not files:
        return []
    # Read here rather than through grep: the files are a few megabytes, the
    # pattern is one regex, and a subprocess is one more thing to be missing.
    # On Windows grep is missing, and this whole function is a macOS path that
    # never has files to read there, but a benchmark should not depend on that
    # to avoid raising FileNotFoundError.
    counts = {}
    for f in files:
        try:
            blob = pathlib.Path(f).read_bytes()
        except OSError:
            continue
        for h in UUID_RE.findall(blob):
            h = h.decode("ascii")
            counts[h] = counts.get(h, 0) + 1
    return sorted(counts, key=lambda c: -counts[c])


def key():
    """TIINY_KEY if you set it. Otherwise the farm's, then the saved one, then
    whatever TiinyOS on this Mac is using.

    The scrape is last on purpose and is a convenience for the machine running
    TiinyOS and nothing more: it reads the app's own local storage and keeps the
    first candidate the device accepts.
    """
    global KEY_SOURCE
    env = os.environ.get("TIINY_KEY", "").strip()
    if env:
        KEY_SOURCE = "TIINY_KEY"
        return env
    farm = (_farm_device().get("key") or "").strip()
    if farm:
        KEY_SOURCE = str(FARM_DEVICE)
        return farm
    saved = (_config().get("key") or "").strip()
    if saved:
        KEY_SOURCE = str(CONFIG)
        return saved
    for c in _tiinyos_keys():
        if "_error" not in api(gw("/api/v1/models/running"), c, timeout=20):
            KEY_SOURCE = "TiinyOS local storage"
            return c
    sys.exit("No API key. Set TIINY_KEY, paste one into the web UI, "
             "or run this on the Mac running TiinyOS.")


class _StayOnBox(urllib.request.HTTPRedirectHandler):
    """Follow a redirect back to the address the request was made to.

    nginx on the device answers /api/v1/models with a 301 to the absolute URL
    http://p8800.api.tiiny/api/v1/models/ - a vhost name that resolves only on
    a Mac running the TiinyOS app, and resolves there to a proxy that answers
    502. Left alone, urllib takes that redirect off the box and a plain model
    listing comes back as a gateway error. So the redirect is followed, but the
    host and port are put back to the ones we were talking to and the Host
    header we set is carried along.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = urllib.parse.urlsplit(newurl)
        old = urllib.parse.urlsplit(req.full_url)
        if new.netloc and new.netloc != old.netloc:
            newurl = urllib.parse.urlunsplit(
                (old.scheme, old.netloc, new.path, new.query, ""))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


OPENER = urllib.request.build_opener(_StayOnBox())


def _request(target, tok, body, timeout, method, read, raw=None, ctype=None):
    """One call to one service, over whichever transport reaches it.

    The service's own port is tried first. A refused connection, and only a
    refused connection, moves that service onto the port 80 vhost for the rest
    of the run. An HTTP error still tells us the transport was right, so it is
    recorded before the error is handed back.

    raw and ctype are for the one endpoint that does not take JSON: ASR wants a
    multipart upload. They are threaded through here rather than given their own
    copy of the walk above, because a second copy is a second thing to keep in
    step with the transport rules.
    """
    err = None
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    sent = ctype or ("application/json" if body else None)
    for url, extra, mode in _attempts(target):
        req = urllib.request.Request(
            # POST is decided by whether a body was offered, not by whether it
            # encoded to any bytes: body={} is a real POST with nothing in it,
            # which is exactly what the model start route is sent.
            url, method=method or ("POST" if (body is not None or raw is not None)
                                   else "GET"),
            data=data,
            headers={"Authorization": f"Bearer {tok}", **extra,
                     **({"Content-Type": sent} if sent else {})})
        try:
            with OPENER.open(req, timeout=timeout) as r:
                _mark(target, mode)
                return r.read() if read else json.load(r)
        except urllib.error.HTTPError as e:
            _mark(target, mode)
            # The device says why in the body, and flattening that to a status
            # line threw away the only thing that tells a missing model apart
            # from a shape this app guessed wrong.
            try:
                raw = e.read()[:MAX_CAPTURE]
                text = raw.decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                text = ""
            return {"_error": str(e)[:140], "_status": e.code, "_body": text,
                    "_ctype": e.headers.get("Content-Type") if e.headers else None}
        except Exception as e:  # noqa: BLE001
            err = e
            if mode == "direct" and _refused(e) and not isinstance(target, str):
                _mark(target, "vhost")
                continue
            break
    return {"_error": str(err)[:140]}


def api(target, tok, body=None, timeout=900, method=None):
    return _request(target, tok, body, timeout, method, read=False)


def api_raw(target, tok, body=None, timeout=300):
    """Same as api() but for endpoints that hand back a file, not JSON."""
    return _request(target, tok, body, timeout, None, read=True)


def _multipart(fields, files):
    """Encode a form the way an OpenAI-shaped upload endpoint expects it.

    fields is {name: str}; files is {name: (filename, content type, bytes)}.
    Small enough to write out rather than reach for a dependency, and the
    boundary is fixed because nothing here is adversarial and a fixed one keeps
    a captured request comparable between runs.
    """
    bound = "----tiinybench7f3c9a21"
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{bound}\r\nContent-Disposition: form-data; name=\"{name}\""
                f"\r\n\r\n{value}\r\n").encode()
    for name, (filename, ctype, blob) in files.items():
        out += (f"--{bound}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        out += blob + b"\r\n"
    out += f"--{bound}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={bound}"


def api_form(target, tok, fields, files, timeout=300):
    """POST a multipart form and read JSON back. ASR is the only caller."""
    raw, ctype = _multipart(fields, files)
    return _request(target, tok, None, timeout, "POST", read=False,
                    raw=raw, ctype=ctype)


class DeviceError(Exception):
    """The device could not be reached, or answered with something unusable."""


def open_call(target, tok, body=None, timeout=900, extra=None):
    """(response, status) for one call, over whichever transport reaches it.

    api() flattens every failure into {"_error": ...} with no status and reads
    the whole body, which is right for a benchmark row and wrong for the two
    callers below: one has to tell the device's own refusal from a transport
    failure, and the other has to read a stream as it arrives rather than after
    it ends. An HTTP error comes back as the response object, because
    HTTPError is one and the device's sentence is inside it. Only a transport
    failure raises.

    The port-then-vhost walk is _attempts and _mark, the same as _request: a
    refused connection, and only a refused connection, moves the service onto
    the port 80 vhost for the rest of the run.
    """
    err = None
    for url, hdrs, mode in _attempts(target):
        req = urllib.request.Request(
            url, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body else None,
            headers={"Authorization": f"Bearer {tok}", **hdrs, **(extra or {}),
                     **({"Content-Type": "application/json"} if body else {})})
        try:
            resp = OPENER.open(req, timeout=timeout)
            _mark(target, mode)
            return resp, resp.status
        except urllib.error.HTTPError as exc:
            _mark(target, mode)
            return exc, exc.code
        except Exception as exc:  # noqa: BLE001
            err = exc
            if mode == "direct" and _refused(exc) and not isinstance(target, str):
                _mark(target, "vhost")
                continue
            break
    raise DeviceError(str(err)[:200])


def telemetry(tok):
    s = api(mgmt("/api/v1/sys/status"), tok, timeout=20)
    n = (s.get("npus") or [{}])[0]
    cpu = s.get("cpu") or {}
    return {
        "npu_util_pct": n.get("utilization_percent"),
        "npu_mem_used_mb": n.get("memory_used_mb"),
        "npu_mem_total_mb": n.get("memory_total_mb"),
        "cpu_total_pct": cpu.get("total_percent"),
    }


def derive_stats(timings, usage, wall_s=None):
    """The numbers a completion reports about itself, from the gateway's own blocks.

    Copied verbatim from AINode Pocket 0.1.3 (pocket/bench.py, derive_stats),
    which is the sibling app on the same hardware. Both apps read the same two
    gateway blocks, so a figure in TiinyBench's chat bar, a figure in a saved
    TiinyBench result and a figure in Pocket's chat bar all mean the same thing.
    Two copies of this arithmetic would drift the first time the gateway renamed
    a field; keep them one function apart or not at all.

    `timings` and `usage` are the blocks the gateway sends: at the top level of a
    non-streamed completion, and in the final chunk of a stream asked for with
    stream_options.include_usage. Nothing here is computed from a clock on this
    side, which is why the chat route adds its own measured ttft_ms on top rather
    than replacing ttft_s: ttft_s is prefill plus one token's decode time, the
    only answer available when the whole reply arrives at once.
    """
    timings = timings or {}
    usage = usage or {}
    stats = {
        "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", 0)),
        "out_tokens": usage.get("completion_tokens", timings.get("predicted_n", 0)),
        "prefill_tok_s": round(timings.get("prompt_per_second") or 0, 2),
        "decode_tok_s": round(timings.get("predicted_per_second") or 0, 2),
        "prefill_ms": round(timings.get("prompt_ms") or 0, 1),
        "ttft_s": round((timings.get("prompt_ms") or 0) / 1000
                        + (timings.get("predicted_per_token_ms") or 0) / 1000, 3),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }
    if wall_s is not None:
        stats["wall_s"] = round(wall_s, 3)
    return stats


def chat_stats(timings, usage, total_ms, ttft_ms, finish_reason, device, model):
    """The one object the Chat page reads its numbers out of.

    Copied verbatim from AINode Pocket 0.1.3 (pocket/server.py, chat_stats).

    Everything the device can measure comes from the device's own timings and
    usage blocks through the benchmark's derivation, so a number in the chat bar
    and the same number in a saved benchmark mean the same thing. Only the two
    wall clock figures belong to this machine: when the first token arrived and
    how long the whole request took, neither of which the device can see.

    A block the device never sent reports null here, not zero. The benchmark's
    derivation answers a missing block with zeroes because a saved row wants a
    number in every column, but on this page a zero is a claim: a stream that
    died at the 220 second cap never reaches the chunk carrying these blocks,
    and "out 0" beside a turn that really streamed four hundred tokens is the
    one kind of lie a page about trustworthy numbers cannot tell. The page
    prints a dash for null, which says the device did not report it.
    """
    derived = derive_stats(timings, usage)
    measured, counted = bool(timings), bool(usage) or bool(timings)
    if ttft_ms is None and measured:
        # Nothing streamed, so there was no first token to time here. The
        # device's own answer is prefill plus one token of decode, which is what
        # the benchmark reports for the same request.
        ttft_ms = round(derived["ttft_s"] * 1000, 1)
    return {"ttft_ms": ttft_ms,
            "prefill_ms": derived["prefill_ms"] if measured else None,
            "decode_tok_s": derived["decode_tok_s"] if measured else None,
            "prefill_tok_s": derived["prefill_tok_s"] if measured else None,
            "prompt_tokens": derived["prompt_tokens"] if counted else None,
            "out_tokens": derived["out_tokens"] if counted else None,
            "cached_tokens": derived["cached_tokens"] if usage else None,
            "total_ms": round(total_ms, 1),
            "finish_reason": finish_reason,
            "device": device,
            "model": model}


# What the suite sends, and does not send. Named here so a result file can
# quote it: a default changing between releases would otherwise be invisible
# in the file and would read as the hardware getting slower.
SAMPLING = {
    "temperature": None, "top_p": None, "top_k": None, "seed": None,
    "note": "none are sent, so the gateway's own defaults for the model apply; "
            "reasoning is switched with chat_template_kwargs.enable_thinking",
}


def chat(tok, model, prompt, max_tokens, thinking=False):
    body = {"model": model, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    v = api(gw("/v1/chat/completions"), tok, body)
    wall = time.time() - t0
    if "_error" in v:
        # The status, the body and the content type, not just the one-line
        # error. Reading the body is what turned six dead ends in the
        # 2026-09-19 sweep into four diagnoses and two measurements, and the
        # four chat tests were the only ones that could not do it.
        return {"error": v["_error"], "wall_s": round(wall, 2), "raw": v}
    # The suite and the Chat page read the same derivation. This used to be a
    # second copy of the same arithmetic sitting right here.
    return derive_stats(v.get("timings"), v.get("usage"), wall)


# ------------------------------------------------------------------ catalog
def catalog(tok):
    """Everything installed on the box, from the box. There is no hand-kept
    model list in this repo on purpose - it would be wrong within a week."""
    d = api(gw("/api/v1/models/"), tok, timeout=60)
    if "_error" in d:
        sys.exit(f"could not read the model catalog: {d['_error']}")
    out = []
    for m in d.get("data", []):
        row = {k: m.get(k) for k in
               ("id", "display_name", "params", "type", "npu_usage",
                "total_size", "thinking", "version", "status")}
        # At least one model ships its params field with a trailing newline,
        # which turns any table built from this into a mess.
        for k, v in row.items():
            if isinstance(v, str):
                row[k] = " ".join(v.split())
        out.append(row)
    return sorted(out, key=lambda m: -(m.get("total_size") or 0))


def running(tok):
    return (api(gw("/api/v1/models/running"), tok, timeout=30)
            .get("running") or [])


def npu_free(tok):
    s = api(gw("/api/v1/models/npu/status"), tok, timeout=30)
    return s.get("npu_available"), s.get("npu_total")


def npu_status(tok):
    """The whole NPU budget reply, not just the two totals.

    The Chat page's rail wants the per-model rows as well, and reading this is
    also what moves a pending load along on real firmware: a start that does not
    fit is accepted, sits here as "loading", and then vanishes. This endpoint is
    the only place that can be seen.
    """
    s = api(gw("/api/v1/models/npu/status"), tok, timeout=30)
    return {} if "_error" in s else s


def catalog_raw(tok):
    """Installed models with every field the device sent, not the seven columns.

    catalog() keeps what a benchmark table needs. The model card beside the
    conversation wants input, output, desc and capabilities too, and those are
    thrown away up there.
    """
    d = api(gw("/api/v1/models/"), tok, timeout=60)
    if "_error" in d:
        raise DeviceError(d["_error"])
    return [m for m in (d.get("data") or []) if isinstance(m, dict)]


def running_detail(tok):
    """What is loaded right now, with each instance's units and status.

    running() answers the list of model ids, which is all the suite needs. The
    rail also has to tell a model that is up from one that is still coming up.
    """
    d = api(gw("/api/v1/models/running"), tok, timeout=30)
    if "_error" in d:
        raise DeviceError(d["_error"])
    return d


# What /v1/chat/completions will actually serve. Distinct from CHAT_TYPES, which
# is every class the benchmark has a test for and covers the image, speech,
# embedding, transcription, OCR, music and reranking models as well, none of
# which answer a chat completion. "main" is the device's own word for the chat
# runtime;
# an older catalogue row carries no capabilities at all, so the type is the
# fallback rather than a refusal.
CHAT_CAPABILITY = "main"
CHAT_MODEL_TYPES = frozenset(["text generation", "image-text-to-text"])

# Plain English for a refusal, so the message says what the model is instead of
# echoing a label out of a catalogue at somebody.
TYPE_PHRASES = {
    "text-to-speech": "a text-to-speech model",
    "asr": "a speech recognition model",
    "text embedding": "an embedding model",
    "text reranking": "a reranking model",
    "text-to-image": "an image generation model",
    "image-to-text": "an OCR model",
    "music generation": "a music generation model",
}


def capability_list(entry):
    """The capabilities a model row claims, lowercased, or an empty list."""
    if not isinstance(entry, dict):
        return []
    raw = entry.get("capabilities")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip().lower() for item in raw if str(item).strip()]


def can_chat(type_label=None, capabilities=None):
    """Whether /v1/chat/completions can serve this model.

    Capabilities win when the device sent any. An empty list is not an answer,
    so it falls through to the type rather than refusing everything on firmware
    that omits the field.
    """
    if capabilities:
        return CHAT_CAPABILITY in [str(item).strip().lower() for item in capabilities]
    return str(type_label or "").strip().lower() in CHAT_MODEL_TYPES


def type_phrase(type_label):
    """"a text-to-speech model" for a type there is a phrase for, else "".

    An empty answer is deliberate: the caller says "is not a chat model" rather
    than inventing a description of a type nobody has seen yet.
    """
    label = str(type_label or "").strip()
    if not label:
        return ""
    return TYPE_PHRASES.get(label.lower(), "")


BUSY_CODE = 150004
# A 150004 means somebody else has the device: another app, or the box doing its
# own work. Backing off briefly is cheap and usually wins.
BUSY_RETRIES = 4
BUSY_BACKOFF = 1.5


def chat_once(tok, body, timeout=240):
    """(status, payload) for one non-streamed completion, errors included.

    The status is kept because the route that calls this has to tell the
    device's own refusal, which carries a sentence worth showing, from a
    transport failure, which carries a different one.
    """
    resp, status = open_call(gw("/v1/chat/completions"), tok, body, timeout)
    with resp:
        try:
            payload = json.load(resp)
        except Exception:  # noqa: BLE001
            payload = {}
    # An in-band busy: HTTP 200 carrying a code and no choices.
    if status == 200 and isinstance(payload, dict) \
            and payload.get("code") and "choices" not in payload:
        return 503, payload
    return status, payload


def chat_lines(tok, body, timeout=600):
    """Yield the device's own SSE lines for a streamed completion."""
    resp, status = open_call(gw("/v1/chat/completions"), tok,
                             dict(body, stream=True), timeout,
                             extra={"Accept": "text/event-stream"})
    if status != 200:
        with resp:
            detail = resp.read().decode("utf-8", "replace")[:160]
        raise DeviceError("the device answered HTTP %d%s"
                          % (status, (": " + detail) if detail.strip() else ""))
    with resp:
        try:
            for raw in resp:
                yield raw.decode("utf-8", "replace").rstrip("\n")
        except Exception as exc:  # noqa: BLE001
            # The gateway closes a single request at about 220 seconds. The
            # tokens already relayed are real, so this is not a failure of the
            # whole turn and the caller says so rather than swallowing it.
            raise DeviceError(str(exc)[:200])


def load(tok, model, poll_s=420):
    """Start a model and wait for it to actually answer.

    The runtime is listed in /running before it is settled, and a call made in
    that window comes back 502. Two seconds of patience here is cheaper than a
    failed benchmark that looks like a slow model."""
    if model in running(tok):
        return True
    enc = urllib.parse.quote(model, safe="")
    say(f"    loading {model} ...")
    t0 = time.time()
    api(gw(f"/api/v1/models/{enc}/start"), tok, body={}, timeout=poll_s)
    while time.time() - t0 < poll_s:
        if model in running(tok):
            time.sleep(2.0)
            say(f"    up in {time.time() - t0:.0f}s")
            return True
        time.sleep(4.0)
    say("    TIMED OUT")
    return False


def unload(tok, model):
    enc = urllib.parse.quote(model, safe="")
    api(gw(f"/api/v1/models/{enc}/stop"), tok, body={}, timeout=180)
    time.sleep(1.5)


def unload_all(tok):
    """Clear the NPU. The device's own app hides this three levels deep inside
    Agents, which is most of the reason this page exists."""
    return api(gw("/api/v1/models/unload_all"), tok, body={}, timeout=180)


def online(tok):
    """The vendor catalogue: everything downloadable, installed or not.

    Read off the box on every call. There is no pinned or hand-kept list in
    this repo, which is the point: a catalogue baked in here would be wrong
    within a week.

    Distinct from catalog(), which is only what is already on the box. A
    benchmark that cannot fetch a model it has not got has a dead end in it,
    and "download it, then measure it" is the whole point of closing this loop.

    A box that does not answer raises. This used to return an empty list, which
    the page then printed as "0 in the catalogue": a silent zero that reads as
    an empty store rather than as a question nobody answered.
    """
    d = api(gw("/api/v1/models/online_models"), tok, timeout=90)
    if isinstance(d, dict) and "_error" in d:
        raise DeviceError(d["_error"])
    rows = d if isinstance(d, list) else (d.get("data") or d.get("models") or [])
    out = []
    for m in rows:
        if not isinstance(m, dict):
            continue
        out.append({
            "id": m.get("model_id") or m.get("fullname") or m.get("id"),
            "name": m.get("name") or m.get("model_id"),
            "type": " ".join(str(m.get("type") or "").split()),
            "params": " ".join(str(m.get("params") or "").split()),
            "npu_usage": m.get("npu_usage"),
            "status": m.get("status"),
            "progress": m.get("progress"),
        })
    return [m for m in out if m["id"]]


def download(tok, model):
    """Start a download. Returns at once; watch it with download_progress."""
    enc = urllib.parse.quote(model, safe="")
    return api(gw(f"/api/v1/models/{enc}/download"), tok,
               body={}, timeout=120)


def download_progress(tok, model):
    enc = urllib.parse.quote(model, safe="")
    d = api(gw(f"/api/v1/models/{enc}/get_progress"), tok, timeout=30)
    return {} if "_error" in d else d


def delete_model(tok, model):
    enc = urllib.parse.quote(model, safe="")
    return api(gw(f"/api/v1/models/{enc}"), tok,
               timeout=180, method="DELETE")


def storage(tok):
    """Disk on the device."""
    d = api(gw("/api/v1/sys/storage"), tok, timeout=30)
    return {} if "_error" in d else d


# ---------------------------------------------------------------- tests
def t_prefill(tok, model):
    """How fast does it ingest a document? Prefill is what long context costs."""
    say("\n  PREFILL SCALING  (document ingestion)")
    say(f"    {'approx tokens':>14} {'prefill tok/s':>15} {'prefill ms':>12}")
    rows = []
    for reps in (2, 12, 60, 240):
        prompt = (FILLER * reps) + "\n\nReply with the single word: ok"
        r = chat(tok, model, prompt, 4)
        if "error" in r:
            capture("prefill", gw("/v1/chat/completions"), r.get("raw"))
            say(f"    {reps:>14} FAILED {r['error'][:50]}"); continue
        rows.append(r)
        say(f"    {r['prompt_tokens']:>14} {r['prefill_tok_s']:>15.2f} {r['prefill_ms']:>12.1f}")
    return rows


def t_sustained(tok, model, total=1500):
    """Does throughput hold, or does it sag as the box heats and KV grows?

    Utilisation is sampled on a thread WHILE the generation runs. Reading it
    once the request returns only ever catches the box going idle again, which
    is how an early version of this suite reported 0% NPU under load."""
    say(f"\n  SUSTAINED GENERATION  ({total} tokens, one unbroken request)")
    before = telemetry(tok)
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.wait(1.0):
            t = telemetry(tok)
            if t.get("npu_util_pct") is not None:
                samples.append(t)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        r = chat(tok, model,
                 "Write a detailed technical explanation of how speculative decoding works "
                 "in large language model inference. Cover the draft model, verification, "
                 "acceptance rates, and why throughput varies with content.", total)
    finally:
        stop.set()
        th.join(timeout=5)
    after = telemetry(tok)
    if "error" in r:
        capture("sustained", gw("/v1/chat/completions"), r.get("raw"))
        say(f"    FAILED {r['error'][:70]}"); return None

    util = [s["npu_util_pct"] for s in samples]
    mem = [s.get("npu_mem_used_mb") or 0 for s in samples]
    during = {
        "samples": len(samples),
        "npu_util_peak": max(util) if util else None,
        "npu_util_median": round(statistics.median(util), 1) if util else None,
        "npu_mem_peak_mb": max(mem) if mem else None,
        "npu_mem_total_mb": (samples[0].get("npu_mem_total_mb") if samples
                             else after.get("npu_mem_total_mb")),
    }
    say(f"    generated {r['out_tokens']} tokens in {r['wall_s']}s at {r['decode_tok_s']} tok/s")
    say(f"    NPU util while running: median {during['npu_util_median']}% "
          f"peak {during['npu_util_peak']}%  ({during['samples']} samples)")
    say(f"    NPU mem peak {during['npu_mem_peak_mb']}/{during['npu_mem_total_mb']} MB")
    return {"run": r, "telemetry_before": before, "telemetry_after": after,
            "during": during}


def t_concurrency(tok, model, levels=(1, 2, 4, 8), per=160):
    """Aggregate throughput as more people use the box at the same time."""
    say(f"\n  CONCURRENCY  ({per} tokens per request)")
    say(f"    {'parallel':>9} {'aggregate tok/s':>17} {'per-stream':>12} {'wall s':>9}")
    rows = []
    for n in levels:
        res, errs = [], []
        def worker(i):
            r = chat(tok, model,
                     f"Explain concept number {i}: why memory bandwidth limits "
                     f"token generation on edge devices. Be specific.", per)
            (errs if "error" in r else res).append(r)
        t0 = time.time()
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.time() - t0
        if not res:
            # "all failed" on its own does not say whether the box was busy,
            # out of memory or refusing the request, and at eight parallel
            # streams those are three different findings.
            if errs:
                capture("concurrency", gw("/v1/chat/completions"),
                        errs[0].get("raw"),
                        f"every one of {n} parallel requests failed")
            say(f"    {n:>9} all failed"); continue
        agg = sum(x["out_tokens"] for x in res) / wall
        per_stream = statistics.median(x["decode_tok_s"] for x in res)
        rows.append({"parallel": n, "aggregate_tok_s": round(agg, 2),
                     "per_stream_tok_s": round(per_stream, 2),
                     "wall_s": round(wall, 2), "ok": len(res), "failed": len(errs)})
        say(f"    {n:>9} {agg:>17.2f} {per_stream:>12.2f} {wall:>9.2f}")
    return rows


def t_thinking(tok, model):
    """A reasoning model's hidden tokens are not free. What do they cost?"""
    say("\n  REASONING COST  (same prompt, thinking on vs off)")
    q = ("A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph. "
         "When does the second catch the first?")
    out = {}
    for name, flag in (("off", False), ("on", True)):
        r = chat(tok, model, q, 700, thinking=flag)
        if "error" in r:
            capture("thinking", gw("/v1/chat/completions"), r.get("raw"))
            say(f"    thinking {name:<3} FAILED"); continue
        out[name] = r
        say(f"    thinking {name:<3} {r['out_tokens']:>4} tokens  "
              f"{r['wall_s']:>6.2f}s  {r['decode_tok_s']:>6.2f} tok/s")
    if "on" in out and "off" in out and out["off"]["wall_s"]:
        say(f"    -> reasoning costs {out['on']['wall_s'] / out['off']['wall_s']:.1f}x the wall time")
    return out


def t_image(tok, model):
    """Seconds per 512x512 plate. The only figure an image model is judged by
    on this box, because 512 is the only size the firmware will render."""
    say("\n  IMAGE GENERATION  (512x512, 8 steps)")
    rows = []
    for i, prompt in enumerate(IMAGE_PROMPTS):
        t0 = time.time()
        raw = api_raw(gw("/v1/image/generate"), tok,
                      {"model": model, "prompt": prompt, "negative_prompt": "",
                       "width": 512, "height": 512, "seed": 1000 + i, "steps": 8},
                      timeout=300)
        wall = time.time() - t0
        if not isinstance(raw, (bytes, bytearray)):
            if not_loaded(raw):
                say("    no image model is resident; nothing to measure")
                return not_measured("no Text-to-Image model was resident on the device")
            capture("image", gw("/v1/image/generate"), raw)
            say(f"    image {i+1} FAILED {str(raw)[:60]}"); continue
        rows.append({"wall_s": round(wall, 2), "bytes": len(raw), "seed": 1000 + i})
        say(f"    image {i+1}  {wall:6.2f}s  {len(raw)//1024:>5} KB")
    if not rows:
        return None
    med = statistics.median(r["wall_s"] for r in rows)
    say(f"    median {med:.2f}s per plate")
    return {"runs": rows, "s_per_image": round(med, 2)}


def _rejected_field(v):
    """The field name a 400 says it would not accept, if it says one.

    The music service does not agree with itself across models: Foundation-1
    takes a duration, SongGeneration-v2-large answers 400 with "Extra inputs
    are not permitted in request: duration". Rather than keep a table of which
    model takes which key, the request is sent once and the refusal is read.
    """
    if not isinstance(v, dict) or v.get("_status") != 400:
        return None
    blob = str(v.get("_body") or "")
    m = re.search(r"[Ee]xtra inputs are not permitted in request:\s*([A-Za-z_][A-Za-z0-9_]*)",
                  blob)
    return m.group(1) if m else None


# Dropping a field the box names is safe for decoration and dangerous for
# content. SongGeneration answered "Extra inputs are not permitted in request:
# prompt" and then, once the prompt was gone, "config.lyrics, config.caption,
# config.instruction, or prompt field is required". Two validators disagreeing
# with each other is not something to resolve by deleting the request.
NEVER_DROP = frozenset({"model", "prompt"})


def _required_fields(v):
    """The fields a refusal says are required, in the order it lists them."""
    if not isinstance(v, dict):
        return []
    m = re.search(r"([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*"
                  r"(?:\s*,?\s*or\s+[A-Za-z_][\w.]*)?)\s+(?:field\s+)?is required",
                  str(v.get("_body") or ""))
    if not m:
        return []
    parts = re.split(r"\s*,\s*|\s+or\s+", m.group(1))
    return [x for x in (re.sub(r"^or\s+", "", q).strip() for q in parts) if x]


def _set_path(body, dotted, value):
    """Set a possibly-nested field named the way the device names it."""
    node = body
    parts = dotted.split(".")
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value
    return body


def _as_wav(raw):
    """A WAV out of whatever shape the device chose to answer in.

    Three shapes have come back from this box: the raw RIFF body, and a JSON
    envelope carrying the same bytes base64-encoded under one of several key
    names, and a job to poll. This handles the first two. Written because the
    music service answered {"success": true, "audio_data": "UklGR..."} and the
    benchmark recorded a working model as FAILED for a whole sweep.
    """
    if isinstance(raw, (bytes, bytearray)) and raw[:4] == b"RIFF":
        return bytes(raw)
    env = raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            env = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return None
    if not isinstance(env, dict):
        return None
    for key in ("audio_data", "audio_base64", "audio", "b64_json", "data",
                "wav", "content"):
        v = env.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            v = v[0].get("audio_data") or v[0].get("b64_json") or v[0].get("audio")
        if not isinstance(v, str) or len(v) < 32:
            continue
        if v.startswith("data:"):
            v = v.split(",", 1)[-1]
        try:
            blob = base64.b64decode(v, validate=False)
        except Exception:  # noqa: BLE001
            continue
        if blob[:4] == b"RIFF":
            return blob
    return None


def _wav_seconds(raw):
    """Duration out of a RIFF header, so the real-time factor is measured
    rather than guessed from a character count."""
    try:
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return None
        i, rate, ch, bits = 12, None, None, None
        while i + 8 <= len(raw):
            cid = raw[i:i+4]
            size = int.from_bytes(raw[i+4:i+8], "little")
            body = raw[i+8:i+8+size]
            if cid == b"fmt ":
                ch = int.from_bytes(body[2:4], "little")
                rate = int.from_bytes(body[4:8], "little")
                bits = int.from_bytes(body[14:16], "little")
            elif cid == b"data" and rate and ch and bits:
                return size / (rate * ch * max(1, bits // 8))
            i += 8 + size + (size & 1)
    except Exception:  # noqa: BLE001
        return None
    return None


# The speech route defaults to a custom-voice mode, and two of the four
# text-to-speech models on this box do not implement it: they answer 500
# "custom_voice is not supported by this model". Naming a voice instead does
# not help. Story Lantern, which drives this route in production, found the
# CustomVoice variant needs no voice field and its siblings reject all 35
# known speaker names, and the 2026-09-19 sweep agrees: CustomVoice measured
# 1.58x real time with the plain body, Base and VoiceDesign measured nothing.
# So this does not guess at voice names. It says which wall it hit.

def _voice_mode_refused(v):
    """True when the device is refusing the voice mode, not the text."""
    if not isinstance(v, dict):
        return False
    blob = (str(v.get("_body") or "") + str(v.get("_error") or "")).lower()
    return (("voice" in blob or "speaker" in blob)
            and ("not supported" in blob or "unsupported" in blob))


def _offered_voices(v):
    """The speaker names a refusal lists, when it lists any.

    Supertone answers 500 "Unsupported speaker: serena. Supported speakers:
    ['F1', 'F2', ...]", which is the device handing over the answer. Reading
    it is not guessing; the Qwen variants name no alternatives and get none
    invented for them.
    """
    if not isinstance(v, dict):
        return []
    blob = str(v.get("_body") or "")
    m = re.search(r"[Ss]upported (?:speakers?|voices?)[^\[]*\[([^\]]*)\]", blob)
    if not m:
        return []
    return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]


def t_speech(tok, model):
    """Real-time factor: seconds of audio produced per second of wall clock.
    Above 1.0 means it can talk faster than a person listens, which is the only
    threshold that matters for anything conversational."""
    say("\n  SPEECH  (real-time factor)")
    rows, voice = [], None
    for i, text in enumerate(SPEECH_TEXTS):
        body = {"model": model, "input": text, "response_format": "wav"}
        if voice:
            body["voice"] = voice
        t0 = time.time()
        raw = api_raw(gw("/v1/audio/speech"), tok, body, timeout=300)
        if _voice_mode_refused(raw) and voice is None:
            offered = _offered_voices(raw)
            if offered:
                voice = offered[0]
                say(f"    this model names its speakers; using {voice} "
                    f"of {len(offered)}")
                t0 = time.time()
                raw = api_raw(gw("/v1/audio/speech"), tok,
                              dict(body, voice=voice), timeout=300)
            else:
                capture("speech", gw("/v1/audio/speech"), raw)
                say("    this model does not implement the voice mode the "
                    "speech route defaults to, and names no alternative")
                return not_measured(
                    "the speech route defaults to a custom-voice mode this "
                    "model does not implement, and it answered 500 rather "
                    "than audio")
        wall = time.time() - t0
        wav = _as_wav(raw)
        if not wav:
            capture("speech", gw("/v1/audio/speech"), raw,
                    "no WAV came back, in any envelope this knows")
            say(f"    clip {i+1} FAILED {str(raw)[:60]}"); continue
        secs = _wav_seconds(wav)
        rtf = round(secs / wall, 2) if secs and wall else None
        rows.append({"chars": len(text), "wall_s": round(wall, 2),
                     "audio_s": round(secs, 2) if secs else None, "rtf": rtf})
        say(f"    clip {i+1}  {len(text):>4} chars  {wall:6.2f}s  "
            + (f"{secs:5.1f}s audio  {rtf}x real time" if secs else "(duration unknown)"))
    good = [r["rtf"] for r in rows if r.get("rtf")]
    if good:
        say(f"    median {statistics.median(good):.2f}x real time")
    if not rows:
        return None
    return {"runs": rows, "voice": voice,
            "rtf": round(statistics.median(good), 2) if good else None}


def t_embed(tok, model):
    """Embeddings per second, at one, eight and thirty-two at a time."""
    say("\n  EMBEDDINGS  (throughput)")
    rows = []
    for n in (1, 8, 32):
        batch = [FILLER[:180] + f" item {i}" for i in range(n)]
        t0 = time.time()
        v = api(gw("/v1/embeddings"), tok,
                {"model": model, "input": batch}, timeout=180)
        wall = time.time() - t0
        if "_error" in v:
            if not_loaded(v):
                say("    no embedding model is resident; nothing to measure")
                return not_measured("no Text Embedding model was resident on the device")
            capture("embed", gw("/v1/embeddings"), v)
            say(f"    batch {n:>3} FAILED {v['_error'][:55]}"); continue
        data = v.get("data") or []
        dim = len((data[0] or {}).get("embedding") or []) if data else 0
        rate = round(len(data) / wall, 1) if wall else 0
        rows.append({"batch": n, "wall_s": round(wall, 3), "returned": len(data),
                     "dim": dim, "per_s": rate})
        say(f"    batch {n:>3}  {wall:6.3f}s  {rate:8.1f} emb/s  dim {dim}")
    if not rows:
        return None
    return {"runs": rows, "emb_per_s": max(r["per_s"] for r in rows),
            "dim": rows[0]["dim"]}


# ====================================================================== #
#  Provenance                                                            #
# ====================================================================== #
# What a number needs carried with it to stay comparable once it leaves this
# machine. The principle: anything that could change the figure, or that a
# stranger would need to know before trusting it.
#
# A benchmark driven from a Mac over a USB cable and one driven from a Windows
# box over Wi-Fi are not the same measurement, and until now nothing in the
# file said which. Neither did anything say what else was resident on the
# device at the time, which on a box with a hundred NPU units and one
# accelerator is the single condition that most decides the answer.
#
# Two rules hold this together.
#
#   Everything is recorded raw and local. The transform that makes a record
#   safe to publish lives in exactly one function, public_view, so the upload
#   path and the report cannot disagree about what is safe to show.
#
#   A field that could not be read is recorded as null with a reason, never
#   omitted and never defaulted. A missing key reads as an oversight; a zero
#   reads as a measurement. "Unavailable, and here is why" reads as neither.

# Bump when the shape changes. A reader in six months needs to know what a
# file promised, and "no number" means a file written before there was one.
ENVELOPE_SCHEMA = 1

# Facts this app cannot obtain without starting another program, which it may
# not do: the farm's archive scanner refuses a tree that can, and that refusal
# is worth more than these fields.
_NO_SUBPROCESS = "not readable without starting a program, which this app may not do"


def _ram_bytes():
    """Physical memory, from the C library's own constants where they exist."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _machine_model():
    """(model, reason it is missing). Linux publishes it in a file; macOS and
    Windows want a program run, so there it stays honestly unknown."""
    for f in ("/sys/devices/virtual/dmi/id/product_name",
              "/sys/firmware/devicetree/base/model"):
        try:
            text = pathlib.Path(f).read_text(encoding="utf-8", errors="replace").strip("\x00 \n")
            if text:
                return text, None
        except OSError:
            continue
    return None, _NO_SUBPROCESS


def host_facts():
    """The machine driving the benchmark, which is half of what it measures."""
    model, why = _machine_model()
    out = {
        "os": platform.system() or None,
        "os_release": platform.release() or None,
        "os_version": (platform.mac_ver()[0] or None
                       if platform.system() == "Darwin" else platform.version() or None),
        "arch": platform.machine() or None,
        "machine_model": model,
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
    }
    if why:
        out["unavailable"] = {"machine_model": why}
    return out


def tool_facts():
    """This app: what version measured, and which tree it was built from."""
    commit, dirty = _git_head()
    return {"bench_version": VERSION, "suite_version": 2,
            "envelope_schema": ENVELOPE_SCHEMA,
            "git_commit": commit, "git_dirty": dirty}


def _git_head():
    """(short sha, working tree dirty) read out of .git by hand.

    Reading the files rather than running git, for the same reason as
    everywhere else here. A packed HEAD and a worktree both resolve; anything
    else gives up rather than guessing. Dirtiness cannot be known without
    running git, so it is None rather than False, which would be a claim.
    """
    root = pathlib.Path(__file__).resolve().parent
    git = root / ".git"
    try:
        if git.is_file():                      # a worktree points elsewhere
            git = pathlib.Path(git.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip())
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            f = git / ref
            if f.exists():
                return f.read_text(encoding="utf-8").strip()[:12], None
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:12], None
            return None, None
        return head[:12], None
    except (OSError, IndexError, ValueError):
        return None, None


def link_rtt_ms(tok, tries=3):
    """Round trip to the device before any measuring starts.

    Cheap, and it tells a reader whether the link was the bottleneck. The
    smallest of a few tries, because the smallest is the one least polluted by
    something else happening at the time.
    """
    best = None
    for _ in range(tries):
        t0 = time.time()
        r = api(mgmt("/api/v1/sys/device_info"), tok, timeout=15)
        if "_error" in r:
            return None
        ms = (time.time() - t0) * 1000
        best = ms if best is None else min(best, ms)
    return round(best, 1) if best is not None else None


def transport_facts(tok):
    """Which wire, and how far away the box was down it."""
    out = dict(where())
    out["rtt_ms"] = link_rtt_ms(tok)
    return out


def device_facts(tok, info=None):
    """What the box says it is, plus the size of its NPU budget."""
    info = info if isinstance(info, dict) else api(
        mgmt("/api/v1/sys/device_info"), tok, timeout=20)
    if "_error" in info:
        info = {}
    free, total = npu_free(tok)
    return {
        "name": info.get("device_name") or info.get("name"),
        "model": info.get("device_model_name") or info.get("model"),
        # The device spells it "sn" in device_info and "serial" elsewhere.
        "serial": info.get("sn") or info.get("serial") or DEVICE.get("serial"),
        "tiiny_os": info.get("tiiny_os"),
        "service_version": info.get("version"),
        "npu_units_total": total or None,
        "ram": info.get("ram"),
        "storage": info.get("storage"),
    }


def conditions(tok, cat=None):
    """What else the box was doing when the measurement started.

    This is the part nobody records and the part that decides everything. One
    accelerator serving one sequence at a time means a model that shared the
    box with something else was not measured on the same box as one that did
    not, and without the resident set at the moment the test began there is no
    way to tell the two apart afterwards.
    """
    cat = cat or {}
    live = running(tok)
    free, total = npu_free(tok)
    resident = [{"model": m, "npu_usage": (cat.get(m) or {}).get("npu_usage")}
                for m in live]
    tel = telemetry(tok)
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "resident": resident,
        "npu_units_used": sum(r["npu_usage"] or 0 for r in resident) or None,
        "npu_units_free": free if total else None,
        "npu_util_pct": tel.get("npu_util_pct"),
        "npu_mem_used_mb": tel.get("npu_mem_used_mb"),
        "cpu_total_pct": tel.get("cpu_total_pct"),
        "device_lock_held_by": _lock_holder(),
        # Recorded as unavailable rather than left out: a later reader has to
        # be able to tell "nobody measured this" from "it measured zero".
        "temperature_c": None,
        "power_w": None,
        "unavailable": {
            "temperature_c": "the device's status API returns null for it",
            "power_w": "the device's status API returns null for it",
        },
    }


# The advisory lock Turnstile takes around a whole sectioned job on this box.
# A run that waited behind another app is not a clean run, and nothing in a
# result file would have shown it.
LOCK_PATHS = ("~/.turnstile/tiiny.lock", "~/.onelane/tiiny.lock")


def _lock_holder():
    """Who held the device lock when this started, or None if nobody did.

    The lock file carries the holder's own description. It is read rather than
    taken: this is a record of the conditions, not a claim on the device, and
    a benchmark that fought for the lock would change the thing it measures.
    """
    for spec in LOCK_PATHS:
        p = pathlib.Path(os.path.expanduser(spec))
        try:
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                blob = json.loads(text)
                return blob.get("owner") or blob.get("holder") or blob.get("app") or text[:120]
            except ValueError:
                return text[:120]
        except OSError:
            continue
    return None


def envelope(tok, cat=None, info=None):
    """Everything that makes a number comparable, in one block."""
    return {
        "schema": ENVELOPE_SCHEMA,
        "tool": tool_facts(),
        "host": host_facts(),
        "transport": transport_facts(tok),
        "device": device_facts(tok, info),
        "conditions": conditions(tok, cat),
    }


# ------------------------------------------------------------ publishing
# One function, so the upload path and the report cannot disagree about what is
# safe to show. Everything above is recorded raw and stays on this machine.

def hash_serial(serial):
    """A stable, non-reversible handle for a device.

    Published runs have to be groupable by box without naming the box. A plain
    sha256 of a serial is reversible by anyone who can enumerate the format, so
    it is salted with a fixed string: this does not have to resist an attacker
    with the salt, it has to stop a serial being read straight out of a
    published file.
    """
    if not serial:
        return None
    return "tiiny-" + hashlib.sha256(
        ("tiinybench/v1/" + str(serial)).encode("utf-8")).hexdigest()[:16]


# The fields that identify a network or a person rather than a measurement,
# named by where they sit and not by what they are called. A blanket rule on
# the name "host" dropped the block describing the machine that drove the
# benchmark, which is a measurement fact and has to survive: one key, two
# meanings, and the blunt version threw away the wrong one.
DROP_PATHS = frozenset({
    ("host",),                              # the device's address
    ("provenance", "transport", "host"),
    ("provenance", "transport", "gateway_vhost"),
    ("provenance", "transport", "candidates"),
    ("provenance", "transport", "config"),
    ("provenance", "device", "name"),       # somebody's name for their box
    ("connection", "host"),                 # the pre-envelope spelling
    ("connection", "gateway_vhost"),
    ("connection", "config"),
    ("connection", "candidates"),
})

# Wherever one of these appears, at any depth, because a serial is a serial.
HASH_KEYS = frozenset({"serial", "sn"})


def public_view(record, account=None):
    """The shareable form of a result record. The only one.

    What it does, and why each:

      a serial becomes a salted hash wherever it appears, so runs can be
      grouped by box without naming one; the device's address, its vhost name
      and somebody's chosen name for it are dropped, because they identify a
      network rather than a measurement; any string that looks like a path
      under a home directory is dropped, since a path carries a username; and
      an account attribution is added when one is given, so a published run
      has an owner by choice rather than by leak.

    Nothing else changes. Every measured number survives untouched, which is
    the point: this makes a record publishable, not smaller. It works on a
    record with no envelope at all, because files written before there was one
    still have to be publishable.
    """
    def clean(node, path=()):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                here = path + (k,)
                if here in DROP_PATHS:
                    continue
                if k in HASH_KEYS:
                    out["device_hash"] = hash_serial(v)
                    continue
                out[k] = clean(v, here)
            return out
        if isinstance(node, list):
            return [clean(v, path) for v in node]
        if isinstance(node, str) and _looks_like_a_home_path(node):
            return None
        return node

    out = clean(json.loads(json.dumps(record)))
    out["published"] = {"account": account, "envelope_schema": ENVELOPE_SCHEMA,
                        "transform": "public_view/1"}
    return out


_HOME_RE = re.compile(r"(^|[\s\"'])(~/|/home/|/Users/|[A-Za-z]:\\\\Users\\\\)")


def _looks_like_a_home_path(text):
    return bool(_HOME_RE.search(text))


# =================================================================== #
#  Two failures that look the same and are not                        #
# =================================================================== #
# A test returning nothing can mean the model was not resident, which is
# ordinary and expected on a box with a hundred NPU units and nine classes of
# model, or it can mean the device answered in a shape this app did not
# anticipate, which is a bug here. Those read identically in a result file as
# a null, and the second one costs a model load to reproduce once the sweep
# has moved on. So they are recorded differently.

MAX_CAPTURE = 2000

# The envelope the gateway returns when no model of the needed class is
# running, measured off live firmware. Nothing else on the device uses it.
NOT_LOADED_TYPE = "service_unavailable"


def not_loaded(v):
    """Whether a failed call failed because no model of that class is up."""
    if not isinstance(v, dict) or "_error" not in v:
        return False
    if v.get("_status") != 503:
        return False
    try:
        err = (json.loads(v.get("_body") or "{}") or {}).get("error") or {}
    except ValueError:
        return False
    return err.get("type") == NOT_LOADED_TYPE or "no suitable model" in str(
        err.get("message", "")).lower()


def not_measured(reason):
    """What a test returns instead of nothing, so the file says which."""
    return {"not_measured": reason}


# First unexpected response per test per run. A large audio or image payload
# would bloat a result file, so it is truncated; the status, the path and the
# content type go with it because those are usually enough on their own.
UNPARSED = {}


def capture(test, path, v, note=""):
    """Record the first response from this test that could not be used.

    Once per test per run: the tenth copy of the same surprise adds nothing
    and the first is what gets read. Kept in the result file and passed
    through public_view on the way out like everything else, because a raw
    device response could carry anything.
    """
    if test in UNPARSED:
        return
    body = v if isinstance(v, (str, bytes)) else json.dumps(v, default=str)
    if isinstance(body, bytes):
        body = "<%d bytes, not text>" % len(body) if b"\x00" in body[:64] \
            else body[:MAX_CAPTURE].decode("utf-8", "replace")
    UNPARSED[test] = {
        "path": str(path),
        "status": (v.get("_status") if isinstance(v, dict) else None),
        "content_type": (v.get("_ctype") if isinstance(v, dict) else None),
        "note": note or "the response could not be read as this test expects",
        "body": body[:MAX_CAPTURE],
        "truncated": len(body) > MAX_CAPTURE,
    }


# ---------------------------------------------------------------- fixtures
# Four classes of model need something to chew on that is not text. All of it
# is built here rather than committed, so the repository stays readable and
# every run gets byte-identical input: a fixture you cannot diff is a fixture
# you cannot trust when a number moves.

def _wav_bytes(seconds, rate=16000):
    """A tone gated on and off four times a second, as a 16-bit mono WAV.

    It is not speech and will not transcribe to anything, which is the point:
    what this measures is how fast the model chews through a known duration of
    audio, and the duration is exact because this wrote the header.
    """
    n = int(seconds * rate)
    period = max(2, rate // 220)
    frames = bytearray()
    for i in range(n):
        on = int(i * 4 / rate) % 2 == 0
        v = (9000 if (i % period) < period / 2 else -9000) if on else 0
        frames += struct.pack("<h", v)
    data = bytes(frames)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


# Seven segments in a 6 by 11 cell. The bars meet at the corners rather than
# stopping short, because a digit drawn with gaps at the corners reads as a pile
# of bars and this page has to be legible to a model that was trained on print.
_SEG_BOX = {"a": (0, 0, 5, 0), "b": (5, 0, 5, 5), "c": (5, 5, 5, 10),
            "d": (0, 10, 5, 10), "e": (0, 5, 0, 10), "f": (0, 0, 0, 5),
            "g": (0, 5, 5, 5)}
_SEG_ON = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
           "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def _png_grey(width, height, rows):
    """Eight-bit greyscale PNG. zlib and struct are the whole toolkit."""
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body)))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def _digits_png(text=OCR_DIGITS, scale=10, margin=30):
    """A white page with black seven-segment digits on it.

    Seven segments rather than a font table: a font is three hundred lines of
    glyph data nobody can check by eye, and digits are enough to tell whether
    the page was read.
    """
    cell_w, cell_h, gap = 6, 11, 3
    grid = [[0] * (len(text) * (cell_w + gap)) for _ in range(cell_h)]
    for i, ch in enumerate(text):
        ox = i * (cell_w + gap)
        for seg in _SEG_ON.get(ch, ""):
            x0, y0, x1, y1 = _SEG_BOX[seg]
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    grid[y][ox + x] = 1
    w = len(grid[0]) * scale + margin * 2
    h = cell_h * scale + margin * 2
    rows = [[255] * w for _ in range(h)]
    for y, row in enumerate(grid):
        for x, on in enumerate(row):
            if not on:
                continue
            for dy in range(scale):
                line = rows[margin + y * scale + dy]
                for dx in range(scale):
                    line[margin + x * scale + dx] = 0
    return _png_grey(w, h, rows), w, h


def _digit_run(text):
    """The longest run of digits in whatever the model said."""
    best = ""
    for run in re.findall(r"\d+", text or ""):
        if len(run) > len(best):
            best = run
    return best


# ------------------------------------------------------------------- tests

def t_asr(tok, model):
    """Real-time factor: seconds of audio transcribed per second of wall clock.

    The same figure speech is judged by, the other way round. Above 1.0 means
    the box can keep up with someone talking, which is the threshold that
    decides whether transcription can be live or only after the fact.

    Accuracy is deliberately not scored. Scoring it needs a clip of known
    speech, and there is no way to generate one here without a text-to-speech
    model loaded, which would make an ASR benchmark depend on a second model.
    What each run does record is the text that came back, so a model returning
    nothing at all is visible rather than hidden behind a good rate.
    """
    say("\n  SPEECH RECOGNITION  (real-time factor)")
    rows = []
    for secs in (2.0, 5.0, 10.0):
        clip = _wav_bytes(secs)
        t0 = time.time()
        r = api_form(gw("/v1/audio/transcriptions"), tok,
                     {"model": model, "response_format": "json"},
                     {"file": ("clip.wav", "audio/wav", clip)}, timeout=300)
        wall = time.time() - t0
        if "_error" in r:
            if not_loaded(r):
                say("    no ASR model is resident; nothing to measure")
                return not_measured("no ASR model was resident on the device")
            capture("asr", gw("/v1/audio/transcriptions"), r)
            say(f"    {secs:>5.1f}s clip FAILED {r['_error'][:55]}"); continue
        if not isinstance(r.get("text"), str):
            capture("asr", gw("/v1/audio/transcriptions"), r,
                    "answered 200 with no text field")
        text = (r.get("text") or "").strip()
        rtf = round(secs / wall, 2) if wall else None
        rows.append({"audio_s": secs, "wall_s": round(wall, 2), "rtf": rtf,
                     "chars": len(text), "text": text[:200]})
        say(f"    {secs:>5.1f}s clip  {wall:6.2f}s  {rtf}x real time  "
            f"{len(text):>4} chars back")
    if not rows:
        return None
    good = [r["rtf"] for r in rows if r.get("rtf")]
    if good:
        say(f"    median {statistics.median(good):.2f}x real time")
    return {"runs": rows,
            "rtf": round(statistics.median(good), 2) if good else None,
            "returned_text": any(r["chars"] for r in rows)}


def t_ocr(tok, model):
    """Seconds per page, and whether the digits on the page came back.

    Two kinds of model wear this class on the box: a vision language model that
    will read the page through chat completions, and a dedicated OCR server
    behind the gateway's own /v1/ocr route. The gateway is tried first and chat
    second, and whichever answered is recorded, because "two seconds a page" is
    not comparable between the two paths and the record has to say which it was.
    """
    say("\n  OCR  (seconds per page)")
    page, w, h = _digits_png()
    b64 = base64.b64encode(page).decode()
    rows, via = [], None
    for i in range(3):
        t0 = time.time()
        text, how = None, None
        r = api(gw("/v1/ocr"), tok,
                {"model": model, "image": b64}, timeout=300)
        if "_error" not in r:
            text, how = _ocr_text(r), "ocr gateway"
        else:
            c = api(gw("/v1/chat/completions"), tok, {
                "model": model, "max_tokens": 64, "messages": [{"role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + b64}},
                        {"type": "text",
                         "text": "Read the digits on this page. Reply with the "
                                 "digits only."}]}]}, timeout=300)
            if "_error" not in c:
                text = (((c.get("choices") or [{}])[0].get("message") or {})
                        .get("content") or "")
                how = "chat completions"
            else:
                # Both paths failed. Record the second one too: printing only
                # the gateway's 404 hides why the fallback did not work, and
                # the fallback is the path a vision model would have taken.
                # A separate key, not "ocr": capture keeps the first record
                # per key, so filing both refusals under one name would throw
                # away whichever arrived second, and the pair is the finding.
                capture("ocr fallback", gw("/v1/chat/completions"), c,
                        "the chat fallback refused it too, so there is no "
                        "route left to read a page with")
        wall = time.time() - t0
        if text is None:
            if not_loaded(r):
                say("    no OCR model is resident; nothing to measure")
                return not_measured("no Image-to-Text model was resident on the device")
            capture("ocr", gw("/v1/ocr"), r)
            say(f"    page {i+1} FAILED {str(r.get('_error'))[:55]}"); continue
        if not _digit_run(text):
            capture("ocr", gw("/v1/ocr"), {"text_returned": text[:400]},
                    "answered with no digits in it, so either the page was not "
                    "read or the reply is shaped differently than expected")
        via = via or how
        got = _digit_run(text)
        rows.append({"wall_s": round(wall, 2), "via": how, "read": got,
                     "correct": got == OCR_DIGITS})
        say(f"    page {i+1}  {wall:6.2f}s  via {how:<16} read {got or '(nothing)'}"
            + ("  correct" if got == OCR_DIGITS else ""))
    if not rows:
        return None
    med = statistics.median(r["wall_s"] for r in rows)
    hit = sum(1 for r in rows if r["correct"])
    say(f"    median {med:.2f}s per page, {hit} of {len(rows)} read correctly")
    return {"runs": rows, "s_per_page": round(med, 2), "via": via,
            "expected": OCR_DIGITS, "correct": hit,
            "page_px": [w, h]}


def _ocr_text(payload):
    """Pull the text out of whatever shape the OCR upstream returned.

    The /v1/ocr route is a gateway in front of whichever OCR server the model
    ships, and those do not agree on a response shape. Rather than guess one,
    this walks the payload and takes every string it finds, which is enough to
    answer the only question asked of it: did the digits come back.
    """
    found = []

    def walk(node):
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    return " ".join(found)


def t_music(tok, model):
    """Seconds of audio produced per second of wall clock.

    The music service is the one subsystem on this box that may answer either
    way: some of its routes hand back a file, and the presence of /progress and
    /sessions endpoints says others hand back a job to poll. Both are handled
    and the record says which happened, because a number from a blocking call
    and a number from a polled job are not the same measurement.
    """
    say("\n  MUSIC  (seconds of audio per second of wall clock)")
    rows = []
    for want in (8, 16):
        t0 = time.time()
        body = {"model": model, "prompt": MUSIC_PROMPT,
                "duration": want, "format": "wav"}
        raw = api_raw(gw("/v1/music/generate"), tok, body, timeout=900)
        # It objects to one field at a time: duration first, then format. So
        # keep dropping whatever it names until it stops naming things, with
        # a bound so a device that refuses everything cannot spin here.
        for _ in range(4):
            drop = _rejected_field(raw)
            if not drop or drop not in body:
                break
            if drop in NEVER_DROP:
                # It will not take the prompt where every other model takes
                # it. Ask once, with the field gone, what it wants instead:
                # the refusal to an empty request names the alternatives.
                say(f"    it refuses {drop}; asking what it wants instead")
                probe = dict(body)
                probe.pop(drop)
                raw = api_raw(gw("/v1/music/generate"), tok, probe, timeout=900)
                break
            say(f"    this model will not take {drop}; asking again without it")
            body.pop(drop)
            t0 = time.time()
            raw = api_raw(gw("/v1/music/generate"), tok, body, timeout=900)
        # A refusal that names what it wants instead is the box handing over
        # the shape. SongGeneration asks for config.lyrics, config.caption,
        # config.instruction or prompt; the first config path it offers gets
        # the same prompt every other music model is given.
        need = [f for f in _required_fields(raw) if f.startswith("config.")]
        if need and not _as_wav(raw):
            say(f"    it asks for {need[0]}; sending the prompt there instead")
            t0 = time.time()
            raw = api_raw(gw("/v1/music/generate"), tok,
                          _set_path({"model": model}, need[0], MUSIC_PROMPT),
                          timeout=900)
        audio, how = _as_wav(raw), None
        if audio:
            how = "direct"
        else:
            job = raw
            if isinstance(raw, (bytes, bytearray)):
                try:
                    job = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    job = {}
            sid = (job or {}).get("session_id") or (job or {}).get("id")
            if sid:
                audio, how = _music_wait(tok, sid), "session"
        wall = time.time() - t0
        if not audio:
            if not_loaded(raw):
                say("    no music model is resident; nothing to measure")
                return not_measured("no Music Generation model was resident on the device")
            capture("music", gw("/v1/music/generate"), raw,
                    "neither a WAV nor a session id came back")
            say(f"    {want:>3}s FAILED {str(raw)[:60]}"); continue
        secs = _wav_seconds(audio)
        ratio = round(secs / wall, 2) if secs and wall else None
        rows.append({"asked_s": want, "audio_s": round(secs, 2) if secs else None,
                     "wall_s": round(wall, 2), "audio_per_s": ratio, "via": how})
        say(f"    {want:>3}s asked  {wall:7.2f}s  "
            + (f"{secs:5.1f}s audio  {ratio}x" if secs else "(duration unknown)")
            + f"  via {how}")
    if not rows:
        return None
    good = [r["audio_per_s"] for r in rows if r.get("audio_per_s")]
    if good:
        say(f"    median {statistics.median(good):.2f}s of audio per second")
    return {"runs": rows,
            "audio_per_s": round(statistics.median(good), 2) if good else None,
            "via": rows[0]["via"]}


def _music_wait(tok, session, limit=900):
    """Poll a music job until it has something to download, or time runs out."""
    deadline = time.time() + limit
    while time.time() < deadline:
        p = api(gw(f"/v1/music/progress?session_id={session}"), tok, timeout=60)
        state = str((p or {}).get("status") or (p or {}).get("state") or "").lower()
        if state in ("failed", "error"):
            return None
        if state in ("done", "finished", "completed", "success") or \
                (p or {}).get("progress") == 100:
            break
        time.sleep(2)
    return _as_wav(api_raw(gw(f"/v1/music/sessions/{session}/download"),
                           tok, timeout=300))


def t_rerank(tok, model):
    """Query-document pairs scored per second, and whether the right one wins.

    Throughput is the headline, because reranking is something you do to a
    whole result set at once and what matters is how big a set fits inside a
    search that still feels instant. The sense check costs nothing and is worth
    having: one of the four passages actually answers the query, so a model
    that is fast and ranks it below the others has told you something.
    """
    say("\n  RERANKING  (pairs per second)")
    rows, top1 = [], None
    for n in (4, 16, 64):
        # The four real passages, repeated to reach the batch size, with the
        # answer kept at a known index so the check still means something.
        docs = [RERANK_DOCS[i % len(RERANK_DOCS)] for i in range(n)]
        t0 = time.time()
        r = api(gw("/v1/rerank"), tok,
                {"model": model, "query": RERANK_QUERY, "documents": docs,
                 "top_n": len(docs)}, timeout=300)
        wall = time.time() - t0
        if "_error" in r:
            if not_loaded(r):
                say("    no reranking model is resident; nothing to measure")
                return not_measured("no Text Reranking model was resident on the device")
            capture("rerank", gw("/v1/rerank"), r)
            say(f"    {n:>3} docs FAILED {r['_error'][:55]}"); continue
        ranked = r.get("results") or r.get("data") or []
        if not ranked:
            capture("rerank", gw("/v1/rerank"), r,
                    "answered 200 with neither a results nor a data list")
        rate = round(len(docs) / wall, 1) if wall else 0
        if top1 is None and ranked:
            first = ranked[0]
            idx = first.get("index") if isinstance(first, dict) else None
            top1 = (idx % len(RERANK_DOCS)) == RERANK_ANSWER if isinstance(idx, int) else None
        rows.append({"docs": n, "wall_s": round(wall, 3),
                     "returned": len(ranked), "pairs_per_s": rate})
        say(f"    {n:>3} docs  {wall:6.3f}s  {rate:8.1f} pairs/s  "
            f"{len(ranked)} scored")
    if not rows:
        return None
    say(f"    best {max(r['pairs_per_s'] for r in rows):.1f} pairs/s"
        + ("" if top1 is None else
           ("; the passage that answers the query ranked first" if top1
            else "; the passage that answers the query did NOT rank first")))
    return {"runs": rows, "pairs_per_s": max(r["pairs_per_s"] for r in rows),
            "top1_correct": top1}


TESTS = {"prefill": t_prefill, "sustained": t_sustained,
         "concurrency": t_concurrency, "thinking": t_thinking,
         "image": t_image, "speech": t_speech, "embed": t_embed,
         "asr": t_asr, "ocr": t_ocr, "music": t_music, "rerank": t_rerank}

# Which tests mean anything for which kind of model. The keys are the device's
# own type strings, exactly as the model store spells them, because that is what
# a record is matched against.
SUITES = {
    "Text Generation":    ["prefill", "sustained", "concurrency", "thinking"],
    "Image-Text-to-Text": ["prefill", "sustained", "concurrency", "thinking"],
    "Text-to-Image":      ["image"],
    "Text-to-Speech":     ["speech"],
    "Text Embedding":     ["embed"],
    "ASR":                ["asr"],
    "Image-to-Text":      ["ocr"],
    "Music Generation":   ["music"],
    "Text Reranking":     ["rerank"],
}

# The headline figure per class: (key in results, unit, what it means).
CLASS_METRIC = {
    "Text Generation":    ("decode_tok_s", "tok/s", "sustained decode"),
    "Image-Text-to-Text": ("decode_tok_s", "tok/s", "sustained decode"),
    "Text-to-Image":      ("s_per_image", "s/img", "per 512 plate"),
    "Text-to-Speech":     ("rtf", "x", "faster than real time"),
    "Text Embedding":     ("emb_per_s", "emb/s", "embeddings per second"),
    "ASR":                ("rtf", "x", "faster than real time"),
    "Image-to-Text":      ("s_per_page", "s/page", "per generated page"),
    "Music Generation":   ("audio_per_s", "x", "audio per second of wall clock"),
    "Text Reranking":     ("pairs_per_s", "pairs/s", "query-document pairs"),
}
LOWER_IS_BETTER = {"s_per_image", "s_per_page"}

# Every class the suite has a test for, which is SUITES read the other way.
# It used to be a second hand-written set and the two drifted the moment a class
# was added, so now there is one list and this is a view of it.
CHAT_TYPES = set(SUITES)


def _came_up_slow(test):
    """Did this test fail only because the model was not answering yet?

    load() waits for the model to appear in /running and the device puts it
    there before its runtime is accepting, so the first call can come back
    502 Bad Gateway. Two embedding models recorded nothing at all in the
    2026-09-19 sweep for this reason and both measured fine minutes later,
    which is a false negative published as a fact.

    There is no field on the device that separates "listed" from "answering",
    so the only honest signal is the 502 itself.
    """
    rec = UNPARSED.get(test) or {}
    return rec.get("status") == 502


def suite(tok, model, want, meta, cat=None):
    """One model, all the requested tests, returned as a record."""
    # Per model, not per process: the web app serves for days and one sweep's
    # surprise must not be reported against the next sweep's model.
    UNPARSED.clear()
    tel = telemetry(tok)
    # The resident set at the moment THIS model's tests begin, not at the
    # moment the sweep began. A sweep loads and unloads as it goes, so the
    # conditions the fifth model met are not the ones the first met, and a
    # single envelope at the top of the file would quietly claim otherwise.
    began = conditions(tok, cat)
    say(f"\n  ---- {model} " + "-" * max(0, 56 - len(model)))
    results = {}
    t0 = time.time()
    # A leaderboard that ranks a speech model by tokens per second is measuring
    # nothing. Each class only runs the tests that mean something for it.
    allowed = SUITES.get(meta.get("type"), [])
    todo = [n for n in want if n in allowed] or allowed
    skipped = [n for n in want if n not in allowed]
    if skipped:
        say(f"    skipping {', '.join(skipped)}: not meaningful for a "
            f"{meta.get('type')} model")
    for i, name in enumerate(todo):
        emit("test", model=model, test=name, index=i, total=len(todo))
        results[name] = TESTS[name](tok, model)
        if not results[name] and _came_up_slow(name):
            say("    the runtime answered 502; giving it 15s and asking again")
            UNPARSED.pop(name, None)
            time.sleep(15)
            results[name] = TESTS[name](tok, model)
        emit("test_done", model=model, test=name, index=i, total=len(todo))
    # Keys may be a test name or a test name plus a qualifier ("ocr
    # fallback"), so match on the first word rather than the whole key,
    # or a second capture from the same test silently vanishes here.
    unparsed = {k: v for k, v in UNPARSED.items()
                if k in todo or k.split()[0] in todo}
    return {
        "model": model,
        "params": meta.get("params"),
        "type": meta.get("type"),
        "npu_usage": meta.get("npu_usage"),
        "total_size": meta.get("total_size"),
        "context_window": meta.get("context_window"),
        "elapsed_s": round(time.time() - t0, 1),
        "npu_mem_total_mb": tel.get("npu_mem_total_mb"),
        "conditions_at_start": began,
        # What was actually sent, rather than what the defaults are today. A
        # sampling change between releases would otherwise be invisible in a
        # file and would look like the hardware getting slower.
        "sampling": SAMPLING,
        "tests_run": todo,
        "tests_skipped": skipped,
        # The first response per test this app could not use. Absent when
        # everything parsed, which is the common case.
        "unparsed": unparsed or None,
        "results": results,
    }


def selfcheck(tok):
    """Prove the install works, in about ten seconds.

    Five things, in the order they fail for a newcomer: can we reach the box,
    does the key work, is anything loaded, does a real inference come back, and
    can we write a result. A bad key or an empty box should say so plainly here
    rather than halfway through a benchmark."""
    ok = True

    def step(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        say(f"  {'ok  ' if passed else 'FAIL'} {label:<34} {detail}")

    w = where()
    say(f"\n  TiinyBench {VERSION} selfcheck")
    say(f"  device   {w['host']}   found by {w['source']}   {w['plane']} plane"
        + (f"   serial {w['serial']}" if w.get("serial") else ""))
    say("  gateway  " + (f"port 80 with Host: {w['gateway_vhost']}"
                         if w["gateway_transport"] == "vhost"
                         else f"direct on port {w['gateway_port']}")
        + f"   ({w['gateway_transport']})\n")

    t0 = time.time()
    info = api(mgmt("/api/v1/sys/device_info"), tok, timeout=20)
    step("device reachable", "_error" not in info,
         info.get("_error", "") or f"TiinyOS {info.get('tiiny_os', '?')}  "
                                   f"service {info.get('version', '?')}  "
                                   f"{(time.time()-t0)*1000:.0f}ms")

    cat = api(gw("/api/v1/models/"), tok, timeout=60)
    n = len(cat.get("data") or [])
    step("api key accepted", "_error" not in cat and n > 0,
         cat.get("_error", "") or f"{n} models installed")

    live = running(tok)
    free, total = npu_free(tok)
    step("something is loaded", bool(live),
         (", ".join(m.split("/")[-1] for m in live) or
          "nothing loaded - load a model in TiinyOS")
         + (f"   NPU {total - free}/{total}" if total else ""))

    # Any loaded model can prove the transport carries a real inference, and a
    # box serving embeddings is a box doing real work. Timing one embedding
    # rather than reporting a failure means this check never asks somebody to
    # load a chat model just to satisfy it.
    target = etarget = None
    for m in live:
        meta = next((x for x in (cat.get("data") or []) if x.get("id") == m), {})
        if not target and meta.get("type") in ("Text Generation", "Image-Text-to-Text"):
            target = m
        if not etarget and meta.get("type") == "Text Embedding":
            etarget = m
    if target:
        r = chat(tok, target, "Reply with the single word: ok", 8)
        step("inference returns", "error" not in r,
             r.get("error", "") or
             f"{target.split('/')[-1]}  {r.get('decode_tok_s', 0):.1f} tok/s  "
             f"{r.get('wall_s', 0):.2f}s")
    elif etarget:
        t1 = time.time()
        v = api(gw("/v1/embeddings"), tok,
                {"model": etarget, "input": ["selfcheck"]}, timeout=120)
        dim = len(((v.get("data") or [{}])[0] or {}).get("embedding") or [])
        step("inference returns", "_error" not in v and dim > 0,
             v.get("_error", "") or
             f"{etarget.split('/')[-1]}  1 embedding, dim {dim}  "
             f"{time.time() - t1:.2f}s  (no chat model loaded)")
    else:
        step("inference returns", False,
             "nothing loaded that this can time; load a model in TiinyOS")

    try:
        OUT.mkdir(exist_ok=True)
        probe = OUT / ".selfcheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        step("results directory writable", True, str(OUT))
    except Exception as exc:  # noqa: BLE001
        step("results directory writable", False, str(exc)[:60])

    say("")
    if ok:
        say("  All good. Run one with:   tiiny-bench --label first-run")
        say("  Or open the app with:     tiiny-bench --serve")
    else:
        say("  Something above needs fixing before a benchmark will mean anything.")
    say("")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(prog="tiiny-bench")
    p.add_argument("--label", help="name this run (required unless --catalog/--report)")
    p.add_argument("--only", default="", help="comma list: " + ",".join(TESTS))
    p.add_argument("--all", action="store_true",
                   help="sweep every text model. LOADS AND UNLOADS MODELS.")
    p.add_argument("--model", help="benchmark one model by id. LOADS AND UNLOADS IT.")
    p.add_argument("--catalog", action="store_true", help="list what is installed")
    p.add_argument("--selfcheck", action="store_true",
                   help="prove the install works: reach the device, time one real call")
    p.add_argument("--report", action="store_true", help="build report.html from results")
    p.add_argument("--show", help="print a saved result file")
    p.add_argument("--serve", nargs="?", const=8425, type=int, metavar="PORT",
                   help="run the web app (default port 8425)")
    p.add_argument("--host", help="the Tiiny's address, instead of finding it. "
                                 "TIINY_BASE and the farm's device file win over this.")
    p.add_argument("--serial", help="pick a box by serial when more than one answers")
    p.add_argument("--rescan", action="store_true",
                   help="ignore the saved address and look for boxes again")
    p.add_argument("--where", action="store_true",
                   help="find the Tiiny, say how it was reached, and stop. "
                        "Touches nothing on the device.")
    p.add_argument("--version", action="version", version="tiiny-bench " + VERSION)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)

    if a.show:
        print(json.dumps(json.loads(
            pathlib.Path(a.show).read_text(encoding="utf-8")), indent=2)[:4000])
        return 0
    if a.report:
        import report
        out = report.build(OUT, HERE / "report.html")
        print(f"  wrote {out}")
        return 0

    if a.serve:
        import serve
        return serve.run(a.serve, host=a.host, serial=a.serial, rescan=a.rescan)

    if a.where:
        # --where is what somebody runs when nothing is working, so a box that
        # cannot be found is the answer it prints, not a failure it raises. It
        # exits 0 either way: the question was "what do you see", and "nothing"
        # is a complete reply to it.
        w, err = connect_soft(host=a.host, serial=a.serial, rescan=a.rescan)
        if not w:
            print(f"\n  tiiny-bench {VERSION}")
            print("  " + err.strip().replace("\n", "\n  "))
            print("")
            return 0
        info = api(mgmt("/api/v1/sys/device_info"), "probe", timeout=10)
        print(f"\n  tiiny-bench {VERSION}")
        print(f"  address    {w['host']}")
        print(f"  found by   {w['source']}")
        print(f"  plane      {w['plane']}")
        if w.get("serial"):
            print(f"  serial     {w['serial']}")
        print("  gateway    " + (
            f"port 80, Host: {w['gateway_vhost']}" if w["gateway_transport"] == "vhost"
            else f"direct on port {w['gateway_port']}")
            + f"   ({w['gateway_transport']})")
        fw = firmware(info)
        if fw:
            print(f"  firmware   TiinyOS {fw.get('tiiny_os')}  "
                  f"service {fw.get('version')}")
        print("")
        return 0

    # Resolving the address is the first thing that happens, and it happens
    # here rather than at import so that --help and --report still work on a
    # machine with no Tiiny attached.
    connect(host=a.host, serial=a.serial, rescan=a.rescan)

    tok = key()

    if a.selfcheck:
        return selfcheck(tok)
    if a.catalog:
        rows = catalog(tok)
        live = set(running(tok))
        free, total = npu_free(tok)
        print(f"\n  {len(rows)} models installed   NPU {total - free}/{total} in use\n")
        print(f"  {'':1} {'params':>7} {'size':>7} {'npu':>4}  {'type':<20} id")
        for m in rows:
            mark = "*" if m["id"] in live else " "
            gb = (m.get("total_size") or 0) / 1e9
            print(f"  {mark} {str(m.get('params') or '-'):>7} {gb:>6.1f}G "
                  f"{str(m.get('npu_usage') or '-'):>4}  {(m.get('type') or '-')[:20]:<20} {m['id']}")
        print("\n  * currently running")
        return 0

    if not a.label:
        p.error("--label is required")

    want = [s.strip() for s in a.only.split(",") if s.strip()] or list(TESTS)
    for w in want:
        if w not in TESTS:
            p.error(f"unknown test {w!r}; known: {', '.join(TESTS)}")

    info = api(mgmt("/api/v1/sys/device_info"), tok, timeout=20)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Before the record, because the envelope needs it: the resident set is
    # only worth recording if each model's unit cost comes with it.
    cat = {m["id"]: m for m in catalog(tok)}
    rec = {"label": a.label, "stamp": stamp, "build": info.get("tiiny_os"),
           "host": HOST, "suite_version": 2,
           "bench_version": VERSION,
           # Everything that makes these numbers comparable to somebody else's
           # box six months from now: which machine drove the benchmark, down
           # which wire, against what firmware, with what else resident at the
           # moment each test began. Recorded raw and local; public_view is
           # the one place that makes a record safe to publish.
           "provenance": envelope(tok, cat, info),
           # Which address, which plane and which transport produced these
           # numbers. Two runs of the same box over USB and over the port 80
           # vhost are not the same measurement, and a result that does not say
           # which one it was cannot be compared with anything later.
           "connection": dict(where(), round_trip=round_trip(tok)),
           "firmware": firmware(info),
           "models": []}
    path = OUT / f"{stamp}-suite-{a.label}.json"

    was_running = running(tok)

    # ---- which models, and are we allowed to touch the box? ----------------
    if a.all:
        targets = [m for m in cat.values() if m.get("type") in CHAT_TYPES]
        skipped = [m for m in cat.values() if m.get("type") not in CHAT_TYPES]
        print(f"\n  SWEEP: {len(targets)} text models, one at a time.")
        print(f"  This LOADS AND UNLOADS models. Currently running: "
              f"{', '.join(was_running) or 'nothing'}")
        print(f"  Skipping {len(skipped)} non-chat models "
              f"({', '.join(sorted({m['type'] for m in skipped}))})")
        print(f"  Expect roughly {len(targets) * 7} minutes. Results are written after "
              f"every model, so this is safe to interrupt.\n")
    elif a.model:
        if a.model not in cat:
            p.error(f"{a.model!r} is not installed. tiiny-bench --catalog lists what is.")
        targets = [cat[a.model]]
        print(f"\n  Benchmarking {a.model}. This loads it and unloads it after.\n")
    else:
        live = running(tok)
        if not live:
            sys.exit("no model loaded. Load one in TiinyOS, or pass --model / --all.")
        targets = [cat.get(live[0], {"id": live[0]})]
        print(f"\n  Benchmarking whatever is loaded: {live[0]}")
        print("  Nothing will be loaded or unloaded.\n")

    touching = bool(a.all or a.model)

    for i, meta in enumerate(targets, 1):
        model = meta["id"]
        emit("model", model=model, index=i, total=len(targets))
        print(f"\n  [{i}/{len(targets)}] {model}")
        if touching:
            for other in running(tok):
                if other != model:
                    print(f"    unloading {other} to make room")
                    unload(tok, other)
            if not load(tok, model):
                rec["models"].append({"model": model, "error": "failed to load"})
                path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                continue
        try:
            rec["models"].append(suite(tok, model, want, meta, cat))
        except KeyboardInterrupt:
            print("\n  interrupted; what finished is saved")
            break
        finally:
            # Checkpoint after EVERY model. A sweep is an hour long and a box
            # that reboots at minute 50 should not cost the whole run.
            path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        if touching:
            unload(tok, model)

    # Put the box back the way we found it.
    if touching and was_running:
        print("\n  restoring what was loaded before:", ", ".join(was_running))
        for m in running(tok):
            if m not in was_running:
                unload(tok, m)
        for m in was_running:
            load(tok, m)

    print(f"\n  saved {path}")
    print(f"  build the report with:  {sys.argv[0]} --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
