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
import errno
import glob
import json
import pathlib
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import os

VERSION = "0.1.3"

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

# Deterministic filler so prompt lengths are repeatable across runs.
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Engineers measured the throughput carefully and recorded every result. ")

# The suite talks to chat completions, so these are the model types it can
# actually exercise. Everything else on the box is listed but skipped, with the
# reason shown, rather than silently dropped.
# Every class the suite has a test for. Anything else is listed and skipped
# with the reason shown, rather than silently dropped.
CHAT_TYPES = {"Text Generation", "Image-Text-to-Text", "Text-to-Image",
              "Text-to-Speech", "Text Embedding"}


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


def _request(target, tok, body, timeout, method, read):
    """One call to one service, over whichever transport reaches it.

    The service's own port is tried first. A refused connection, and only a
    refused connection, moves that service onto the port 80 vhost for the rest
    of the run. An HTTP error still tells us the transport was right, so it is
    recorded before the error is handed back.
    """
    err = None
    for url, extra, mode in _attempts(target):
        req = urllib.request.Request(
            url, method=method or ("POST" if body is not None else "GET"),
            data=json.dumps(body).encode() if body else None,
            headers={"Authorization": f"Bearer {tok}", **extra,
                     **({"Content-Type": "application/json"} if body else {})})
        try:
            with OPENER.open(req, timeout=timeout) as r:
                _mark(target, mode)
                return r.read() if read else json.load(r)
        except urllib.error.HTTPError as e:
            _mark(target, mode)
            return {"_error": str(e)[:140]}
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


def chat(tok, model, prompt, max_tokens, thinking=False):
    body = {"model": model, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    v = api(gw("/v1/chat/completions"), tok, body)
    wall = time.time() - t0
    if "_error" in v:
        return {"error": v["_error"], "wall_s": round(wall, 2)}
    t = v.get("timings") or {}
    u = v.get("usage") or {}
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": u.get("prompt_tokens", t.get("prompt_n", 0)),
        "out_tokens": u.get("completion_tokens", t.get("predicted_n", 0)),
        "prefill_tok_s": round(t.get("prompt_per_second") or 0, 2),
        "decode_tok_s": round(t.get("predicted_per_second") or 0, 2),
        "prefill_ms": round(t.get("prompt_ms") or 0, 1),
        "ttft_s": round((t.get("prompt_ms") or 0) / 1000
                        + (t.get("predicted_per_token_ms") or 0) / 1000, 3),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }


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

    Distinct from catalog(), which is only what is already on the box. A
    benchmark that cannot fetch a model it has not got has a dead end in it,
    and "download it, then measure it" is the whole point of closing this loop.
    """
    d = api(gw("/api/v1/models/online_models"), tok, timeout=90)
    if isinstance(d, dict) and "_error" in d:
        return []
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
            say(f"    image {i+1} FAILED {str(raw)[:60]}"); continue
        rows.append({"wall_s": round(wall, 2), "bytes": len(raw), "seed": 1000 + i})
        say(f"    image {i+1}  {wall:6.2f}s  {len(raw)//1024:>5} KB")
    if not rows:
        return None
    med = statistics.median(r["wall_s"] for r in rows)
    say(f"    median {med:.2f}s per plate")
    return {"runs": rows, "s_per_image": round(med, 2)}


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


def t_speech(tok, model):
    """Real-time factor: seconds of audio produced per second of wall clock.
    Above 1.0 means it can talk faster than a person listens, which is the only
    threshold that matters for anything conversational."""
    say("\n  SPEECH  (real-time factor)")
    rows = []
    for i, text in enumerate(SPEECH_TEXTS):
        t0 = time.time()
        raw = api_raw(gw("/v1/audio/speech"), tok,
                      {"model": model, "input": text, "response_format": "wav"},
                      timeout=300)
        wall = time.time() - t0
        if not isinstance(raw, (bytes, bytearray)):
            say(f"    clip {i+1} FAILED {str(raw)[:60]}"); continue
        secs = _wav_seconds(raw)
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
    return {"runs": rows,
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


TESTS = {"prefill": t_prefill, "sustained": t_sustained,
         "concurrency": t_concurrency, "thinking": t_thinking,
         "image": t_image, "speech": t_speech, "embed": t_embed}

# Which tests mean anything for which kind of model.
SUITES = {
    "Text Generation":    ["prefill", "sustained", "concurrency", "thinking"],
    "Image-Text-to-Text": ["prefill", "sustained", "concurrency", "thinking"],
    "Text-to-Image":      ["image"],
    "Text-to-Speech":     ["speech"],
    "Text Embedding":     ["embed"],
}

# The headline figure per class: (key in results, unit, what it means).
CLASS_METRIC = {
    "Text Generation":    ("decode_tok_s", "tok/s", "sustained decode"),
    "Image-Text-to-Text": ("decode_tok_s", "tok/s", "sustained decode"),
    "Text-to-Image":      ("s_per_image", "s/img", "per 512 plate"),
    "Text-to-Speech":     ("rtf", "x", "faster than real time"),
    "Text Embedding":     ("emb_per_s", "emb/s", "embeddings per second"),
}
LOWER_IS_BETTER = {"s_per_image"}


def suite(tok, model, want, meta):
    """One model, all the requested tests, returned as a record."""
    tel = telemetry(tok)
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
        emit("test_done", model=model, test=name, index=i, total=len(todo))
    return {
        "model": model,
        "params": meta.get("params"),
        "type": meta.get("type"),
        "npu_usage": meta.get("npu_usage"),
        "total_size": meta.get("total_size"),
        "elapsed_s": round(time.time() - t0, 1),
        "npu_mem_total_mb": tel.get("npu_mem_total_mb"),
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
    rec = {"label": a.label, "stamp": stamp, "build": info.get("tiiny_os"),
           "host": HOST, "suite_version": 2,
           "bench_version": VERSION,
           # Which address, which plane and which transport produced these
           # numbers. Two runs of the same box over USB and over the port 80
           # vhost are not the same measurement, and a result that does not say
           # which one it was cannot be compared with anything later.
           "connection": where(),
           "firmware": firmware(info),
           "models": []}
    path = OUT / f"{stamp}-suite-{a.label}.json"

    cat = {m["id"]: m for m in catalog(tok)}
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
            rec["models"].append(suite(tok, model, want, meta))
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
