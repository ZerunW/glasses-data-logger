"""Wi-Fi link: the iPhone connects to this computer over TLS 1.3.

Trust is established once by scanning a QR code shown on the computer (web page or
terminal). The code carries the SHA-256 fingerprint of this receiver's self-signed
certificate, so the phone only talks to this computer (certificate pinning), and a
random token the phone presents inside the encrypted connection, so this computer
only accepts a paired phone.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

import ifaddr
import numpy as np
import qrcode
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from zeroconf import ServiceInfo, Zeroconf

from . import protocol as p
from . import paths
from .receiver import Receiver
from .session import Session, computer_name

log = logging.getLogger(__name__)

STATE_DIR = paths.state_dir()  # per platform, see paths.py
PAIRING_SCHEME = "gdpair"
HANDSHAKE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Identity:
    cert_path: Path
    key_path: Path
    fingerprint: str  # lowercase hex SHA-256 of the DER certificate
    token: str
    name: str


def load_or_create_identity(state_dir: Path = STATE_DIR, reset: bool = False) -> Identity:
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    cert_path, key_path, pairing_path = state_dir / "cert.pem", state_dir / "key.pem", state_dir / "pairing.json"
    if reset:
        for path in (cert_path, key_path, pairing_path):
            path.unlink(missing_ok=True)
    if not (cert_path.exists() and key_path.exists()):
        _create_certificate(cert_path, key_path)
    if pairing_path.exists():
        token = json.loads(pairing_path.read_text())["token"]
    else:
        token = secrets.token_urlsafe(32)
        _write_private(pairing_path, json.dumps({"token": token}).encode())
    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    fingerprint = hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest()
    return Identity(cert_path, key_path, fingerprint, token, computer_name())


def _create_certificate(cert_path: Path, key_path: Path) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "GlassesData Receiver")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    _write_private(
        key_path,
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)  # the mode argument does not tighten an existing file
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


# Linux bridges for containers and virtual machines: the phone can never reach these, and
# 172.17.0.1 (Docker) would otherwise be tried before a carrier-grade NAT address.
_VIRTUAL_ADAPTERS = ("docker", "br-", "veth", "virbr", "lxcbr", "lxdbr", "vmnet", "cni", "flannel")


def local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    for adapter in ifaddr.get_adapters():
        if str(adapter.nice_name).startswith(_VIRTUAL_ADAPTERS) or str(adapter.name).startswith(_VIRTUAL_ADAPTERS):
            continue
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and not ip.ip.startswith(("127.", "169.254.")):
                addresses.append(ip.ip)
    return order_hosts(addresses)


# The phone tries the hosts in order, waiting seconds for each one that does not answer,
# so the likely LAN address goes first and carrier-grade NAT (Tailscale and the like) last.
_HOST_PREFERENCE = [
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
]
_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")


def order_hosts(addresses: list[str]) -> list[str]:
    """Private LAN ranges first (192.168, 10, 172.16-31), then the rest, 100.64.0.0/10 last."""

    def rank(text: str) -> tuple[int, ipaddress.IPv4Address]:
        address = ipaddress.IPv4Address(text)
        for index, network in enumerate(_HOST_PREFERENCE):
            if address in network:
                return index, address
        return (len(_HOST_PREFERENCE) + 1 if address in _CGNAT else len(_HOST_PREFERENCE)), address

    return sorted(set(addresses), key=rank)


def pairing_uri(identity: Identity, hosts: list[str], port: int) -> str:
    query = urlencode(
        {"name": identity.name, "hosts": ",".join(hosts), "port": port, "fp": identity.fingerprint, "token": identity.token},
        quote_via=quote,
    )
    return f"{PAIRING_SCHEME}://v1?{query}"


def qr_image(text: str, box_size: int = 6) -> np.ndarray:
    """A BGR image of the QR code (served as PNG by the web page)."""
    code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=box_size, border=3)
    code.add_data(text)
    code.make(fit=True)
    matrix = np.array(code.get_matrix(), dtype=np.uint8)
    pixels = np.where(matrix, 0, 255).astype(np.uint8)
    pixels = np.kron(pixels, np.ones((box_size, box_size), dtype=np.uint8))
    return np.stack([pixels] * 3, axis=-1)


def qr_png(text: str) -> bytes:
    """The QR code as PNG bytes, for the web page."""
    import cv2

    ok, buffer = cv2.imencode(".png", qr_image(text, box_size=8))
    if not ok:  # pragma: no cover - encoding a small grey image does not fail
        raise ValueError("could not encode the pairing code")
    return buffer.tobytes()


def qr_ascii(text: str) -> str:
    code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=2)
    code.add_data(text)
    code.make(fit=True)
    matrix = code.get_matrix()
    lines = []
    for top, bottom in zip(matrix[0::2], matrix[1::2] + [[False] * len(matrix[0])]):
        lines.append("".join({(0, 0): "█", (0, 1): "▀", (1, 0): "▄", (1, 1): " "}[(int(a), int(b))] for a, b in zip(top, bottom)))
    return "\n".join(lines)


def server_context(identity: Identity) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(identity.cert_path, identity.key_path)
    return context


class WifiSource(threading.Thread):
    """Accepts paired iPhones over TLS. ``port`` 0 picks a free port; ``port`` holds the
    bound one once ``wait_ready`` returns True."""

    def __init__(self, receiver: Receiver, identity: Identity, port: int, stop: threading.Event, advertise: bool = True) -> None:
        super().__init__(name="gd-wifi", daemon=True)
        self._receiver = receiver
        self._identity = identity
        self.port = port
        self._stop_event = stop
        self._advertise = advertise
        self._context = server_context(identity)
        self._zeroconf: Zeroconf | None = None
        self._ready = threading.Event()
        self._connections_lock = threading.Lock()
        self._connections: dict[threading.Thread, socket.socket] = {}

    def wait_ready(self, timeout: float) -> bool:
        """True once the listener accepts connections; False if it failed or timed out."""
        deadline = time.monotonic() + timeout
        while not self._ready.wait(0.05):
            if not self.is_alive() or time.monotonic() >= deadline:
                return self._ready.is_set()
        return True

    def run(self) -> None:
        status = self._receiver.set_source_status
        try:
            listener = socket.create_server(("0.0.0.0", self.port), reuse_port=False)
        except OSError as error:
            status("wifi", f"cannot listen on port {self.port} ({error.strerror or error})")
            log.error("Wi-Fi: cannot listen on port %d (%s)", self.port, error.strerror or error)
            return
        self.port = listener.getsockname()[1]
        listener.settimeout(0.5)
        with listener:
            self._start_bonjour()
            try:
                status("wifi", f"listening on port {self.port} - use Pair over Wi-Fi on the web page to show the pairing code")
                self._ready.set()
                while not self._stop_event.is_set():
                    try:
                        connection, address = listener.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    thread = threading.Thread(target=self._serve, args=(connection, address), name="gd-wifi-conn", daemon=True)
                    with self._connections_lock:
                        # A duplicate, because wrap_socket detaches ``connection`` from its descriptor.
                        self._connections[thread] = connection.dup()
                    thread.start()
            finally:
                self._stop_bonjour()
        self._close_connections()

    def _serve(self, connection: socket.socket, address: tuple[str, int]) -> None:
        try:
            self._serve_connection(connection, address)
        finally:
            with self._connections_lock:
                waker = self._connections.pop(threading.current_thread(), None)
            if waker is not None:
                waker.close()

    def _serve_connection(self, connection: socket.socket, address: tuple[str, int]) -> None:
        status = self._receiver.set_source_status
        connection.settimeout(HANDSHAKE_TIMEOUT_S)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            secure = self._context.wrap_socket(connection, server_side=True)
        except (ssl.SSLError, OSError) as error:
            log.info("TLS handshake with %s failed: %s", address[0], error)
            connection.close()
            return
        reason = Session(secure, "wifi", address[0], self._receiver, token=self._identity.token, name=self._identity.name).run()
        if reason.startswith("rejected"):
            log.warning("Connection from %s %s", address[0], reason)
        if not self._stop_event.is_set():
            status("wifi", f"listening on port {self.port} ({reason})")

    def _close_connections(self) -> None:
        """Wake connections still in the handshake or waiting for HELLO, and wait for them.

        shutdown() on our duplicate descriptor ends the TCP connection itself, which wakes
        the serving thread whatever object (plain or TLS) it is blocked on.
        """
        with self._connections_lock:
            pending = dict(self._connections)
        for connection in pending.values():
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread in pending:
            thread.join(1.0)

    def _start_bonjour(self) -> None:
        if not self._advertise:
            return
        try:
            addresses = [socket.inet_aton(ip) for ip in local_ipv4_addresses()]
            # The instance name is derived from the certificate, so it is unique per receiver
            # and the phone can find its paired Mac again after the Mac's IP address changes.
            info = ServiceInfo(
                p.BONJOUR_TYPE,
                f"GlassesData-{self._identity.fingerprint[:12]}.{p.BONJOUR_TYPE}",
                addresses=addresses,
                port=self.port,
                properties={"fp": self._identity.fingerprint, "name": self._identity.name, "v": "1"},
            )
            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(info)
        except Exception as error:  # discovery is a convenience; the QR code also carries addresses
            log.warning("Bonjour advertising unavailable: %s", error)
            self._stop_bonjour()

    def _stop_bonjour(self) -> None:
        zeroconf, self._zeroconf = self._zeroconf, None
        if zeroconf is None:
            return
        for step in (zeroconf.unregister_all_services, zeroconf.close):
            try:
                step()
            except Exception as error:
                log.debug("Bonjour %s failed: %s", step.__name__, error)
