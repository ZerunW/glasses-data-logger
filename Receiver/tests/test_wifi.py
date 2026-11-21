import dataclasses
import hashlib
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization

from glassesdata_receiver import protocol as p
from glassesdata_receiver import wifi
from glassesdata_receiver.receiver import Receiver


def mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_identity_is_created_reloaded_and_reset(tmp_path):
    state = tmp_path / "state"
    first = wifi.load_or_create_identity(state)
    assert {path.name for path in state.iterdir()} == {"cert.pem", "key.pem", "pairing.json"}
    again = wifi.load_or_create_identity(state)
    assert (again.fingerprint, again.token) == (first.fingerprint, first.token)
    reset = wifi.load_or_create_identity(state, reset=True)
    assert reset.fingerprint != first.fingerprint and reset.token != first.token
    assert len(reset.token) >= 40  # 32 random bytes, URL-safe base64


def test_private_files_are_owner_only(tmp_path):
    state = tmp_path / "state"
    identity = wifi.load_or_create_identity(state)
    assert mode(identity.key_path) == 0o600
    assert mode(state / "pairing.json") == 0o600
    assert mode(state) == 0o700


def test_fingerprint_is_sha256_of_the_der_certificate(tmp_path):
    identity = wifi.load_or_create_identity(tmp_path)
    certificate = x509.load_pem_x509_certificate(identity.cert_path.read_bytes())
    der = certificate.public_bytes(serialization.Encoding.DER)
    assert identity.fingerprint == hashlib.sha256(der).hexdigest()
    assert len(identity.fingerprint) == 64 and identity.fingerprint == identity.fingerprint.lower()


def test_pairing_uri_round_trips(tmp_path):
    identity = dataclasses.replace(wifi.load_or_create_identity(tmp_path), name="Zé's Mac & Co/1?")
    uri = wifi.pairing_uri(identity, ["192.168.1.20", "10.0.0.5"], 7900)
    parts = urlsplit(uri)
    assert (parts.scheme, parts.netloc) == (wifi.PAIRING_SCHEME, "v1")
    query = {key: values[0] for key, values in parse_qs(parts.query, strict_parsing=True).items()}
    assert query == {
        "name": identity.name,
        "hosts": "192.168.1.20,10.0.0.5",
        "port": "7900",
        "fp": identity.fingerprint,
        "token": identity.token,
    }
    assert " " not in uri


def test_qr_image_is_a_bgr_square():
    image = wifi.qr_image("gdpair://v1?x=1")
    assert image.ndim == 3 and image.shape[2] == 3 and image.shape[0] == image.shape[1]


@pytest.fixture
def wifi_receiver(tmp_path):
    identity = wifi.load_or_create_identity(tmp_path / "state")
    hub = Receiver(tmp_path / "recordings")
    stop = threading.Event()
    source = wifi.WifiSource(hub, identity, 0, stop, advertise=False)
    source.start()
    assert source.wait_ready(5.0)
    assert source.port != 0
    yield identity, source
    stop.set()
    hub.shutdown()
    source.join(3.0)
    assert not source.is_alive(), "WifiSource did not stop"
    with pytest.raises(OSError):  # the listener is closed
        socket.create_connection(("127.0.0.1", source.port), timeout=1).close()


def tls_hello(port: int, token: str) -> tuple[str, str, dict]:
    """Connect like the iPhone does: TLS 1.3, trust by fingerprint only."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with context.wrap_socket(raw) as secure:
            fingerprint = hashlib.sha256(secure.getpeercert(binary_form=True)).hexdigest()
            version = secure.version()
            secure.sendall(p.encode_json(p.MessageType.HELLO, {"protocol": p.PROTOCOL_VERSION, "app": "test", "token": token}))
            reader = p.FrameReader()
            while True:
                data = secure.recv(65536)
                assert data, "closed before HELLO_ACK"
                for message_type, payload in reader.feed(data):
                    message = p.decode_message(message_type, payload)
                    if message is not None and message.type is p.MessageType.HELLO_ACK:
                        return fingerprint, version, message.body


def test_tls_loopback_accepts_the_paired_token(wifi_receiver):
    identity, source = wifi_receiver
    fingerprint, version, ack = tls_hello(source.port, identity.token)
    assert fingerprint == identity.fingerprint
    assert version == "TLSv1.3"
    assert ack["ok"] is True and ack["name"] == identity.name


def test_tls_loopback_rejects_a_wrong_token(wifi_receiver):
    identity, source = wifi_receiver
    fingerprint, _, ack = tls_hello(source.port, "not-the-token")
    assert fingerprint == identity.fingerprint
    assert ack == {
        "protocol": p.PROTOCOL_VERSION, "receiver": ack["receiver"], "ok": False,
        "error": "not paired with this Mac", "code": "pairing", "name": identity.name,
    }


def test_tls_1_2_clients_are_refused(wifi_receiver):
    _, source = wifi_receiver
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    with socket.create_connection(("127.0.0.1", source.port), timeout=5) as raw:
        with pytest.raises((ssl.SSLError, ConnectionError)):
            context.wrap_socket(raw).close()


def test_listener_failure_is_reported(tmp_path):
    identity = wifi.load_or_create_identity(tmp_path / "state")
    hub = Receiver(tmp_path / "recordings")
    with socket.create_server(("0.0.0.0", 0)) as taken:
        source = wifi.WifiSource(hub, identity, taken.getsockname()[1], threading.Event(), advertise=False)
        source.start()
        assert source.wait_ready(3.0) is False
    hub.shutdown()
    assert "cannot listen" in hub.status().sources["wifi"]


def test_shutdown_wakes_connections_stuck_before_hello(tmp_path):
    identity = wifi.load_or_create_identity(tmp_path / "state")
    hub = Receiver(tmp_path / "recordings")
    stop = threading.Event()
    source = wifi.WifiSource(hub, identity, 0, stop, advertise=False)
    source.start()
    assert source.wait_ready(5.0)
    silent = socket.create_connection(("127.0.0.1", source.port))  # never starts TLS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    no_hello = context.wrap_socket(socket.create_connection(("127.0.0.1", source.port)))  # TLS, then nothing
    try:
        deadline = time.monotonic() + 3
        while len(source._connections) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        started = time.monotonic()
        stop.set()
        hub.shutdown()
        source.join(3.0)
        assert not source.is_alive()
        assert time.monotonic() - started < 1.5  # not the 5 s handshake or HELLO timeout
        assert not source._connections
    finally:
        silent.close()
        no_hello.close()


def test_cli_prints_the_pairing_uri_and_uses_the_state_dir(tmp_path):
    state = tmp_path / "state"
    result = subprocess.run(
        [sys.executable, "-m", "glassesdata_receiver", "--wifi", "--no-bonjour", "--print-pairing", "--state-dir", str(state),
         "--port", "0", "--no-hands", "--headless", "--duration", "0.5", "--out", str(tmp_path / "recordings")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("PAIRING_URI=")]
    assert len(lines) == 1
    assert "WEB_URL" not in result.stdout  # --headless: no web page
    query = {key: values[0] for key, values in parse_qs(urlsplit(lines[0].removeprefix("PAIRING_URI=")).query).items()}
    identity = wifi.load_or_create_identity(state)
    assert (query["fp"], query["token"]) == (identity.fingerprint, identity.token)
    assert int(query["port"]) > 0


def test_hosts_are_ordered_lan_first_cgnat_last():
    hosts = ["100.110.171.0", "8.8.8.8", "172.20.1.5", "192.168.1.251", "10.0.0.7", "100.64.0.1", "192.168.1.20",
             "172.32.0.1", "10.0.0.7"]
    assert wifi.order_hosts(hosts) == [
        "192.168.1.20", "192.168.1.251", "10.0.0.7", "172.20.1.5", "8.8.8.8", "172.32.0.1", "100.64.0.1", "100.110.171.0",
    ]
    assert wifi.order_hosts(list(reversed(hosts))) == wifi.order_hosts(hosts)
