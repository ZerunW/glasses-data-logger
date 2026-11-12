#!/usr/bin/env python3
"""Local TLS security checks. All keys and payloads are public fixtures."""
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = b"GLASSESDATA_SECRET_TEST_FRAME"
FIXTURE_KEY = "42" * 32

def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def run():
    with tempfile.TemporaryDirectory(prefix="glasses-security-") as tmp:
        executable = Path(tmp) / "server"
        source = Path(tmp) / "main.swift"
        source.write_bytes((ROOT / "SecurityTests/TLSTestServer.swift").read_bytes())
        subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(Path(tmp) / "cache"),
                        str(ROOT / "CameraAccess/Streaming/SecureTransport.swift"),
                        str(source), "-o", str(executable)], check=True)
        port = free_port()
        server = subprocess.Popen([str(executable), str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            ready, _, _ = select.select([server.stdout], [], [], 10)
            if not ready or server.stdout.readline().strip() != b"READY":
                raise RuntimeError("TLS test server did not start")
            wire = bytearray()
            proxy_port = free_port()
            listening = socket.socket()
            listening.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listening.bind(("127.0.0.1", proxy_port))
            listening.listen(1)
            def proxy(tamper=False):
                client, _ = listening.accept()
                remote = socket.create_connection(("127.0.0.1", port), timeout=5)
                pending = bytearray()
                with client, remote:
                    client.settimeout(5)
                    remote.settimeout(5)
                    while True:
                        readable, _, _ = select.select([client, remote], [], [], 5)
                        if not readable:
                            break
                        for source in readable:
                            chunk = source.recv(65536)
                            if not chunk:
                                return
                            wire.extend(chunk)
                            if tamper and source is remote:
                                pending.extend(chunk)
                                while len(pending) >= 5:
                                    record_length = 5 + int.from_bytes(pending[3:5], "big")
                                    if len(pending) < record_length:
                                        break
                                    record = bytearray(pending[:record_length])
                                    del pending[:record_length]
                                    if record[0] == 23 and not tampered.is_set():
                                        record[-1] ^= 1
                                        tampered.set()
                                    client.sendall(record)
                            else:
                                (remote if source is client else client).sendall(chunk)
            worker = threading.Thread(target=proxy, daemon=True)
            worker.start()
            def client(p, key):
                return subprocess.run(["openssl", "s_client", "-connect", f"127.0.0.1:{p}",
                    "-tls1_2", "-cipher", "PSK-AES128-GCM-SHA256", "-psk", key,
                    "-psk_identity", "GlassesData-TetherCam-v1", "-quiet"],
                    input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            good = client(proxy_port, FIXTURE_KEY)
            worker.join(timeout=6)
            listening.close()
            assert MARKER in good.stdout, "Correct key did not receive test frame"
            assert MARKER not in wire, "Frame appeared in captured TLS traffic"
            tampered = threading.Event()
            proxy_port = free_port()
            listening = socket.socket()
            listening.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listening.bind(("127.0.0.1", proxy_port))
            listening.listen(1)
            worker = threading.Thread(target=lambda: proxy(tamper=True), daemon=True)
            worker.start()
            altered = client(proxy_port, FIXTURE_KEY)
            worker.join(timeout=6)
            listening.close()
            assert tampered.is_set(), "Test did not alter a TLS application record"
            assert MARKER not in altered.stdout, "Tampered TLS record was accepted"
            bad = client(port, "43" * 32)
            assert MARKER not in bad.stdout, "Incorrect key received the test frame"
            with socket.create_connection(("127.0.0.1", port), timeout=5) as plain:
                plain.settimeout(3)
                plain.sendall(b"IUCM plaintext request")
                try:
                    received = plain.recv(4096)
                except (OSError, socket.timeout):
                    received = b""
                assert MARKER not in received, "Plaintext client received the test frame"
            print("PASS: valid key, wrong key, tampered record, plaintext rejection, and captured-wire confidentiality")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

if __name__ == "__main__":
    try:
        run()
    except (AssertionError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
