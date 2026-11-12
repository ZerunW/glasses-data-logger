#!/usr/bin/env python3
"""Check actual IUCM frames across the TLS relay before and after approval."""
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix='glasses-wirecheck-') as temporary:
    work = Path(temporary)
    text = (root / 'MacRelay/main.swift').read_text()
    relay = text.split('final class Relay: @unchecked Sendable {', 1)[1].split('\nstruct ContentView:', 1)[0]
    (work / 'Relay.swift').write_text('import Foundation\nimport Network\n\nfinal class Relay: @unchecked Sendable {' + relay)
    shutil.copy(root / 'SecurityTests/WirelessInteropSender.swift', work / 'main.swift')
    sources = ['CameraAccess/Streaming/SecureTransport.swift', 'CameraAccess/Streaming/CameraBroadcast.swift',
               'CameraAccess/Streaming/IucmMessage.swift', 'CameraAccess/Streaming/IucmFrameParser.swift',
               'CameraAccess/Recording/MonotonicClock.swift', 'CameraAccess/Media/VideoCaptureHandler.swift']
    sender = work / 'sender'
    subprocess.run(['xcrun', 'swiftc', '-suppress-warnings', '-module-cache-path', str(work / 'cache'),
                    *[str(root / source) for source in sources], str(work / 'Relay.swift'),
                    str(work / 'main.swift'), '-o', str(sender)], check=True)
    process = subprocess.Popen([str(sender), str(root / 'CameraAccess/TestResources/plant.mp4')],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert 'Ready for encrypted relay' in process.stdout.readline(), 'Relay did not become ready'
        with socket.create_connection(('127.0.0.1', 7879), timeout=5) as peer:
            peer.settimeout(.25)
            buffer = bytearray()
            hello_at = None
            saw_config = False
            saw_video = False
            deadline = time.monotonic() + 13
            while time.monotonic() < deadline:
                try:
                    chunk = peer.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    raise AssertionError('Connection closed before a video frame')
                buffer.extend(chunk)
                while len(buffer) >= 12:
                    assert buffer[:4] == b'IUCM', 'Invalid message header'
                    kind = buffer[4]
                    size = struct.unpack_from('<I', buffer, 8)[0]
                    assert size <= 8 * 1024 * 1024
                    if len(buffer) < 12 + size:
                        break
                    del buffer[:12 + size]
                    if kind == 1:
                        assert hello_at is None
                        hello_at = time.monotonic()
                        start = struct.pack('<BHHHI', 0, 1280, 720, 24, 4000)
                        peer.sendall(b'IUCM' + bytes((2, 0, 0, 0)) + struct.pack('<I', len(start)) + start)
                    elif kind == 0x10:
                        assert hello_at is not None
                        saw_config = True
                    elif kind == 0x11:
                        assert hello_at is not None and saw_config
                        assert time.monotonic() - hello_at >= 5, 'Video arrived before delayed approval'
                        saw_video = True
                        break
                if saw_video:
                    break
            assert hello_at is not None and saw_config and saw_video, 'HELLO, CONFIG, and approved VIDEO are required'
            print('PASS: actual TLS-relayed CONFIG arrived while video stayed gated; VIDEO arrived after approval')
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
