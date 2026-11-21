#!/usr/bin/env python3
"""Pretends to be the GlassesData iPhone app, so the receiver can be tested without hardware.

USB style (the receiver connects to us, like it does through usbmuxd)::

    python tools/fake_sender.py VIDEO.mp4 --listen 7901
    python -m glassesdata_receiver --tcp 127.0.0.1:7901

Wi-Fi style (we connect to the receiver over TLS with certificate pinning)::

    python tools/fake_sender.py VIDEO.mp4 --connect 127.0.0.1:7900 --fingerprint HEX --token TOKEN

Remote record control, like the app's Record button: ``--record-after 2 --record-for 5``
sends CONTROL record_start 2 s after the connection and record_stop 5 s later. Every
RECORDER_STATE from the receiver is printed as ``RECORDER_STATE {json}``.

STATS carries ``queue_frames``, ``record_priority`` (true while the receiver says it is
recording, like the app, which then buffers instead of dropping) and ``gaps`` (drop
episodes). ``--lag-ms MS`` stamps every frame MS earlier, as if the phone had queued it
that long: the receiver's preview then runs MS behind.

VIDEO must hold an HEVC track without B-frames, like the glasses stream.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import select
import socket
import ssl
import sys
import time
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glassesdata_receiver import protocol as p  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001 - tests and scripts read our output through a pipe


def load_video(path: Path) -> tuple[p.VideoFormat, list[tuple[bytes, int, bool]]]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    if stream.codec_context.name != "hevc":
        raise SystemExit(f"{path} is {stream.codec_context.name}, expected hevc")
    fmt = p.VideoFormat(stream.codec_context.width, stream.codec_context.height, round(float(stream.average_rate)), bytes(stream.codec_context.extradata))
    packets = [(bytes(pk), int(pk.pts * stream.time_base * 1_000_000), bool(pk.is_keyframe)) for pk in container.demux(stream) if pk.size]
    return fmt, packets


def serve(sock: socket.socket, fmt: p.VideoFormat, packets: list[tuple[bytes, int, bool]], args: argparse.Namespace) -> None:
    clock_skew_ns = args.clock_skew_ms * 1_000_000  # the phone's monotonic clock is unrelated to the Mac's

    def phone_now() -> int:
        return time.monotonic_ns() + clock_skew_ns

    hello = {"protocol": p.PROTOCOL_VERSION, "app": "fake-sender", "device": "iPhone (simulated)", "name": "Fake iPhone"}
    if args.token:
        hello["token"] = args.token
    sock.sendall(p.encode_json(p.MessageType.HELLO, hello))
    reader = p.FrameReader()
    sock.sendall(p.encode_video_format(fmt))
    sock.sendall(p.encode_json(p.MessageType.DEVICE_STATE, {
        "t": phone_now(), "battery": 80, "charging": "notCharging", "don": "donned", "hinge": "open", "thermal": "normal", "link": "connected"}))

    frame_interval = 1 / (args.fps or fmt.fps)
    duration_us = packets[-1][1] + int(frame_interval * 1e6)
    started = time.monotonic()
    sent = dropped = motion = index = gaps = 0
    receiver_recording = False
    lag_ns = int(args.lag_ms * 1_000_000)
    next_frame = next_motion = next_stats = started
    loops = 0
    # Remote record control: (time, command) pairs still to send.
    controls: list[tuple[float, str]] = []
    if args.record_after is not None:
        controls.append((started + args.record_after, p.CONTROL_RECORD_START))
        if args.record_for is not None:
            controls.append((started + args.record_after + args.record_for, p.CONTROL_RECORD_STOP))
    while True:
        now = time.monotonic()
        if args.seconds and now - started > args.seconds:
            return
        while controls and now >= controls[0][0]:
            _, command = controls.pop(0)
            print(f"CONTROL {command}")
            sock.sendall(p.encode_json(p.MessageType.CONTROL, {"t": phone_now(), "cmd": command}))
        if now >= next_frame:
            data, pts, keyframe = packets[index]
            if args.drop_every and sent and (sent + dropped) % args.drop_every == 0 and not keyframe:
                dropped += 1
                gaps += 1
            else:
                sock.sendall(p.encode_video(p.VideoPacket(keyframe, pts + loops * duration_us, phone_now() - lag_ns, data)))
                sent += 1
            index += 1
            if index == len(packets):
                if not args.loop:
                    for _, command in controls:  # the video ended first: send what is left now
                        print(f"CONTROL {command} (the video ended early; use --loop for longer runs)")
                        sock.sendall(p.encode_json(p.MessageType.CONTROL, {"t": phone_now(), "cmd": command}))
                    time.sleep(0.5)
                    return
                index, loops = 0, loops + 1
            next_frame += frame_interval
        if now >= next_motion:
            t = now - started
            sock.sendall(p.encode_json(p.MessageType.MOTION, {
                "t": phone_now(), "gt": int(t * 1e9), "src": "glasses",
                "a": [round(math.sin(t), 4), 9.81, round(math.cos(t), 4)], "g": [0.01, round(0.2 * math.sin(2 * t), 4), 0.0],
                "m": [20.0, -5.0, 40.0], "q": [1.0, 0.0, 0.0, 0.0]}))
            motion += 1
            next_motion += 1 / 30
        if now >= next_stats:
            sock.sendall(p.encode_json(p.MessageType.STATS, {
                "t": phone_now(), "glasses_frames": sent + dropped, "sent_frames": sent, "dropped_frames": dropped,
                "motion_samples": motion, "queue_bytes": 0, "queue_frames": round(args.lag_ms / 1000 * (args.fps or fmt.fps)),
                "record_priority": receiver_recording, "gaps": gaps}))
            next_stats += 1.0
        due = [next_frame, next_motion, next_stats] + [at for at, _ in controls[:1]]
        timeout = max(0.0, min(due) - time.monotonic())
        pending = sock.pending() if isinstance(sock, ssl.SSLSocket) else 0
        readable = [sock] if pending else select.select([sock], [], [], timeout)[0]
        if readable:
            data = sock.recv(65536)
            if not data:
                print("receiver closed the connection")
                return
            for message_type, payload in reader.feed(data):
                message = p.decode_message(message_type, payload)
                if message is None:
                    continue
                if message.type is p.MessageType.PING:
                    sock.sendall(p.encode_pong(p.Pong(message.body.mac_time_ns, phone_now())))
                elif message.type is p.MessageType.HELLO_ACK:
                    print("receiver:", message.body)
                    if not message.body.get("ok"):
                        return
                elif message.type is p.MessageType.RECORDER_STATE:
                    receiver_recording = bool(message.body.get("recording"))
                    print("RECORDER_STATE", json.dumps(message.body, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--listen", type=int, metavar="PORT", help="wait for the receiver on 127.0.0.1:PORT (0: any free port)")
    parser.add_argument("--connect", metavar="HOST:PORT", help="connect to a --wifi receiver over TLS")
    parser.add_argument("--fingerprint", help="expected SHA-256 certificate fingerprint (hex)")
    parser.add_argument("--token", help="pairing token")
    parser.add_argument("--loop", action="store_true", help="repeat the video with continuous timestamps")
    parser.add_argument("--seconds", type=float, help="stop after this long")
    parser.add_argument("--fps", type=float, help="override the frame rate")
    parser.add_argument("--drop-every", type=int, default=0, help="simulate the phone dropping every Nth frame")
    parser.add_argument("--clock-skew-ms", type=int, default=123_456, help="offset of the simulated phone clock")
    parser.add_argument("--lag-ms", type=float, default=0.0, help="stamp every frame this much earlier (a phone that queued it)")
    parser.add_argument("--record-after", type=float, metavar="N", help="send CONTROL record_start N seconds after connecting")
    parser.add_argument("--record-for", type=float, metavar="M", help="with --record-after: send record_stop M seconds later")
    args = parser.parse_args()
    if args.record_for is not None and args.record_after is None:
        parser.error("--record-for needs --record-after")
    fmt, packets = load_video(args.video)
    print(f"{args.video.name}: {fmt.width}x{fmt.height} @ {fmt.fps} fps, {len(packets)} frames")

    if args.connect:
        host, _, port = args.connect.rpartition(":")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # trust comes from the pinned fingerprint below
        raw = socket.create_connection((host, int(port)), timeout=5)
        sock = context.wrap_socket(raw)
        actual = hashlib.sha256(sock.getpeercert(binary_form=True)).hexdigest()
        if args.fingerprint and actual != args.fingerprint.lower():
            print(f"certificate fingerprint mismatch: {actual}")
            return 2
        print("TLS", sock.version(), "fingerprint ok" if args.fingerprint else f"fingerprint {actual} (not checked)")
        serve(sock, fmt, packets, args)
        sock.close()
        return 0

    with socket.create_server(("127.0.0.1", p.DEFAULT_PORT if args.listen is None else args.listen)) as server:
        print(f"waiting for the receiver on 127.0.0.1:{server.getsockname()[1]}")
        connection, _ = server.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            serve(connection, fmt, packets, args)
        except (BrokenPipeError, ConnectionResetError):
            print("receiver disconnected")
        finally:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
