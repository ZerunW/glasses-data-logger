import pytest

from glassesdata_receiver import protocol as p

# Byte fixtures shared with CameraAccessTests/LinkProtocolTests.swift. Change both together.
VIDEO_FIXTURE = bytes.fromhex(
    "0000001a"  # length = 1 type + 17 header + 8 data
    "11"  # VIDEO
    "01"  # keyframe
    "fffffffffffffc18"  # pts_us = -1000
    "000000000000007b"  # phone_time_ns = 123
    "00000004deadbeef"  # one 4-byte NAL unit
)
VIDEO_FORMAT_FIXTURE = bytes.fromhex("0000000a" "10" "02d0" "0500" "0018" "0102" "03")
PONG_FIXTURE = bytes.fromhex("00000011" "41" "0000000000000001" "0000000000000002")
STATS_FIXTURE = bytes.fromhex("00000014") + b'\x30{"dropped":0,"t":5}'  # 1 type + 19 JSON bytes


def test_video_fixture_round_trip():
    packet = p.VideoPacket(True, -1000, 123, bytes.fromhex("00000004deadbeef"))
    assert p.encode_video(packet) == VIDEO_FIXTURE
    frames = list(p.FrameReader().feed(VIDEO_FIXTURE))
    assert p.decode_message(*frames[0]) == p.Message(p.MessageType.VIDEO, packet)


def test_video_format_fixture():
    fmt = p.VideoFormat(720, 1280, 24, bytes.fromhex("010203"))
    assert p.encode_video_format(fmt) == VIDEO_FORMAT_FIXTURE
    assert p.decode_message(*next(p.FrameReader().feed(VIDEO_FORMAT_FIXTURE))).body == fmt


def test_pong_and_json_fixtures():
    assert p.encode_pong(p.Pong(1, 2)) == PONG_FIXTURE
    assert p.encode_json(p.MessageType.STATS, {"t": 5, "dropped": 0}) == STATS_FIXTURE
    message = p.decode_message(*next(p.FrameReader().feed(STATS_FIXTURE)))
    assert message.body == {"t": 5, "dropped": 0}


def test_reader_handles_fragments_and_multiple_frames():
    stream = VIDEO_FIXTURE + PONG_FIXTURE + VIDEO_FORMAT_FIXTURE
    reader = p.FrameReader()
    frames = []
    for byte in stream:
        frames.extend(reader.feed(bytes([byte])))
    assert [t for t, _ in frames] == [p.MessageType.VIDEO, p.MessageType.PONG, p.MessageType.VIDEO_FORMAT]
    assert reader.pending_bytes == 0


@pytest.mark.parametrize("length", [0, p.MAX_FRAME_LENGTH + 1])
def test_reader_rejects_bad_lengths(length):
    with pytest.raises(p.ProtocolError):
        list(p.FrameReader().feed(length.to_bytes(4, "big") + b"\x11"))


def test_unknown_type_is_ignored_and_truncated_payload_rejected():
    assert p.decode_message(0x7F, b"anything") is None
    with pytest.raises(p.ProtocolError):
        p.decode_message(p.MessageType.VIDEO, b"\x01\x00")
    with pytest.raises(p.ProtocolError):
        p.decode_message(p.MessageType.PONG, bytes(15))
    with pytest.raises(p.ProtocolError):
        p.decode_message(p.MessageType.MOTION, b"[1,2]")


def test_encode_json_rejects_binary_types():
    with pytest.raises(p.ProtocolError):
        p.encode_json(p.MessageType.VIDEO, {})


# Remote record control. Byte fixtures for the iOS side (LinkProtocolTests.swift should
# decode CONTROL_FIXTURE's layout from the app and RECORDER_STATE_FIXTURE from the receiver).
CONTROL_FIXTURE = bytes.fromhex("00000025") + b'\x50{"cmd":"record_start","t":123456789}'
RECORDER_STATE_FIXTURE = (
    bytes.fromhex("00000054")  # 1 type + 83 JSON bytes
    + b'\x51{"frames":42,"name":"2026-11-20_212051","note":null,"recording":true,"seconds":1.5}'
)


def test_control_and_recorder_state_types():
    assert p.MessageType.CONTROL == 0x50 and p.MessageType.RECORDER_STATE == 0x51
    assert p.MessageType.CONTROL in p.JSON_TYPES and p.MessageType.RECORDER_STATE in p.JSON_TYPES
    assert (p.CONTROL_RECORD_START, p.CONTROL_RECORD_STOP) == ("record_start", "record_stop")


def test_control_fixture_round_trip():
    body = {"t": 123456789, "cmd": "record_start"}
    assert p.encode_json(p.MessageType.CONTROL, body) == CONTROL_FIXTURE
    message = p.decode_message(*next(p.FrameReader().feed(CONTROL_FIXTURE)))
    assert message == p.Message(p.MessageType.CONTROL, body)


def test_recorder_state_fixture_round_trip():
    body = {"recording": True, "name": "2026-11-20_212051", "seconds": 1.5, "frames": 42, "note": None}
    assert p.encode_json(p.MessageType.RECORDER_STATE, body) == RECORDER_STATE_FIXTURE
    message = p.decode_message(*next(p.FrameReader().feed(RECORDER_STATE_FIXTURE)))
    assert message == p.Message(p.MessageType.RECORDER_STATE, body)
