import csv
import json
from dataclasses import dataclass
from datetime import datetime

import av
import pytest

from glassesdata_receiver import recorder
from glassesdata_receiver.recorder import Recording


@dataclass(frozen=True)
class FakeHand:
    handedness: str
    score: float
    landmarks: list
    world_landmarks: list


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def decoded_frame_count(path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def record(root, fmt, packets) -> Recording:
    recording = Recording(root, fmt, {"link": "test"})
    for index, packet in enumerate(packets):
        recording.write_video(packet, 1_000_000 + index)
    recording.close()
    return recording


def test_mp4_holds_every_frame_in_frames_csv(tmp_path, plant):
    fmt, packets = plant
    recording = record(tmp_path, fmt, packets)
    frames = rows(recording.directory / "frames.csv")
    assert frames[0] == recorder.FRAMES_HEADER
    assert len(frames) - 1 == len(packets) == recording.frames_written
    assert decoded_frame_count(recording.directory / "video.mp4") == len(packets)
    assert [int(row[1]) for row in frames[1:]] == [packet.pts_us for packet in packets]
    with av.open(str(recording.directory / "video.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "hevc"
        assert stream.codec_tag == "hvc1"
        assert (stream.codec_context.width, stream.codec_context.height) == (fmt.width, fmt.height)


def test_stream_starting_mid_gop_waits_for_a_keyframe(tmp_path, plant):
    fmt, packets = plant
    first_key = next(i for i, packet in enumerate(packets) if i and packet.keyframe)
    recording = record(tmp_path, fmt, packets[1:])
    frames = rows(recording.directory / "frames.csv")[1:]
    assert recording.frames_written == len(packets) - first_key
    assert frames[0][1] == str(packets[first_key].pts_us) and frames[0][4] == "1"
    assert decoded_frame_count(recording.directory / "video.mp4") == recording.frames_written


def test_non_increasing_pts_is_skipped(tmp_path, plant):
    fmt, packets = plant
    sequence = packets[:6] + [packets[5], packets[3]] + packets[6:12]
    recording = record(tmp_path, fmt, sequence)
    assert recording.frames_written == 12
    pts = [int(row[1]) for row in rows(recording.directory / "frames.csv")[1:]]
    assert pts == [packet.pts_us for packet in packets[:12]]


def test_sensor_hand_and_clock_rows(tmp_path, plant):
    fmt, packets = plant
    recording = Recording(tmp_path, fmt, {})
    hand = FakeHand("Left", 0.98765, [(0.5, 0.25, -0.1)] * 21, [(0.01, 0.02, 0.03)] * 21)
    recording.write_hands(packets[1].pts_us, 1, [hand])  # no video yet: not in video.mp4, dropped
    recording.write_video(packets[1], 0)  # not a keyframe: still nothing recorded
    recording.write_video(packets[0], 0)
    recording.write_motion(
        {"t": 1, "gt": 2, "a": [0.1, 0.2, 0.3], "g": [1, 2, 3], "m": [4, 5], "q": [1, 0, 0, 0], "src": "glasses"}, 99
    )
    recording.write_input({"t": 3, "gt_ms": 4, "src": "glasses", "type": "tap", "detail": "a,b"}, 100)
    recording.write_device_state({"t": 5, "battery": 80, "charging": "no", "don": "donned", "hinge": "open"}, 101)
    recording.write_clock(1000, 1600, 5300, 5000)
    recording.write_hands(40_000, 7, [hand, hand])
    recording.close()
    recording.write_motion({"t": 9}, 1)  # after close: dropped, not raised
    folder = recording.directory

    motion = rows(folder / "motion.csv")
    assert motion[0] == recorder.MOTION_HEADER
    assert motion[1] == ["1", "2", "0.1", "0.2", "0.3", "1", "2", "3", "", "", "", "1", "0", "0", "0", "glasses", "99"]
    assert len(motion) == 2
    assert rows(folder / "inputs.csv")[1] == ["3", "4", "glasses", "tap", "a,b", "100"]
    assert rows(folder / "device_state.csv")[1] == ["5", "80", "no", "donned", "open", "", "", "101"]
    assert rows(folder / "clock.csv")[1] == ["1000", "1600", "5300", "600", "5000"]
    hands = rows(folder / "hands.csv")
    assert hands[0] == recorder.HANDS_HEADER and len(hands[0]) == 5 + 2 * 63
    assert [row[:5] for row in hands[1:]] == [["40000", "7", "0", "Left", "0.9877"], ["40000", "7", "1", "Left", "0.9877"]]
    assert hands[1][5:8] == ["0.5", "0.25", "-0.1"] and hands[1][-3:] == ["0.01", "0.02", "0.03"]


def test_info_json_has_end_fields(tmp_path, plant):
    fmt, packets = plant
    recording = Recording(tmp_path, fmt, {"link": "usb", "device": "Test iPhone"})
    started = json.loads((recording.directory / "info.json").read_text())
    assert "endedAt" not in started and started["device"] == "Test iPhone"
    for packet in packets[:24]:
        recording.write_video(packet, 0)
    recording.close()
    info = json.loads((recording.directory / "info.json").read_text())
    assert info["framesWritten"] == 24
    assert info["firstPtsUs"] == packets[0].pts_us and info["lastPtsUs"] == packets[23].pts_us
    assert info["handRows"] == 0
    assert info["endedMacTimeNs"] >= info["startedMacTimeNs"]
    assert datetime.fromisoformat(info["endedAt"]) >= datetime.fromisoformat(info["startedAt"])
    assert info["video"] == {"width": fmt.width, "height": fmt.height, "configuredFps": fmt.fps}
    assert info["link"] == "usb"


def test_recordings_in_the_same_second_get_distinct_folders(tmp_path, plant, monkeypatch):
    fmt, _ = plant
    fixed = datetime(2026, 1, 2, 3, 4, 5)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(recorder, "datetime", FrozenDatetime)
    first, second, third = (Recording(tmp_path, fmt, {}) for _ in range(3))
    for recording in (first, second, third):
        recording.close()
    assert [r.directory.name for r in (first, second, third)] == [
        "2026-01-02_030405", "2026-01-02_030405_2", "2026-01-02_030405_3",
    ]


def test_hand_rows_match_header():
    hand = FakeHand("Right", 0.5, [(0.0, 0.0, 0.0)] * 21, [(0.0, 0.0, 0.0)] * 21)
    (row,) = recorder.hand_rows(1, 2, [hand])
    assert len(row) == len(recorder.HANDS_HEADER)


@pytest.mark.parametrize("bad", [[1, 2], "abc", None])
def test_malformed_vectors_become_empty_cells(tmp_path, plant, bad):
    fmt, _ = plant
    recording = Recording(tmp_path, fmt, {})
    recording.write_motion({"t": 1, "a": bad}, 2)
    recording.close()
    assert rows(recording.directory / "motion.csv")[1][2:5] == ["", "", ""]
