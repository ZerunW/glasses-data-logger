import csv
import json

import pytest

from glassesdata_receiver import offline
from glassesdata_receiver.recorder import HANDS_HEADER, Recording

hands = pytest.importorskip("glassesdata_receiver.hands")


def test_offline_tracks_every_frame(tmp_path, plant):
    if not hands.DEFAULT_MODEL.exists():
        pytest.skip("hand model not downloaded yet (run the receiver once)")
    fmt, packets = plant
    recording = Recording(tmp_path, fmt, {})
    for packet in packets[:24]:
        recording.write_video(packet, 0)
    recording.close()

    assert offline.read_frame_times(recording.directory / "frames.csv") == [
        (packet.pts_us, packet.phone_time_ns) for packet in packets[:24]
    ]
    summary = offline.process(recording.directory, max_hands=2, model=hands.DEFAULT_MODEL)
    assert summary.startswith(f"{recording.directory.name}: 24 frames, ")
    with open(recording.directory / "hands_offline.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == HANDS_HEADER
    known = {(str(packet.pts_us), str(packet.phone_time_ns)) for packet in packets[:24]}
    assert all((row[0], row[1]) in known for row in rows[1:])


class OneHandTracker:
    def __init__(self, *args, **kwargs) -> None:
        self.timestamps: list[int] = []

    def detect(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        point = (0.5, 0.5, 0.0)
        return [hands.Hand("Right", 0.9, [point] * 21, [point] * 21)]

    def close(self) -> None:
        pass


def test_offline_rows_follow_frames_csv_frame_by_frame(tmp_path, plant, monkeypatch):
    fmt, packets = plant
    recording = Recording(tmp_path, fmt, {})
    for packet in packets[5:40]:  # starts mid-GOP, like a real recording
        recording.write_video(packet, 0)
    recording.close()
    monkeypatch.setattr(offline, "HandTracker", OneHandTracker)
    summary = offline.process(recording.directory, model=tmp_path / "unused.task")
    with open(recording.directory / "frames.csv", newline="", encoding="utf-8") as handle:
        frames = [(row["pts_us"], row["phone_time_ns"]) for row in csv.DictReader(handle)]
    with open(recording.directory / "hands_offline.csv", newline="", encoding="utf-8") as handle:
        rows = [(row["pts_us"], row["phone_time_ns"]) for row in csv.DictReader(handle)]
    assert len(frames) == recording.frames_written == 40 - 12
    assert rows == frames  # one hand per frame, every frame, in order
    assert f"{len(frames)} frames, {len(frames)} with hands" in summary


def test_anatomical_labels():
    assert hands.anatomical_label("Left", mirrored_input=False) == "Left"
    assert hands.anatomical_label("Left", mirrored_input=True) == "Right"
    assert hands.anatomical_label("Right", mirrored_input=True) == "Left"
    assert "non-mirrored" in hands.handedness_note(False) and "swapped" in hands.handedness_note(True)


@pytest.mark.parametrize("mirrored", [False, True])
def test_offline_records_the_handedness_convention(tmp_path, plant, monkeypatch, mirrored):
    fmt, packets = plant
    recording = Recording(tmp_path, fmt, {"handedness": hands.handedness_note(False)})
    for packet in packets[:12]:
        recording.write_video(packet, 0)
    recording.close()
    seen = {}

    class Tracker(OneHandTracker):
        def __init__(self, model, max_hands, mirrored_input):
            super().__init__()
            seen["mirrored_input"] = mirrored_input

    monkeypatch.setattr(offline, "HandTracker", Tracker)
    offline.process(recording.directory, model=tmp_path / "unused.task", mirrored_input=mirrored)
    info = json.loads((recording.directory / "info.json").read_text())
    assert seen["mirrored_input"] is mirrored
    assert info["handednessOffline"] == hands.handedness_note(mirrored)
    assert info["handedness"] == hands.handedness_note(False) and info["framesWritten"] == 12
