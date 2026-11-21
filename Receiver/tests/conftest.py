import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glassesdata_receiver import protocol as p  # noqa: E402

# HEVC, 540x960, 30 fps, 266 frames, no B-frames, a keyframe every 12 frames.
PLANT = Path(__file__).resolve().parents[2] / "CameraAccess" / "TestResources" / "plant.mp4"


def load_video(path: Path) -> tuple[p.VideoFormat, list[p.VideoPacket]]:
    """The packets the iPhone would send for this file (phone_time_ns is a stand-in)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        context = stream.codec_context
        fmt = p.VideoFormat(context.width, context.height, round(float(stream.average_rate)), bytes(context.extradata))
        packets = [
            p.VideoPacket(bool(packet.is_keyframe), int(packet.pts * stream.time_base * 1_000_000), 10_000 + index, bytes(packet))
            for index, packet in enumerate(packet for packet in container.demux(stream) if packet.size)
        ]
    return fmt, packets


@pytest.fixture(scope="session")
def plant() -> tuple[p.VideoFormat, list[p.VideoPacket]]:
    if not PLANT.exists():
        pytest.skip(f"{PLANT} is missing")
    return load_video(PLANT)
