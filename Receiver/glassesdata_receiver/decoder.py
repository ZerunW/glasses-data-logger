"""HEVC decoding with PyAV (FFmpeg)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import av
import numpy as np

from .protocol import VideoFormat, VideoPacket

MICROSECONDS = Fraction(1, 1_000_000)


@dataclass(frozen=True)
class DecodedFrame:
    image: np.ndarray  # BGR, height x width x 3
    pts_us: int
    phone_time_ns: int
    index: int


class HevcDecoder:
    """Decodes length-prefixed HEVC samples for one VideoFormat.

    Every packet must be decoded in order (P-frames reference earlier frames), so the
    caller feeds all packets and only drops decoded images it cannot display in time.
    """

    def __init__(self, fmt: VideoFormat, first_index: int = 0) -> None:
        self.format = fmt
        self._context = av.CodecContext.create("hevc", "r")
        self._context.extradata = fmt.hvcc
        self._context.thread_type = "SLICE"  # frame threading would add frames of latency
        self._index = first_index  # lets a new decoder continue the previous one's numbering
        self.errors = 0

    @property
    def next_index(self) -> int:
        return self._index

    def decode(self, packet: VideoPacket) -> list[DecodedFrame]:
        av_packet = av.Packet(packet.data)
        av_packet.pts = packet.pts_us
        av_packet.time_base = MICROSECONDS
        try:
            frames = self._context.decode(av_packet)
        except av.error.InvalidDataError:
            self.errors += 1
            return []
        decoded: list[DecodedFrame] = []
        for frame in frames:
            decoded.append(
                DecodedFrame(frame.to_ndarray(format="bgr24"), packet.pts_us, packet.phone_time_ns, self._index)
            )
            self._index += 1
        return decoded
