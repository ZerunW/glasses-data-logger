"""HEVC bitstream helpers: hvcC parsing and length-prefixed <-> Annex-B conversion."""

from __future__ import annotations

import struct

START_CODE = b"\x00\x00\x00\x01"


class HevcFormatError(ValueError):
    pass


def hvcc_parameter_sets(hvcc: bytes) -> list[bytes]:
    """Return the VPS/SPS/PPS NAL units stored in an HEVCDecoderConfigurationRecord."""
    if len(hvcc) < 23 or hvcc[0] != 1:
        raise HevcFormatError("not an hvcC record")
    count = hvcc[22]
    offset = 23
    units: list[bytes] = []
    try:
        for _ in range(count):
            offset += 1  # array_completeness | reserved | NAL_unit_type
            (nal_count,) = struct.unpack_from(">H", hvcc, offset)
            offset += 2
            for _ in range(nal_count):
                (length,) = struct.unpack_from(">H", hvcc, offset)
                offset += 2
                if offset + length > len(hvcc):
                    raise HevcFormatError("truncated hvcC NAL unit")
                units.append(hvcc[offset : offset + length])
                offset += length
    except struct.error as error:
        raise HevcFormatError("truncated hvcC record") from error
    if not units:
        raise HevcFormatError("hvcC holds no parameter sets")
    return units


def split_length_prefixed(data: bytes, length_size: int = 4) -> list[bytes]:
    units: list[bytes] = []
    offset = 0
    while offset < len(data):
        if offset + length_size > len(data):
            raise HevcFormatError("truncated NAL length")
        length = int.from_bytes(data[offset : offset + length_size], "big")
        offset += length_size
        if length == 0 or offset + length > len(data):
            raise HevcFormatError("invalid NAL length")
        units.append(data[offset : offset + length])
        offset += length
    return units


def to_annexb(units: list[bytes]) -> bytes:
    return b"".join(START_CODE + unit for unit in units)


def length_prefixed_to_annexb(data: bytes) -> bytes:
    return to_annexb(split_length_prefixed(data))


def hvcc_to_annexb(hvcc: bytes) -> bytes:
    return to_annexb(hvcc_parameter_sets(hvcc))
