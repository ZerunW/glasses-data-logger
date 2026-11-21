import pytest

from glassesdata_receiver import hevc
from glassesdata_receiver.decoder import HevcDecoder

VPS, SPS, PPS = 32, 33, 34


def nal_type(unit: bytes) -> int:
    return (unit[0] >> 1) & 0x3F


def test_hvcc_parameter_sets_from_plant(plant):
    fmt, _ = plant
    units = hevc.hvcc_parameter_sets(fmt.hvcc)
    assert [nal_type(unit) for unit in units] == [VPS, SPS, PPS]
    annexb = hevc.hvcc_to_annexb(fmt.hvcc)
    assert annexb == b"".join(hevc.START_CODE + unit for unit in units)


def test_length_prefixed_to_annexb_on_a_keyframe(plant):
    _, packets = plant
    keyframe = packets[0]
    units = hevc.split_length_prefixed(keyframe.data)
    assert units and all(units)
    assert sum(4 + len(unit) for unit in units) == len(keyframe.data)
    annexb = hevc.length_prefixed_to_annexb(keyframe.data)
    assert annexb.startswith(hevc.START_CODE)
    assert len(annexb) == len(keyframe.data)  # 4-byte lengths become 4-byte start codes
    assert annexb.split(hevc.START_CODE)[1:] == units
    assert {19, 20} & {nal_type(unit) for unit in units}  # IDR_W_RADL or IDR_N_LP


def test_split_length_prefixed_with_two_byte_lengths():
    data = b"\x00\x02ab" + b"\x00\x01c"
    assert hevc.split_length_prefixed(data, length_size=2) == [b"ab", b"c"]


@pytest.mark.parametrize(
    "data",
    [
        b"\x00\x00\x00",  # truncated length
        b"\x00\x00\x00\x00",  # zero length
        b"\x00\x00\x00\x05abc",  # runs past the end
        b"\x00\x00\x00\x01a\x00\x00",  # second length truncated
    ],
)
def test_split_length_prefixed_rejects_malformed(data):
    with pytest.raises(hevc.HevcFormatError):
        hevc.split_length_prefixed(data)


def test_hvcc_rejects_malformed(plant):
    fmt, _ = plant
    good = fmt.hvcc
    with pytest.raises(hevc.HevcFormatError):
        hevc.hvcc_parameter_sets(good[:22])  # shorter than the fixed header
    with pytest.raises(hevc.HevcFormatError):
        hevc.hvcc_parameter_sets(b"\x02" + good[1:])  # configurationVersion must be 1
    with pytest.raises(hevc.HevcFormatError):
        hevc.hvcc_parameter_sets(good[:-3])  # last NAL unit truncated
    with pytest.raises(hevc.HevcFormatError):
        hevc.hvcc_parameter_sets(good[:24])  # array header cut inside numNalus
    with pytest.raises(hevc.HevcFormatError):
        hevc.hvcc_parameter_sets(good[:22] + b"\x00")  # no arrays at all


def test_decoder_numbers_frames_across_decoders(plant):
    fmt, packets = plant
    first = HevcDecoder(fmt)
    frames = [frame for packet in packets[:12] for frame in first.decode(packet)]
    assert [frame.index for frame in frames] == list(range(len(frames)))
    assert frames[0].image.shape == (fmt.height, fmt.width, 3)
    second = HevcDecoder(fmt, first.next_index)
    more = [frame for packet in packets[12:14] for frame in second.decode(packet)]
    assert more and more[0].index == first.next_index
    assert first.errors == second.errors == 0
