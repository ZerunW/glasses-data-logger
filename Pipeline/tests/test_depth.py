"""Step depth: gt backend through the runner (memmap, index, stride, cache), conversions, missing models."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.backends import depth_anything
from glassesdata_pipeline.backends import gt as gtb
from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.runner import Runner
from glassesdata_pipeline.steps import depth as depth_step


def test_gt_backend_memmap_index_and_cache(recording_copy):
    ep = Episode(recording_copy)
    report = Runner(ep).run(until="depth")
    assert report.ok, report.table()
    maps = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    index = ep.load_npz("depth/index")
    T = len(ep.timeline()["t"])
    cam = ep.camera()
    w, h = geo.downsample_size(cam.width, cam.height, 4)
    assert maps.dtype == np.float16 and maps.shape == (T, h, w)
    assert list(index["frames"]) == list(range(T)) and np.isclose(index["scale"], h / cam.height)
    assert index["meta"]["backend"] == "gt"
    expected = gtb.depth(ep, ep.config, frames=[0, T - 1])["depth"]
    assert np.array_equal(np.asarray(maps[[0, T - 1]]), expected)
    assert not list((ep.derived / "depth").glob("*.partial"))
    assert Runner(Episode(recording_copy)).run(steps=["depth"]).by_name()["depth"].status == "cached"


def test_stride_and_downsample(recording_copy):
    ep = Episode(recording_copy, {"depth": {"stride": 5, "downsample": 2}})
    assert Runner(ep).run(until="depth").ok
    maps = np.load(ep.derived / "depth" / "depth_f16.npy", mmap_mode="r")
    frames = ep.load_npz("depth/index")["frames"]
    T = len(ep.timeline()["t"])
    assert list(frames) == list(range(0, T, 5))
    cam = ep.camera()
    assert maps.shape == (len(frames),) + geo.downsample_size(cam.width, cam.height, 2)[::-1]


def test_ray_distance_conversion():
    cam = geo.Camera.fov_default(64, 48, 70.0)
    rays = cam.pixel_rays()
    z = np.full((48, 64), 2.0)
    distance = z * np.linalg.norm(rays, axis=-1)
    assert np.allclose(depth_step.ray_to_z(distance, rays), z)
    assert list(depth_step.frame_indices(10, 3)) == [0, 3, 6, 9]


def test_resize_depth():
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert depth_anything.resize_depth(d, (4, 3)) is not None and depth_anything.resize_depth(d, (4, 3)).shape == (3, 4)
    big = np.ones((924, 518), np.float32) * 1.5
    small = depth_anything.resize_depth(big, (180, 320))
    assert small.shape == (320, 180) and np.allclose(small, 1.5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed: the model would load")
@pytest.mark.parametrize("backend", ["depth_anything_v2", "video_depth_anything"])
def test_model_backends_report_missing_torch(recording_copy, backend):
    ep = Episode(recording_copy, {"depth": {"backend": backend}})
    report = Runner(ep).run(until="depth")
    result = report.by_name()["depth"]
    assert result.status == "skipped" and "pip install torch" in result.reason and report.ok
    assert not (ep.derived / "depth" / "depth_f16.npy").exists()
    assert not list((ep.derived / "depth").glob("*.partial"))
    report = Runner(ep).run(steps=["depth"])  # named explicitly: an error
    assert report.by_name()["depth"].status == "failed" and not report.ok


def test_unknown_backend(recording_copy):
    ep = Episode(recording_copy, {"depth": {"backend": "magic"}})
    result = Runner(ep).run(until="depth").by_name()["depth"]
    assert result.status == "failed" and "magic" in result.reason
