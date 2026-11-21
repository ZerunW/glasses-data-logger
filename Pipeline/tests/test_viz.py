import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/fabricate_sim.py (importlib mode)
from fabricate_sim import fabricate_sim  # noqa: E402

from glassesdata_pipeline.episode import Episode  # noqa: E402
from glassesdata_pipeline.runner import Runner  # noqa: E402
from glassesdata_pipeline.viz import main as view_main  # noqa: E402
from glassesdata_pipeline.viz.static import FIGURES, make_figures  # noqa: E402


def _png_ok(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=float)
    return path.stat().st_size > 5_000 and arr.std() > 5  # not blank


def test_figures_with_everything(recording_copy):
    fabricate_sim(recording_copy)
    ep = Episode(recording_copy)
    written = make_figures(ep)
    assert set(written) == set(FIGURES)
    for path in written.values():
        assert path.parent == ep.derived / "viz" and _png_ok(path)


def test_figures_with_partial_inputs(recording_copy):
    fabricate_sim(recording_copy, write=("camera_traj",))
    ep = Episode(recording_copy)
    written = make_figures(ep, with_gt=False)
    assert {"topview", "timelines", "snapshots", "overlay"} >= set(written) >= {"topview", "timelines", "overlay"}
    # nothing derived at all (and no gt): no figure, no exception
    bare = recording_copy.parent / "bare"
    shutil.copytree(recording_copy, bare, ignore=shutil.ignore_patterns("derived", "gt"))
    assert make_figures(Episode(bare), with_gt=False) == {}


def test_viz_step_and_view_command(recording_copy, capsys):
    fabricate_sim(recording_copy)
    ep = Episode(recording_copy, {"viz": {"figures": ["topview", "timelines"]}})
    report = Runner(ep).run(steps=["viz"], deps=False)
    assert report.ok, report.table()
    info = json.loads((ep.derived / "viz" / "viz.json").read_text())
    assert set(info["figures"]) == {"topview", "timelines"} and info["meta"]["step"] == "viz"
    try:
        import rerun  # noqa: F401
    except ImportError:
        assert info["rerun"].startswith("skipped")
    else:
        assert (ep.derived / "viz" / "episode.rrd").is_file()
    assert Runner(ep).run(steps=["viz"], deps=False).by_name()["viz"].status == "cached"
    capsys.readouterr()
    if not (ep.derived / "viz" / "episode.rrd").exists():
        assert view_main([str(recording_copy)]) == 2
        out = capsys.readouterr().out
        assert "episode.rrd" in out and "topview.png" in out
    fake = recording_copy / "x.rrd"
    fake.write_bytes(b"")
    assert view_main([str(fake), "--print"]) == 0
    assert "-m rerun" in capsys.readouterr().out


def test_rerun_export(recording_copy):
    pytest.importorskip("rerun")
    from glassesdata_pipeline.viz.rerun_export import export_rrd

    fabricate_sim(recording_copy)
    ep = Episode(recording_copy)
    info = export_rrd(ep, ep.derived / "viz" / "episode.rrd", {"rerun_video": "jpeg", "rerun_jpeg_stride": 8})
    assert Path(info["path"]).stat().st_size > 10_000


def test_jpeg_fallback_geometry():
    """The JPEG fallback logs the pinhole at the JPEG resolution and 2D keypoints in its pixels."""
    from glassesdata_pipeline import geometry as geo
    from glassesdata_pipeline.viz.rerun_export import jpeg_size, to_image_px

    assert jpeg_size(720, 1280, 640) == (360, 640) and jpeg_size(360, 640, 640) == (360, 640)
    cam = geo.Camera.fov_default(720, 1280, 65.0)
    small = cam.resized(*jpeg_size(720, 1280, 640))
    X = np.array([[0.05, -0.1, 0.6], [-0.2, 0.3, 1.1]])
    uv_full = cam.project(X)
    np.testing.assert_allclose(to_image_px(uv_full, (720, 1280), (360, 640)), small.project(X), atol=1e-9)
    np.testing.assert_array_equal(to_image_px(uv_full, (720, 1280), (720, 1280)), uv_full)
