"""Step ``viz``: static figures ``derived/viz/*.png`` (always), the Rerun recording
``derived/viz/episode.rrd`` (when ``rerun-sdk`` is installed and ``viz.rerun`` allows it) and
``derived/viz/viz.json`` (what was written; carries the meta for the runner's cache).

Works with whatever derived files exist (it only requires ``ingest``); see
:mod:`glassesdata_pipeline.viz.static` for the figures and
:mod:`glassesdata_pipeline.viz.rerun_export` for the ``.rrd``. ``python -m glassesdata_pipeline
view REC`` opens the ``.rrd``.
"""

from __future__ import annotations

import importlib.util
import logging

from ..episode import Episode
from ..errors import MissingDependency, PipelineError
from ..viz import RERUN_PIN, RRD_NAME
from ..viz.static import FIGURES, make_figures

NAME = "viz"
VERSION = 1
REQUIRES = ["ingest"]
INPUTS = [
    "derived/timeline.npz", "derived/camera.json", "derived/camera_traj.npz", "derived/hands_world.npz",
    "derived/hands2d.npz", "derived/arms.npz", "derived/objects/index.json", "derived/objects/table.json",
    "derived/objects/*/poses.npz", "derived/objects/*/mesh.obj", "derived/objects/*/collision/*.obj",
    "derived/retarget_*.npz", "derived/physics.json", "derived/sim/physics_traj.npz", "gt/gt.npz", "video.mp4",
]
OUTPUTS = ["viz/viz.json"]
CONFIG_SECTIONS = ["viz"]

log = logging.getLogger(__name__)


def run(ep: Episode, cfg) -> dict:
    vcfg = cfg.section("viz")
    figures = [str(f) for f in vcfg.get("figures", list(FIGURES))]
    written = make_figures(ep, ep.derived / "viz", figures, with_gt=bool(vcfg.get("ground_truth", True)))
    result = {"figures": {k: f"viz/{p.name}" for k, p in written.items()}, "rerun": None}
    mode = str(vcfg.get("rerun", "auto"))
    if mode not in ("auto", "on", "off"):
        raise PipelineError(f"viz.rerun must be auto, on or off, got {mode!r}")
    if mode != "off":
        if importlib.util.find_spec("rerun") is None:
            if mode == "on":
                raise MissingDependency(f"pip install {RERUN_PIN} (needed for step viz with viz.rerun = 'on')")
            log.info("rerun-sdk is not installed; skipping %s (pip install %s)", RRD_NAME, RERUN_PIN)
            result["rerun"] = f"skipped: rerun-sdk not installed ({RERUN_PIN})"
        else:
            from ..viz.rerun_export import export_rrd

            info = export_rrd(ep, ep.derived / RRD_NAME, vcfg)
            result["rerun"] = {"path": RRD_NAME, "video": info["video"]}
    ep.save_json("viz/viz.json", result, ep.step_meta(backend="matplotlib+rerun" if isinstance(result["rerun"], dict) else "matplotlib"))
    return {"figures": len(written), "rerun": "yes" if isinstance(result["rerun"], dict) else "no"}
