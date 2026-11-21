"""Image-to-3D shape for the ``objects`` step (``objects.mesh_backend = "image_to_3d"``).

A generative model completes the unseen side of the object from one masked frame; its output has
its own frame and scale, so the shape is aligned to the fused depth points with a similarity
transform (:func:`core.icp.align_similarity`: scale + rotations about the vertical + ICP) and the
convex hull of the aligned shape is returned (the step needs a closed mesh; a non-convex
generated mesh can be passed through unchanged later, once its watertightness can be tested on
the GPU machine).

Adapter for Meta's SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects; README and
``notebook/inference.py`` as of 2025-11): after the repository's ``doc/setup.md`` (conda env,
checkpoints in ``checkpoints/hf/``)::

    sys.path.append("<repo>/notebook")
    from inference import Inference
    inference = Inference("<repo>/checkpoints/hf/pipeline.yaml", compile=False)
    output = inference(image_rgb, mask, seed=42)      # dict
    output["gs"]                                       # Gaussian splat (save_ply)

The README documents only ``output["gs"]``; ``notebook/inference.py`` also reads ``"rotation"``,
``"translation"``, ``"scale"`` (object-to-camera) and ``"gaussian"``. This adapter takes, in
order, a mesh (``"glb"`` / ``"mesh"`` with ``vertices``) or the splat centres (``get_xyz`` /
``_xyz``) and does not rely on the predicted pose (the depth-based alignment replaces it).
Written, not run (needs CUDA and the model's environment).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np

from ..core.hull import hull_mesh
from ..core.icp import align_similarity
from ..errors import MissingDependency, PipelineError

log = logging.getLogger(__name__)


def _shape_points(output: Any) -> np.ndarray:
    """Vertices or splat centres from a SAM 3D Objects output dict."""
    for key in ("glb", "mesh"):
        mesh = output.get(key) if isinstance(output, dict) else None
        if mesh is not None and hasattr(mesh, "vertices"):
            return np.asarray(mesh.vertices, dtype=np.float64)
    for key in ("gs", "gaussian"):
        gs = output.get(key) if isinstance(output, dict) else None
        if gs is None:
            continue
        xyz = getattr(gs, "get_xyz", None)
        xyz = xyz() if callable(xyz) else (xyz if xyz is not None else getattr(gs, "_xyz", None))
        if xyz is not None:
            if hasattr(xyz, "detach"):
                xyz = xyz.detach().cpu().numpy()
            return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    raise PipelineError("image_to_3d: the model output has no mesh or Gaussian centres")


def run_model(image_bgr: np.ndarray, mask: np.ndarray, icfg: Any) -> np.ndarray:
    """Shape points in the model's own frame (SAM 3D Objects)."""
    model = str(icfg.get("model", "sam3d_objects"))
    if model != "sam3d_objects":
        raise PipelineError(f"unknown objects.image_to_3d.model {model!r}")
    repo, config = str(icfg.get("repo", "")), str(icfg.get("config", ""))
    if not repo or not config:
        raise MissingDependency("set objects.image_to_3d.repo and .config (clone facebookresearch/sam-3d-objects, "
                                "follow doc/setup.md; needed for objects.mesh_backend = image_to_3d)")
    sys.path.append(f"{repo}/notebook")
    try:
        from inference import Inference  # type: ignore[import-not-found]
    except ImportError as error:
        raise MissingDependency("SAM 3D Objects is not installed (see its doc/setup.md)") from error
    inference = Inference(config, compile=False)
    rgb = np.ascontiguousarray(image_bgr[..., ::-1])
    output = inference(rgb, np.asarray(mask, dtype=bool), seed=int(icfg.get("seed", 42)))
    return _shape_points(output)


def generate_aligned(ep: Any, cfg: Any, frame: int, mask_d: np.ndarray, inp: Any, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate the shape from video frame ``frame`` (mask at depth resolution, upsampled to the
    video), align it to ``target`` (world points of the fused, completed object) and return the
    convex hull of the aligned shape in world coordinates."""
    import cv2

    icfg = cfg.section("objects.image_to_3d")
    image = next(ep.frames([frame]))[1]
    mask = cv2.resize(mask_d.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    shape = run_model(image, mask, icfg)
    up = np.array([0.0, 0.0, 1.0])
    res = align_similarity(shape, target, up=up, n_yaw=int(icfg.get("n_yaw", 8)))
    aligned = shape @ res["T"][:3, :3].T + res["T"][:3, 3]
    log.info("image_to_3d: scale %.4f, rmse %.4f m over %d target points", res["s"], res["rmse"], len(target))
    return hull_mesh(aligned)
