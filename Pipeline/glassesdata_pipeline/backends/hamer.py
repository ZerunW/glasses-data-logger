"""HaMeR backend of step ``hands2d`` (Linux + CUDA; written for that machine, never run on the Mac).

HaMeR (Pavlakos et al., CVPR 2024, https://github.com/geopavlakos/hamer) regresses MANO from a hand
crop. This backend returns, per frame and hand slot, the 21 joints in MediaPipe order (HaMeR's MANO
wrapper already outputs OpenPose hand order: ``mano_to_openpose`` plus the five fingertip
vertices, which is MediaPipe's order), their 2D projection, and the model's own camera-frame 3D
joints (extra array ``joints_cam_model[T,2,21,3]``). ``mp_world`` is filled with those joints
relative to their palm centre, so step ``hand3d`` works unchanged (``shape_frame = "auto"`` treats
them as camera-aligned).

Install (outside this repository)::

    git clone --recursive https://github.com/geopavlakos/hamer && cd hamer
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the driver
    pip install -e .[all]
    pip install -v -e third-party/ViTPose        # only for hamer_detector = "vitpose"
    bash fetch_demo_data.sh                      # checkpoints into ./_DATA
    # MANO: register at https://mano.is.tue.mpg.de, download, put MANO_RIGHT.pkl in _DATA/data/mano/

then ``hands2d.backend = "hamer"``, ``hands2d.repo_dir = "/path/to/hamer"``.

Hand boxes (``hands2d.hamer_detector``):

* ``"mediapipe"`` (default): boxes and handedness from MediaPipe's landmarks (works in egocentric
  views, where HaMeR's demo detector may not find a person at all);
* ``"vitpose"``: HaMeR's demo pipeline: Detectron2 ViTDet person boxes, ViTPose whole-body
  keypoints, hand boxes from the left/right hand keypoints with confidence > 0.5 (needs a visible
  body).

Model API used (``demo.py`` of the repository): ``hamer.models.load_hamer(DEFAULT_CHECKPOINT)`` ->
``(model, model_cfg)``; ``hamer.datasets.vitdet_dataset.ViTDetDataset(model_cfg, img_bgr, boxes,
right, rescale_factor)``; outputs ``pred_cam`` (weak-perspective ``[s, tx, ty]`` of the crop),
``pred_keypoints_3d`` (B, 21, 3); left hands are processed mirrored, so ``pred_cam[:, 1]`` and the
joints' x are negated back; ``hamer.utils.renderer.cam_crop_to_full(pred_cam, box_center, box_size,
img_size, focal_length)`` -> camera translation with the principal point at the image centre.
Here ``focal_length`` is the calibrated focal length (not the demo's 5000/256 x image size guess)
and the translation is corrected for the calibrated principal point.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import MissingDependency, PipelineError, require
from . import mediapipe_hands as mph

log = logging.getLogger(__name__)
PALM = [0, 5, 9, 13, 17]


@contextlib.contextmanager
def working_dir(path: Path) -> Iterator[None]:
    """The model loaders read relative paths (``./_DATA``, ``./mano_data``)."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def repo_dir(hcfg: Mapping[str, Any], marker: str, name: str) -> Path:
    repo = Path(str(hcfg.get("repo_dir", "") or ".")).expanduser().resolve()
    if not (repo / marker).exists():
        raise MissingDependency(f"{name} repository not found in {repo} (no {marker}); clone it and set hands2d.repo_dir "
                                f"(needed for hands2d backend {name.lower()})")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def pick_device(name: str) -> str:
    torch = require("torch", "torch", "the wilor / hamer backends")
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def regress(model: Any, model_cfg: Any, dataset_cls: Any, recursive_to: Callable, cam_crop_to_full: Callable,
            image_bgr: np.ndarray, boxes: np.ndarray, right: np.ndarray, camera: Any, device: str,
            rescale_factor: float = 2.0, batch_size: int = 16, dataset_kwargs: Mapping[str, Any] | None = None) -> list[dict]:
    """Run a HaMeR-style MANO regressor (HaMeR or WiLoR) on hand boxes of one BGR image.

    Returns one dict per box: ``side`` (0 left, 1 right), ``joints_cam`` (21, 3) metres in the
    OpenCV camera frame (calibrated focal length), ``uv`` (21, 2) the pinhole projection the model
    implies (principal point at the calibrated centre)."""
    torch = require("torch", "torch", "the wilor / hamer backends")
    if len(boxes) == 0:
        return []
    dataset = dataset_cls(model_cfg, image_bgr, np.asarray(boxes, dtype=np.float32), np.asarray(right, dtype=np.float32),
                          rescale_factor=rescale_factor, **dict(dataset_kwargs or {}))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    f_px = 0.5 * (camera.fx + camera.fy)
    H, W = image_bgr.shape[:2]
    out: list[dict] = []
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            pred = model(batch)
        multiplier = 2 * batch["right"] - 1
        pred_cam = pred["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                 batch["img_size"].float(), f_px).detach().cpu().numpy()
        joints = pred["pred_keypoints_3d"].detach().float().cpu().numpy().copy()
        sign = multiplier.detach().float().cpu().numpy().reshape(-1)
        joints[:, :, 0] *= sign[:, None]
        # cam_crop_to_full centres the projection at (W/2, H/2); shift to the calibrated (cx, cy)
        cam_t[:, 0] += (W / 2.0 - camera.cx) * cam_t[:, 2] / f_px
        cam_t[:, 1] += (H / 2.0 - camera.cy) * cam_t[:, 2] / f_px
        for b in range(len(joints)):
            J = joints[b] + cam_t[b][None, :]
            uv = np.stack([camera.fx * J[:, 0] / J[:, 2] + camera.cx, camera.fy * J[:, 1] / J[:, 2] + camera.cy], -1)
            out.append({"side": int(sign[b] > 0), "joints_cam": J, "uv": uv})
    return out


def track(ep: Any, cfg: Mapping[str, Any], detect: Callable[[np.ndarray, int], list[tuple[np.ndarray, int, float]]],
          regress_frame: Callable[[np.ndarray, np.ndarray, np.ndarray], list[dict]]) -> dict[str, np.ndarray]:
    """Whole video: ``detect(image, ms) -> [(box xyxy, is_right, score)]``, ``regress_frame(image,
    boxes, right) -> [dict]``; slots as in :func:`mediapipe_hands.assign_slots`."""
    hcfg = cfg.get("hands2d", {}) or {}
    timeline = ep.timeline()
    T = len(timeline["t"])
    ms = mph.timestamps_ms(timeline["t"])
    width = ep.camera().width
    keep_px = float(hcfg.get("keep_slot_frac", 0.08)) * width
    kp = np.full((T, 2, 21, 3), np.nan)
    score = np.full((T, 2), np.nan)
    mp_world = np.full((T, 2, 21, 3), np.nan)
    joints_model = np.full((T, 2, 21, 3), np.nan)
    source = np.zeros((T, 2), np.int8)
    last = np.full((2, 2), np.nan)
    seen = np.full(2, -10**9)
    for index, image in ep.frames():
        if index >= T:
            break
        last[index - seen > int(hcfg.get("continuity_frames", 12))] = np.nan
        dets = sorted(detect(image, int(ms[index])), key=lambda d: -d[2])[:2]
        if not dets:
            continue
        boxes = np.array([d[0] for d in dets], dtype=np.float32)
        right = np.array([d[1] for d in dets], dtype=np.float32)
        preds = regress_frame(image, boxes, right)
        found = []
        for (box, is_right, conf), p in zip(dets, preds):
            J = p["joints_cam"]
            found.append((mph.Detection(int(is_right), float(conf), p["uv"], np.full(21, float(conf)), J - J[PALM].mean(0)), J))
        by_id = {id(d): J for d, J in found}
        for slot, det, how in mph.assign_slots([d for d, _ in found], last, keep_px, float(hcfg.get("trust_label_score", 0.9))):
            kp[index, slot, :, :2] = det.uv
            kp[index, slot, :, 2] = det.kp_score
            score[index, slot] = det.score
            mp_world[index, slot] = det.world
            joints_model[index, slot] = by_id[id(det)]
            source[index, slot] = how
            last[slot] = det.wrist
            seen[slot] = index
    return {"kp": kp, "hand_score": score, "mp_world": mp_world, "joints_cam_model": joints_model, "slot_source": source}


def mediapipe_boxes(cfg: Mapping[str, Any], pad: float = 0.15) -> tuple[Callable, Callable[[], None]]:
    """A detector from MediaPipe's landmarks: ``(detect(image, ms), close())``."""
    hcfg = cfg.get("hands2d", {}) or {}
    model = mph.resolve_model(hcfg.get("model_path") or None, bool(hcfg.get("allow_download", True)))
    landmarker = mph.HandLandmarker(model, num_hands=2, mirrored_input=bool(hcfg.get("mirrored_input", False)))

    def detect(image: np.ndarray, ms: int) -> list[tuple[np.ndarray, int, float]]:
        out = []
        for d in landmarker.detect(image, ms):
            lo, hi = d.uv.min(0), d.uv.max(0)
            margin = pad * float(np.max(hi - lo))
            out.append((np.concatenate([lo - margin, hi + margin]), int(d.side == mph.RIGHT), d.score))
        return out

    return detect, landmarker.close


def vitpose_boxes(repo: Path, device: str, min_keypoint_conf: float = 0.5) -> Callable:
    """HaMeR demo.py's hand boxes: ViTDet people + ViTPose whole-body hand keypoints."""
    from detectron2.config import LazyConfig  # type: ignore[import-not-found]
    import hamer  # type: ignore[import-not-found]
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy  # type: ignore[import-not-found]
    from vitpose_model import ViTPoseModel  # type: ignore[import-not-found]  (repository root)

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    det_cfg = LazyConfig.load(str(cfg_path))
    det_cfg.train.init_checkpoint = (  # as in demo.py (URL not re-verified here)
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    with working_dir(repo):
        detector = DefaultPredictor_Lazy(det_cfg)
        pose = ViTPoseModel(device)

    def detect(image: np.ndarray, ms: int) -> list[tuple[np.ndarray, int, float]]:
        inst = detector(image)["instances"]
        keep = (inst.pred_classes == 0) & (inst.scores > 0.5)
        boxes = inst.pred_boxes.tensor[keep].cpu().numpy()
        scores = inst.scores[keep].cpu().numpy()
        if len(boxes) == 0:
            return []
        poses = pose.predict_pose(image[:, :, ::-1].copy(), [np.concatenate([boxes, scores[:, None]], axis=1)])
        out = []
        for person in poses:
            for keyp, is_right in ((person["keypoints"][-42:-21], 0), (person["keypoints"][-21:], 1)):
                ok = keyp[:, 2] > min_keypoint_conf
                if ok.sum() > 3:
                    box = np.array([keyp[ok, 0].min(), keyp[ok, 1].min(), keyp[ok, 0].max(), keyp[ok, 1].max()])
                    out.append((box, is_right, float(keyp[ok, 2].mean())))
        return out

    return detect


def hands2d(ep: Any, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    hcfg = cfg.get("hands2d", {}) or {}
    repo = repo_dir(hcfg, "hamer", "HaMeR")
    require("torch", "torch", "hands2d backend hamer")
    try:
        from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore[import-not-found]
        from hamer.models import DEFAULT_CHECKPOINT, load_hamer  # type: ignore[import-not-found]
        from hamer.utils import recursive_to  # type: ignore[import-not-found]
        from hamer.utils.renderer import cam_crop_to_full  # type: ignore[import-not-found]
    except ImportError as error:
        raise MissingDependency(f"HaMeR is not installed ({error}); see backends/hamer.py") from error
    device = pick_device(str(hcfg.get("device", "auto")))
    with working_dir(repo):
        model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    camera = ep.camera()
    which = str(hcfg.get("hamer_detector", "mediapipe"))
    close: Callable[[], None] = lambda: None  # noqa: E731
    if which == "mediapipe":
        detect, close = mediapipe_boxes(cfg)
    elif which == "vitpose":
        detect = vitpose_boxes(repo, device)
    else:
        raise PipelineError(f"hands2d.hamer_detector must be 'mediapipe' or 'vitpose', got {which!r}")

    def regress_frame(image: np.ndarray, boxes: np.ndarray, right: np.ndarray) -> list[dict]:
        return regress(model, model_cfg, ViTDetDataset, recursive_to, cam_crop_to_full, image, boxes, right, camera,
                       device, float(hcfg.get("rescale_factor", 2.0)), int(hcfg.get("batch_size", 16)))

    try:
        out = track(ep, cfg, detect, regress_frame)
    finally:
        close()
    out["model"] = "hamer:" + str(DEFAULT_CHECKPOINT)
    return out
