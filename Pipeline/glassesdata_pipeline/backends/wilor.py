"""WiLoR backend of step ``hands2d`` (Linux + CUDA; written for that machine, never run on the Mac).

WiLoR (Potamias et al., 2024, https://github.com/rolpotamias/WiLoR) = a YOLO hand detector (boxes and
handedness) + a HaMeR-style MANO regressor. Outputs as :mod:`.hamer` (same crop dataset, camera
model and left-hand mirroring): 21 joints in MediaPipe order (WiLoR's MANO wrapper uses the same
``mano_to_openpose`` remap + fingertip vertices = OpenPose hand order = MediaPipe order), their 2D
projection, ``joints_cam_model[T,2,21,3]`` (the model's camera-frame joints with the calibrated
focal length) and ``mp_world`` (those joints relative to the palm centre).

Install (outside this repository)::

    git clone --recursive https://github.com/rolpotamias/WiLoR && cd WiLoR
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the driver
    pip install -r requirements.txt
    wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
    wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
    # MANO: register at https://mano.is.tue.mpg.de, download, put MANO_RIGHT.pkl in ./mano_data/

then ``hands2d.backend = "wilor"``, ``hands2d.repo_dir = "/path/to/WiLoR"``. ``load_wilor`` and the
detector read ``./pretrained_models`` and ``./mano_data`` relative to the working directory, so
models are loaded with the repository as the working directory.

API used (``demo.py``): ``wilor.models.load_wilor(checkpoint_path, cfg_path)`` -> ``(model,
model_cfg)``; ``ultralytics.YOLO(detector.pt)(img_bgr, conf=..., verbose=False)[0]`` with
``det.boxes.data[:, :4]`` (xyxy) and ``det.boxes.cls`` (1 = right hand) and ``det.boxes.conf``;
``wilor.datasets.vitdet_dataset.ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor,
fp16)``; ``wilor.utils.renderer.cam_crop_to_full``; ``wilor.utils.recursive_to`` (this last import
path is taken from the HaMeR layout WiLoR copies; not re-verified).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..errors import MissingDependency, require
from . import hamer as common

log = logging.getLogger(__name__)


def hands2d(ep: Any, cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    hcfg = cfg.get("hands2d", {}) or {}
    repo = common.repo_dir(hcfg, "wilor", "WiLoR")
    torch = require("torch", "torch", "hands2d backend wilor")
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as error:
        raise MissingDependency("pip install ultralytics (needed for hands2d backend wilor)") from error
    try:
        from wilor.datasets.vitdet_dataset import ViTDetDataset  # type: ignore[import-not-found]
        from wilor.models import load_wilor  # type: ignore[import-not-found]
        from wilor.utils import recursive_to  # type: ignore[import-not-found]
        from wilor.utils.renderer import cam_crop_to_full  # type: ignore[import-not-found]
    except ImportError as error:
        raise MissingDependency(f"WiLoR is not installed ({error}); see backends/wilor.py") from error
    device = common.pick_device(str(hcfg.get("device", "auto")))
    fp16 = bool(hcfg.get("fp16", False)) and device == "cuda"
    with common.working_dir(repo):
        model, model_cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                      cfg_path="./pretrained_models/model_config.yaml")
        detector = YOLO("./pretrained_models/detector.pt")
    model = model.to(device).eval()
    detector = detector.to(device)
    camera = ep.camera()
    conf = float(hcfg.get("detector_confidence", 0.3))

    def detect(image: np.ndarray, ms: int) -> list[tuple[np.ndarray, int, float]]:
        result = detector(image, conf=conf, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        data = boxes.data.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().reshape(-1)
        scores = boxes.conf.detach().cpu().numpy().reshape(-1)
        return [(data[k, :4].astype(np.float64), int(round(cls[k])), float(scores[k])) for k in range(len(data))]

    def regress_frame(image: np.ndarray, boxes: np.ndarray, right: np.ndarray) -> list[dict]:
        return common.regress(model, model_cfg, ViTDetDataset, recursive_to, cam_crop_to_full, image, boxes, right,
                              camera, device, float(hcfg.get("rescale_factor", 2.0)), int(hcfg.get("batch_size", 16)),
                              dataset_kwargs={"fp16": fp16})

    with torch.inference_mode():
        out = common.track(ep, cfg, detect, regress_frame)
    out["model"] = "wilor:" + str(repo / "pretrained_models" / "wilor_final.ckpt")
    return out
