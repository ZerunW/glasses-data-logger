"""Grasps from hand skeletons (numpy only).

:func:`detect_grasps` finds grasp episodes per hand in world-frame joints ``(T, 2, 21, 3)``: the
thumb-index aperture (joints 4 and 8) *closes* (drops by at least ``close_drop_m`` and to at most
``close_ratio`` of the open aperture seen in the previous ``open_window_s``) while the hand is
*slow* and *near a surface* (an optional callback, e.g. a depth test), and the hand then *moves*
(``move_m`` within ``move_window_s``) while the aperture stays closed. The episode ends when the
aperture re-opens past the middle between its closed and open values (``release_frac``), or when
the hand is lost for longer than ``max_gap_s``. Rejected candidates are returned with the reason.
With the frame timestamps ``t`` the speeds and the open / move / gap windows use the real times
(dropped frames, variable frame rate); without them ``t = frame / fps``.

:func:`contact_grasped` turns fingertip-to-surface distances into ``grasped[T, 2]``: two
fingertips within ``contact_m`` of the object during a detected episode of that hand, or outside
episodes when the object moves together with the hand (co-motion).

:func:`hand_rigid_motion` estimates the rigid motion of a hand that keeps its shape (a held grasp)
from all 21 joints (generalised Procrustes), which is far less noisy than a wrist frame from four
joints; :func:`smooth_poses` smooths a pose sequence.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

THUMB_TIP, INDEX_TIP = 4, 8
FINGERTIPS = (4, 8, 12, 16, 20)

DEFAULTS: dict[str, Any] = {
    "smooth_s": 0.06,
    "close_drop_m": 0.03,
    "close_ratio": 0.75,
    "open_window_s": 1.0,
    "settle_frac": 0.15,
    "release_frac": 0.5,
    "release_frames": 3,
    "slow_mps": 0.30,
    "speed_window_s": 0.12,
    "move_m": 0.05,
    "move_window_s": 2.0,
    "min_duration_s": 0.3,
    "max_gap_s": 0.5,
    "point": "thumb_fingers",
}


def gaussian_smooth(x: Any, sigma_frames: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along axis 0 (normalised by the weights of finite samples;
    samples that are NaN stay NaN)."""
    a = np.asarray(x, dtype=np.float64)
    if sigma_frames <= 0.3 or len(a) < 2:
        return a.copy()
    r = int(math.ceil(3 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    finite = np.isfinite(a)
    v = np.where(finite, a, 0.0)
    num = np.zeros_like(v)
    den = np.zeros_like(v)
    n = len(a)
    for i, w in enumerate(k):
        s = i - r
        lo, hi = max(0, -s), min(n, n - s)
        num[lo:hi] += w * v[lo + s:hi + s]
        den[lo:hi] += w * finite[lo + s:hi + s]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~finite] = np.nan
    return out


def aperture(joints: Any) -> np.ndarray:
    """Thumb-tip to index-tip distance ``(T, 2)``."""
    J = np.asarray(joints, dtype=np.float64)
    return np.linalg.norm(J[..., THUMB_TIP, :] - J[..., INDEX_TIP, :], axis=-1)


def grasp_point(joints: Any, mode: str = "thumb_fingers") -> np.ndarray:
    """The point held between the fingers ``(..., 3)``.

    ``"thumb_fingers"``: midpoint of the thumb tip and the mean of the other four fingertips (the
    opposition centre; robust to power grasps where the index tip is not opposite the thumb).
    ``"thumb_index"``: midpoint of the thumb and index tips (precision grasps)."""
    J = np.asarray(joints, dtype=np.float64)
    if mode == "thumb_index":
        return 0.5 * (J[..., THUMB_TIP, :] + J[..., INDEX_TIP, :])
    if mode == "thumb_fingers":
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all four fingertips missing -> NaN
            return 0.5 * (J[..., THUMB_TIP, :] + np.nanmean(J[..., [8, 12, 16, 20], :], axis=-2))
    raise ValueError(f"unknown grasp point mode {mode!r}")


def speed(points: Any, fps: float, window_frames: int, t: Any = None) -> np.ndarray:
    """Central-difference speed ``|p(i+w) - p(i-w)| / dt`` along axis 0 (one-sided at the ends),
    ``dt`` from the timestamps ``t`` (seconds) when given, else ``2w / fps``. With ``t`` the
    window is ``window_frames / fps`` seconds on each side, so dropped frames do not shorten it."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    i = np.arange(n)
    if t is None:
        w = max(1, int(window_frames))
        a = np.clip(i - w, 0, n - 1)
        b = np.clip(i + w, 0, n - 1)
        dt = np.maximum(b - a, 1) / fps
    else:
        t = np.asarray(t, dtype=np.float64)
        half = max(1, int(window_frames)) / fps
        a = np.minimum(np.searchsorted(t, t - half - 1e-9, side="left"), np.maximum(i - 1, 0))
        b = np.maximum(np.searchsorted(t, t + half + 1e-9, side="right") - 1, np.minimum(i + 1, n - 1))
        with np.errstate(invalid="ignore"):
            dt = np.where(t[b] - t[a] > 0, t[b] - t[a], np.nan)
    return np.linalg.norm(p[b] - p[a], axis=-1) / (dt if p.ndim == 2 else dt[:, None])


def detect_grasps(
    joints: Any,
    fps: float,
    params: Mapping[str, Any] | None = None,
    near_surface: Callable[[int, int, np.ndarray], bool | None] | None = None,
    t: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grasp episodes ``[{"hand", "onset", "release", "point" (3,), "aperture_open",
    "aperture_closed", "speed_onset", "moved_m"}]`` (frames inclusive) and rejected candidates
    (same keys plus ``"reason"``). ``near_surface(hand, frame, point)`` returns True/False, or None
    when it cannot tell (accepted). ``t``: frame timestamps (seconds; default ``frame / fps``)
    for the speeds and the open / move / gap windows; the smoothing and ``min_duration_s`` use
    ``fps`` (the median rate)."""
    p = {**DEFAULTS, **dict(params or {})}
    J = gaussian_smooth(np.asarray(joints, dtype=np.float64), float(p["smooth_s"]) * fps)
    T = J.shape[0]
    times = np.arange(T) / fps if t is None else np.asarray(t, dtype=np.float64)
    eps = 1e-6
    ap = aperture(J)
    gp = grasp_point(J, str(p["point"]))
    wrist = J[:, :, 0, :]
    open_s, move_s, gap_s = float(p["open_window_s"]), float(p["move_window_s"]), float(p["max_gap_s"])
    min_len = max(1, int(round(float(p["min_duration_s"]) * fps)))

    def first_at(time: float) -> int:  # first frame at or after ``time``
        return int(np.searchsorted(times, time - eps, side="left"))

    def last_at(time: float) -> int:  # last frame at or before ``time``
        return int(np.searchsorted(times, time + eps, side="right")) - 1
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for hand in range(J.shape[1]):
        a = ap[:, hand]
        spd = speed(np.where(np.isfinite(wrist[:, hand]), wrist[:, hand], np.nan), fps,
                    int(round(float(p["speed_window_s"]) * fps)), times)
        t = 0
        while t < T:
            if not np.isfinite(a[t]):
                t += 1
                continue
            past = a[first_at(times[t] - open_s):t + 1]
            a_open = float(np.nanmax(past))
            drop = a_open - a[t]
            if not (drop >= float(p["close_drop_m"]) and a[t] <= float(p["close_ratio"]) * a_open):
                t += 1
                continue
            # closing: onset where the aperture settles near its minimum in the next window
            fut = a[t:max(t + 1, min(T, first_at(times[t] + open_s)))]
            a_min = float(np.nanmin(fut))
            settle = a_min + float(p["settle_frac"]) * (a_open - a_min)
            onset = t + int(np.argmax(fut <= settle))
            closed = float(np.nanmedian(a[onset:min(T, onset + max(3, min_len))]))
            open_thr = closed + float(p["release_frac"]) * (a_open - closed)
            # release: aperture above open_thr for release_frames frames, or the hand lost
            release, run, last_ok = T - 1, 0, onset
            for u in range(onset + 1, T):
                if not np.isfinite(a[u]):
                    if times[u] - times[last_ok] > gap_s + eps:
                        release = last_ok
                        break
                    continue
                last_ok = u
                if a[u] >= open_thr:
                    run += 1
                    if run >= int(p["release_frames"]):
                        release = u - run
                        break
                else:
                    run = 0
            else:
                release = last_ok
            cand: dict[str, Any] = {
                "hand": hand, "onset": int(onset), "release": int(max(release, onset)),
                "point": gp[onset, hand].copy(), "aperture_open": a_open, "aperture_closed": closed,
                "speed_onset": float(np.nanmedian(spd[max(0, onset - 2):onset + 3])),
            }
            end = min(cand["release"], max(onset, last_at(times[onset] + move_s)))
            moved = np.linalg.norm(wrist[onset:end + 1, hand] - wrist[onset, hand], axis=-1)
            cand["moved_m"] = float(np.nanmax(moved)) if np.any(np.isfinite(moved)) else 0.0
            reason = None
            if cand["release"] - onset + 1 < min_len:
                reason = "too short"
            elif not cand["speed_onset"] <= float(p["slow_mps"]):
                reason = "hand not slow at onset"
            elif cand["moved_m"] < float(p["move_m"]):
                reason = "hand did not move while closed"
            elif near_surface is not None and near_surface(hand, onset, cand["point"]) is False:
                reason = "not near a surface"
            if reason is None:
                episodes.append(cand)
            else:
                rejected.append({**cand, "reason": reason})
            t = cand["release"] + 1
    episodes.sort(key=lambda e: (e["onset"], e["hand"]))
    return episodes, rejected


def fingertip_distances(joints: Any, T_world_obj: Any, surface_index: Any) -> np.ndarray:
    """Distances ``(T, 2, 5)`` of the fingertips to the object's surface points (a
    :class:`~glassesdata_pipeline.core.knn.KNN` over points in the object frame), NaN where the
    joint or the pose is missing."""
    from .. import geometry as geo

    J = np.asarray(joints, dtype=np.float64)[:, :, FINGERTIPS, :]  # (T,2,5,3)
    Tw = np.asarray(T_world_obj, dtype=np.float64)
    T_ow = geo.inv_T(Tw)
    local = geo.transform(T_ow[:, None, None], J)
    flat = local.reshape(-1, 3)
    out = np.full(len(flat), np.nan)
    ok = np.all(np.isfinite(flat), axis=1)
    if np.any(ok):
        out[ok] = surface_index.nearest(flat[ok])[0]
    return out.reshape(J.shape[:-1])


def _fill_and_prune(x: np.ndarray, gap: int, min_run: int) -> np.ndarray:
    y = x.copy()
    n = len(y)
    idx = np.flatnonzero(y)
    if len(idx) > 1:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap + 1:
                y[a:b] = True
    # remove short runs
    i = 0
    while i < n:
        if y[i]:
            j = i
            while j < n and y[j]:
                j += 1
            if j - i < min_run:
                y[i:j] = False
            i = j
        else:
            i += 1
    return y


def contact_grasped(
    tip_dist: Any,
    episodes: list[dict[str, Any]],
    object_speed: Any,
    wrist_speed: Any,
    contact_m: float = 0.035,
    min_contact_frac: float = 0.3,
    co_motion_mps: float = 0.05,
    co_motion_tol: float = 0.5,
    gap_frames: int = 3,
    min_frames: int = 3,
    min_tips: int = 2,
) -> np.ndarray:
    """``grasped (T, 2)`` from fingertip distances ``(T, 2, 5)`` (thumb first), the episodes that
    belong to this object, and speeds ``(T,)`` of the object / ``(T, 2)`` of the wrists.

    Contact in a frame: at least ``min_tips`` fingertips within ``contact_m`` of the object surface
    (not necessarily the thumb: in power grasps it often rests beyond an edge). A detected episode of a hand marks all its frames when contact holds in at
    least ``min_contact_frac`` of them (the grasp is on this object; single frames lose contact
    through tracking noise). Outside episodes, contact frames where the object moves faster than
    ``co_motion_mps`` together with the wrist (speeds within ``co_motion_tol``) count too."""
    d = np.asarray(tip_dist, dtype=np.float64)
    T = d.shape[0]
    with np.errstate(invalid="ignore"):
        contact = np.sum(np.where(np.isfinite(d), d, np.inf) <= contact_m, axis=-1) >= min_tips
    out = np.zeros((T, 2), dtype=bool)
    in_ep = np.zeros((T, 2), dtype=bool)
    for e in episodes:
        a, b, h = max(0, int(e["onset"])), min(T - 1, int(e["release"])), int(e["hand"])
        in_ep[a:b + 1, h] = True
        if b >= a and contact[a:b + 1, h].mean() >= min_contact_frac:
            out[a:b + 1, h] = True
    vo = np.asarray(object_speed, dtype=np.float64)
    vw = np.asarray(wrist_speed, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        co = (vo[:, None] >= co_motion_mps) & (np.abs(vw - vo[:, None]) <= co_motion_tol * np.maximum(vo[:, None], 1e-9))
    extra = contact & co & ~in_ep
    extra = np.stack([_fill_and_prune(extra[:, h], gap_frames, min_frames) for h in range(extra.shape[1])], axis=1)
    return out | extra


def hand_rigid_motion(joints: Any, frames: Any, iterations: int = 3, outlier: float = 2.5) -> np.ndarray:
    """Rigid motion of a hand that holds its shape (a closed grasp): ``(len(frames), 4, 4)``
    transforms ``H_f`` with ``joints[f] ~ H_f @ S`` for a mean shape ``S`` (generalised Procrustes
    over the given frames; all 21 joints, so the noise of single joints averages out - much better
    than the wrist frame from four joints). Joints with residuals above ``outlier`` x the median
    are down-weighted to zero (fingers that move). Rows are NaN where fewer than 4 joints exist.
    ``S`` is expressed in the frame of the first frame with a valid hand."""
    from .align import umeyama
    from .. import geometry as geo

    J = np.asarray(joints, dtype=np.float64)[np.asarray(frames, dtype=np.int64)]  # (F,21,3)
    F = len(J)
    out = np.full((F, 4, 4), np.nan)
    ok = np.all(np.isfinite(J), axis=-1).sum(axis=1) >= 4
    if not np.any(ok):
        return out
    ref = int(np.flatnonzero(ok)[0])
    S = J[ref].copy()
    w = np.ones(J.shape[:2])
    H = np.full((F, 4, 4), np.nan)
    for _ in range(max(1, iterations)):
        aligned = np.full_like(J, np.nan)
        for i in np.flatnonzero(ok):
            try:
                _, R, t = umeyama(S, J[i], with_scale=False, weights=np.maximum(w[i], 1e-9))
            except ValueError:
                continue
            H[i] = geo.make_T(R, t)
            aligned[i] = (J[i] - t) @ R  # back into the shape frame
        res = np.linalg.norm(aligned - S, axis=-1)
        med = np.nanmedian(res)
        w = np.where(np.isfinite(res) & (res <= outlier * max(med, 1e-6)), 1.0, 0.0)
        with np.errstate(invalid="ignore"):
            S_new = np.nansum(np.where(w[..., None] > 0, aligned, 0.0), axis=0) / np.maximum(w.sum(axis=0), 1)[:, None]
        S = np.where((w.sum(axis=0) > 0)[:, None], S_new, S)
    out[:] = H
    return out


def smooth_poses(T: Any, valid: Any, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a pose sequence ``(N, 4, 4)`` over the valid rows (chordal mean of
    rotations, mean of translations); invalid rows get the smoothed value of their neighbours."""
    from .. import geometry as geo

    P = np.asarray(T, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.all(np.isfinite(P), axis=(-2, -1))
    out = P.copy()
    if sigma <= 0 or not np.any(v):
        return out
    r = int(math.ceil(3 * sigma))
    idx = np.flatnonzero(v)
    for i in range(len(P)):
        js = idx[(idx >= i - r) & (idx <= i + r)]
        if len(js) == 0:
            continue
        wt = np.exp(-0.5 * ((js - i) / sigma) ** 2)
        R = geo.project_to_so3(np.einsum("n,nij->ij", wt, P[js, :3, :3]))
        t = (wt[:, None] * P[js, :3, 3]).sum(axis=0) / wt.sum()
        out[i] = geo.make_T(R, t)
    return out
