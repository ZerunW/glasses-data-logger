"""The synthetic choreography as smooth functions of time (room frame, floor z = 0, z up).

Every quantity is a C2-continuous function of time (quintic smoothstep keyframes and sinusoids),
so the IMU can be synthesised by finite differences at any rate. Phases are fractions of the
clip length, so short test clips run the same script faster.

Right hand: rest -> reach (fingers open) -> approach and grasp box_a -> lift ~0.15 m -> move ~0.3 m
(with a yaw turn) -> place -> release -> back off -> retract. box_a follows the wrist rigidly while
grasped. Left hand: resting near the table edge (small drift, finger taps).
Head: leaning over the table (eye ~1.40-1.48 m), looking down (pitch about -41..-55 deg), following
the right hand (low-passed), look-around of +-25 deg yaw at the start and end, nod/sway/roll.

The choreography is scaled to the clip length. At the default 8 s the head moves like a person
(peak acceleration ~1.5 m/s^2, angular rate < 100 deg/s); clips shorter than ~6 s are fast (a 2 s
test clip reaches ~1.5 g of head acceleration) - fine for consistency tests, not for realism.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import geometry as geo
from .. import skeleton as sk


def smoothstep(x: np.ndarray) -> np.ndarray:
    """Quintic smoothstep (C2): 0 below 0, 1 above 1."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def bump(x: np.ndarray) -> np.ndarray:
    """C2 bump on [0, 1], peak 1 at 0.5, zero (with zero slope and curvature) at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return 64.0 * x**3 * (1.0 - x) ** 3


class Track:
    """Keyframed vectors with smoothstep blending (holds the first/last value outside)."""

    def __init__(self, times: list[float], values: list[np.ndarray]) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64).reshape(len(times), -1)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("keyframe times must increase")

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        k = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        s = smoothstep((t - self.times[k]) / (self.times[k + 1] - self.times[k]))
        return self.values[k] + (self.values[k + 1] - self.values[k]) * s[:, None]


class RotTrack:
    """Keyframed rotations, geodesic interpolation with smoothstep timing."""

    def __init__(self, times: list[float], rotations: list[np.ndarray]) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        self.R = np.asarray(rotations, dtype=np.float64)
        self.delta = geo.so3_log(np.swapaxes(self.R[:-1], -1, -2) @ self.R[1:])

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        k = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2)
        s = smoothstep((t - self.times[k]) / (self.times[k + 1] - self.times[k]))
        return self.R[k] @ geo.so3_exp(self.delta[k] * s[:, None])


def hand_rotation(distal: np.ndarray, palmar: np.ndarray) -> np.ndarray:
    """Rotation whose columns are the canonical hand axes x = y cross z, y = distal, z = palmar."""
    y = np.asarray(distal, dtype=np.float64)
    y = y / np.linalg.norm(y)
    z = np.asarray(palmar, dtype=np.float64)
    z = z - y * (z @ y)
    z = z / np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=-1)


# Finger poses: flexion (5, 3) then abduction (5,), flattened to 20 values.
def _fingers(flex_fingers, flex_thumb, abd_fingers, abd_thumb) -> np.ndarray:
    flex = np.vstack([flex_thumb, np.tile(flex_fingers, (4, 1))])
    abd = np.array([abd_thumb] + [abd_fingers] * 4)
    return np.concatenate([flex.reshape(-1), abd])


RELAXED = _fingers([0.25, 0.35, 0.20], [0.10, 0.15, 0.10], 0.04, 0.25)
OPEN = _fingers([0.08, 0.10, 0.05], [0.00, 0.00, 0.00], 0.12, 0.55)
CLOSED = _fingers([0.60, 1.10, 0.60], [0.35, 0.40, 0.35], 0.00, 0.20)
FLAT_REST = _fingers([0.12, 0.18, 0.10], [0.10, 0.10, 0.05], 0.06, 0.30)  # resting palm-down on a table

HEAD_FOLLOW_S = 0.35  # Gaussian time constant of the head following the right hand (seconds)
REACH_SMOOTH_S = 0.12  # Gaussian smoothing of the reach correction (seconds; less for clips < 3 s)


def _gauss_taps(sigma: float, n: int = 6) -> list[tuple[float, float]]:
    """Offsets and weights of a normalized Gaussian sampled at +-3 sigma (2n+1 taps)."""
    offsets = np.linspace(-3.0 * sigma, 3.0 * sigma, 2 * n + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return list(zip(offsets.tolist(), weights.tolist()))


def box_signed_distance(points: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Signed distance (negative inside) of points given in a box's frame (centre origin)."""
    q = np.abs(points) - np.asarray(size) / 2
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


# The grasp: box centre in the wrist frame (x radial, y distal, z palmar).
GRASP_OFFSET = np.array([-0.010, 0.072, 0.046])


@dataclass
class Phases:
    """Phase boundaries as fractions of the clip."""

    look_start: tuple[float, float] = (0.0, 0.13)
    reach: tuple[float, float] = (0.13, 0.28)
    approach: tuple[float, float] = (0.28, 0.36)
    lift: tuple[float, float] = (0.36, 0.46)
    carry: tuple[float, float] = (0.46, 0.63)
    place: tuple[float, float] = (0.63, 0.71)
    release: tuple[float, float] = (0.71, 0.76)
    back_off: tuple[float, float] = (0.76, 0.80)
    retract: tuple[float, float] = (0.80, 0.90)
    look_end: tuple[float, float] = (0.86, 1.0)


class Script:
    """All ground-truth motion as functions of time ``t`` (seconds, 1-D arrays).

    Args:
        duration: clip length in seconds.
        layout: from :func:`scene.build_scene` (initial object poses, table).
        profile: body profile (hand size, arm lengths, shoulder offsets).
        seed: varies sinusoid phases and amplitudes slightly.
    """

    def __init__(self, duration: float, layout: dict, profile: dict, seed: int = 0) -> None:
        self.duration = float(duration)
        self.layout = layout
        self.profile = sk.complete_profile(profile)
        rng = np.random.default_rng(seed + 1000)
        self.phase = rng.uniform(0, 2 * math.pi, size=16)
        self.amp = rng.uniform(0.8, 1.2, size=16)
        P = Phases()
        D = self.duration
        self.P = P
        top = layout["table"]["top_z"]
        T_obj0 = layout["object_poses"][0]
        self.T_obj0 = T_obj0

        # ---------------- right hand keyframes
        rest_R = hand_rotation([0.95, 0.28, -0.05], [0.0, 0.0, -1.0])  # palm down, fingers forward-inward
        rest_p = self._resting(rest_R, [0.28, -0.25], top, "right")
        grasp_yaw = 0.12
        grasp_R = geo.rot_z(grasp_yaw) @ hand_rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # thumb up, palm left
        grasp_p = T_obj0[:3, 3] - grasp_R @ GRASP_OFFSET
        T_grasp = geo.make_T(grasp_R, grasp_p)
        pre_p = grasp_p - 0.06 * grasp_R[:, 2] - 0.03 * grasp_R[:, 1] + np.array([0, 0, 0.02])
        lift_T = T_grasp.copy()
        lift_T[2, 3] += 0.15
        # Placement: box moved ~0.3 m to the left, turned by +25 deg about the vertical.
        turn = math.radians(25.0)
        place_center = T_obj0[:3, 3] + np.array([0.02, 0.27, 0.0])
        T_obj_place = geo.make_T(geo.rot_z(turn) @ T_obj0[:3, :3], place_center)
        T_place = T_obj_place @ geo.inv_T(T_obj0) @ T_grasp
        carry_T = T_place.copy()
        carry_T[2, 3] += 0.15
        back_p = T_place[:3, 3] - 0.07 * T_place[:3, 2] - 0.02 * T_place[:3, 1] + np.array([0, 0, 0.03])
        rest2_R = hand_rotation([0.95, 0.25, -0.05], [0.0, 0.0, -1.0])
        rest2_p = self._resting(rest2_R, [0.28, -0.23], top, "right")
        f = lambda x: x * D  # noqa: E731
        times = [0.0, f(P.reach[0]), f(P.reach[1]), f(P.approach[1]), f(P.lift[1]), f(P.carry[1]),
                 f(P.place[1]), f(P.release[1]), f(P.back_off[1]), f(P.retract[1]), D]
        positions = [rest_p, rest_p, pre_p, grasp_p, lift_T[:3, 3], carry_T[:3, 3], T_place[:3, 3],
                     T_place[:3, 3], back_p, rest2_p, rest2_p]
        rotations = [rest_R, rest_R, grasp_R, grasp_R, lift_T[:3, :3], carry_T[:3, :3], T_place[:3, :3],
                     T_place[:3, :3], T_place[:3, :3], rest2_R, rest2_R]
        self._right_track = Track(times, positions)
        # C2 arcs lift the hand over the table clutter while reaching and retracting.
        self._arcs = [(f(P.reach[0]), f(P.reach[1]), 0.08), (f(P.back_off[1]), f(P.retract[1]), 0.05)]
        self._right_R = RotTrack(times, rotations)
        self.t_grasp = f(P.approach[1])
        self.t_release = f(P.place[1])
        self.T_grasp = T_grasp
        self.T_obj_place = T_obj_place
        ftimes = [0.0, f(P.reach[0]), f(P.reach[0]) + 0.7 * (f(P.reach[1]) - f(P.reach[0])),
                  f(P.approach[0]) + 0.35 * (f(P.approach[1]) - f(P.approach[0])), f(P.approach[1]),
                  f(P.place[1]), f(P.release[1]), f(P.retract[0]) + 0.5 * (f(P.retract[1]) - f(P.retract[0])), D]
        self.closed = self._fit_grasp(T_grasp, T_obj0, np.asarray(layout["object_sizes"][0]))
        fvalues = [FLAT_REST, FLAT_REST, OPEN, OPEN, self.closed, self.closed, OPEN, FLAT_REST, FLAT_REST]
        self._right_fingers = Track(ftimes, fvalues)

        # ---------------- left hand: resting near the front-left table edge
        self._left_R0 = hand_rotation([0.95, -0.22, -0.05], [0.0, 0.0, -1.0])
        self._left_p0 = self._resting(self._left_R0, [0.29, 0.33], top, "left", extra_flex=0.06)

        # ---------------- head
        self._eye0 = np.array([0.10, -0.02, 1.44])
        self._wrist0 = rest_p
        self.max_reach = 0.88  # the eye (and shoulders) shift towards a wrist that would be further

    def _fit_grasp(self, T_wrist: np.ndarray, T_obj: np.ndarray, size: np.ndarray, clearance: float = 0.008) -> np.ndarray:
        """Closed-hand finger values: per finger, the largest fraction (<= 1.6) of the CLOSED
        flexion that keeps every joint of that finger ``clearance`` outside the box (joints are
        bone centre lines, so the clearance stands in for the finger's thickness)."""
        T_obj_wrist = geo.inv_T(T_obj) @ T_wrist
        result = CLOSED.copy()
        for f, name in enumerate(sk.FINGER_NAMES):
            base = CLOSED[3 * f:3 * f + 3].copy()
            lo, hi = 0.0, 1.6
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                trial = result.copy()
                trial[3 * f:3 * f + 3] = base * mid
                J = sk.pose_hand(self.profile, "right", T_obj_wrist, trial[:15].reshape(5, 3), trial[15:])
                if box_signed_distance(J[sk.FINGERS[name][1:]], size).min() >= clearance:
                    lo = mid
                else:
                    hi = mid
            result[3 * f:3 * f + 3] = base * lo
        return result

    def _resting(self, R: np.ndarray, xy, top: float, side: str, extra_flex: float = 0.0,
                 clearance: float = 0.008) -> np.ndarray:
        """Wrist position of a hand resting palm-down at ``xy`` whose lowest joint is
        ``clearance`` above the table (joints are bone centre lines)."""
        flex = FLAT_REST[:15].reshape(5, 3) + extra_flex
        J = sk.pose_hand(self.profile, side, geo.make_T(R, [xy[0], xy[1], 0.0]), flex, FLAT_REST[15:])
        return np.array([xy[0], xy[1], top + clearance - J[:, 2].min()])

    # ------------------------------------------------------------------ hands

    def wrist_pose(self, t: np.ndarray, side: str) -> np.ndarray:
        """``T_room_wrist`` ``(N, 4, 4)`` of the canonical hand frame."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if side == "right":
            return geo.make_T(self._right_R(t), self._right_p(t))
        ph, a = self.phase, self.amp
        drift = np.stack([
            0.004 * a[0] * np.sin(2 * np.pi * t / 3.1 + ph[0]),
            0.004 * a[1] * np.sin(2 * np.pi * t / 2.3 + ph[1]),
            0.0015 * a[2] * np.sin(2 * np.pi * t / 1.7 + ph[2]),
        ], axis=-1)
        wobble = geo.so3_exp(np.stack([
            0.02 * np.sin(2 * np.pi * t / 2.9 + ph[3]),
            0.015 * np.sin(2 * np.pi * t / 3.7 + ph[4]),
            0.03 * np.sin(2 * np.pi * t / 4.3 + ph[5]),
        ], axis=-1))
        return geo.make_T(self._left_R0 @ wobble, self._left_p0 + drift)

    def _right_p(self, t: np.ndarray) -> np.ndarray:
        """Right wrist position: keyframes plus vertical arcs."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        p = self._right_track(t)
        for t0, t1, height in self._arcs:
            p[:, 2] += height * bump((t - t0) / (t1 - t0))
        return p

    def fingers(self, t: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Flexion ``(N, 5, 3)`` and abduction ``(N, 5)``."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if side == "right":
            v = self._right_fingers(t)
        else:
            v = np.tile(FLAT_REST, (len(t), 1))
            tap = 0.35 * np.maximum(0.0, np.sin(2 * np.pi * t / 1.9 + self.phase[6])) ** 4
            v[:, 3] -= tap  # index MCP extension: the finger lifts off the table
            v[:, 4] -= 0.3 * tap
            v[:, 6:15] += 0.05 * np.sin(2 * np.pi * t[:, None] / 3.3 + self.phase[7])
        return v[:, :15].reshape(-1, 5, 3), v[:, 15:]

    def joints(self, t: np.ndarray, side: str) -> np.ndarray:
        """``(N, 21, 3)`` hand joints in the room."""
        T = self.wrist_pose(t, side)
        flex, abd = self.fingers(t, side)
        return np.stack([sk.pose_hand(self.profile, side, T[i], flex[i], abd[i]) for i in range(len(T))])

    # ------------------------------------------------------------------ objects

    def object_pose(self, t: np.ndarray, k: int) -> np.ndarray:
        """``T_room_obj`` ``(N, 4, 4)`` of object k (0 is grasped, 1 static)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        T0 = self.layout["object_poses"][k]
        out = np.repeat(T0[None], len(t), axis=0)
        if k != 0:
            return out
        attached = (t >= self.t_grasp) & (t <= self.t_release)
        if np.any(attached):
            Tw = self.wrist_pose(t[attached], "right")
            out[attached] = Tw @ (geo.inv_T(self.T_grasp) @ T0)
        out[t > self.t_release] = self.T_obj_place
        return out

    def grasped(self, t: np.ndarray) -> np.ndarray:
        """``(N, K, 2)`` bool: object k held by hand slot (0 left, 1 right)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        K = len(self.layout["object_ids"])
        out = np.zeros((len(t), K, 2), dtype=bool)
        out[:, 0, 1] = (t >= self.t_grasp) & (t <= self.t_release)
        return out

    # ------------------------------------------------------------------ arms

    def swivel(self, t: np.ndarray, side: str) -> np.ndarray:
        """Elbow swivel (radians; 0 = elbow lowest, positive = outward), see skeleton.elbow_from_swivel."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ph = self.phase
        if side == "right":
            carry = bump((t / self.duration - self.P.lift[0]) / (self.P.place[1] - self.P.lift[0]))
            return 0.45 + 0.10 * np.sin(2 * np.pi * t / 4.1 + ph[8]) + 0.20 * carry
        return 0.30 + 0.05 * np.sin(2 * np.pi * t / 5.3 + ph[9])

    # ------------------------------------------------------------------ head / camera

    def eye(self, t: np.ndarray) -> np.ndarray:
        """Camera centre ``(N, 3)``: the scripted path plus a smooth shift towards any wrist that
        would otherwise be further than ``max_reach`` x (upper arm + forearm) from its shoulder.
        The shift is Gaussian-smoothed (``REACH_SMOOTH_S``) so the head never jerks."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        shift = np.zeros((len(t), 3))
        for tau, w in _gauss_taps(min(REACH_SMOOTH_S, 0.04 * self.duration)):
            tt = t + tau
            shift += w * self._reach_shift(tt, self._eye_base(tt), self.head_angles(tt)[0])
        return self._eye_base(t) + shift

    def _reach_shift(self, t: np.ndarray, eye: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        p = self.profile
        limit = self.max_reach * (p["upper_arm_length_m"] + p["forearm_length_m"])
        T_cam = np.zeros((len(t), 4, 4))  # only the heading matters for the shoulders
        T_cam[:, :3, :3] = geo.rot_z(yaw) @ np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        T_cam[:, :3, 3] = eye
        T_cam[:, 3, 3] = 1.0
        S = sk.shoulders_from_head(T_cam, p)
        shift = np.zeros_like(eye)
        eps = 0.01
        for slot, side in enumerate(("left", "right")):
            v = self.wrist_pose(t, side)[:, :3, 3] - S[:, slot]
            vh = v.copy()
            vh[:, 2] = 0.0
            u = vh / np.maximum(np.linalg.norm(vh, axis=1, keepdims=True), 1e-9)
            b = np.sum(v * u, axis=1)
            disc = b * b - np.sum(v * v, axis=1) + limit * limit
            if np.any(disc < 0):
                raise ValueError("a wrist is out of reach even with the shoulder right above it; change the script")
            delta = b - np.sqrt(disc)  # horizontal shift that brings the wrist to the limit
            delta = 0.5 * (delta + np.sqrt(delta * delta + eps * eps))  # smooth max(delta, 0)
            shift += delta[:, None] * u
        return shift

    def _wrist_lp(self, t: np.ndarray) -> np.ndarray:
        """Right wrist position low-passed with a Gaussian of ``HEAD_FOLLOW_S`` seconds: the head
        follows the hand slowly whatever the clip length."""
        out = np.zeros((len(t), 3))
        for tau, w in _gauss_taps(HEAD_FOLLOW_S):
            out += w * self._right_p(t + tau)
        return out

    def _eye_base(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ph, a = self.phase, self.amp
        wrist = self._wrist_lp(t)
        lean = 0.30 * (wrist - self._wrist0) * np.array([1.0, 1.0, 0.25])
        sway = np.stack([
            0.012 * a[10] * np.sin(2 * np.pi * t / 2.7 + ph[10]),
            0.015 * a[11] * np.sin(2 * np.pi * t / 3.4 + ph[11]),
            0.006 * a[12] * np.sin(2 * np.pi * t / 1.9 + ph[12]),
        ], axis=-1)
        slow = np.sin(np.pi * t / max(self.duration, 4.0))
        drift = np.stack([0.03 * slow, 0.05 * slow**2, -0.02 * slow], axis=-1)
        return self._eye0 + lean + sway + drift

    def head_angles(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Yaw, pitch, roll (radians) of the camera; pitch negative = looking down."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ph, a = self.phase, self.amp
        D = self.duration
        eye = self._eye_base(t)
        wrist = self._wrist_lp(t)
        to_wrist = np.arctan2(wrist[:, 1] - eye[:, 1], wrist[:, 0] - eye[:, 0])
        to_table = np.arctan2(0.05 - eye[:, 1], 0.62 - eye[:, 0])
        look_amp = math.radians(25.0) * min(1.0, D / 6.0)  # short clips: smaller, still smooth look-arounds
        look = (look_amp * bump((t / D - self.P.look_start[0]) / (self.P.look_start[1] - self.P.look_start[0]))
                - look_amp * bump((t / D - self.P.look_end[0]) / (self.P.look_end[1] - self.P.look_end[0])))
        yaw = 0.35 * to_wrist + 0.65 * to_table + look + math.radians(3.0) * a[13] * np.sin(2 * np.pi * t / 2.3 + ph[13])
        target = wrist + np.array([0.18, 0.0, 0.0])
        horizontal = np.hypot(target[:, 0] - eye[:, 0], target[:, 1] - eye[:, 1])
        raw_pitch = np.arctan2(target[:, 2] - eye[:, 2], horizontal)
        centre, width = math.radians(-45.0), math.radians(7.0)
        pitch = centre + width * np.tanh((raw_pitch - centre) / width)
        pitch = pitch + math.radians(2.0) * np.sin(2 * np.pi * t * 0.9 + ph[14]) + math.radians(0.8) * np.sin(2 * np.pi * t * 2.1 + ph[15])
        roll = math.radians(3.0) * np.sin(2 * np.pi * t / 4.7 + ph[0] + 1.0)
        return yaw, pitch, roll

    def camera_pose(self, t: np.ndarray) -> np.ndarray:
        """``T_room_cam`` ``(N, 4, 4)`` (OpenCV camera axes)."""
        yaw, pitch, roll = self.head_angles(t)
        f = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], axis=-1)
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(f, up)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        y = np.cross(f, x)
        R = np.stack([x, y, f], axis=-1) @ geo.rot_z(roll)
        return geo.make_T(R, self.eye(t))

    # ------------------------------------------------------------------ derived quantities

    def shoulders(self, t: np.ndarray) -> np.ndarray:
        """``(N, 2, 3)`` shoulders (slot 0 left, 1 right) from the head pose (skeleton model)."""
        return sk.shoulders_from_head(self.camera_pose(t), self.profile)

    def elbows(self, t: np.ndarray) -> np.ndarray:
        """``(N, 2, 3)`` elbows from the two-sphere construction with the scripted swivel."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        S = self.shoulders(t)
        out = np.zeros_like(S)
        for slot, side in enumerate(("left", "right")):
            W = self.wrist_pose(t, side)[:, :3, 3]
            out[:, slot] = sk.elbow_from_swivel(
                S[:, slot], W, self.profile["upper_arm_length_m"], self.profile["forearm_length_m"],
                self.swivel(t, side), side,
            )
        return out

    def reach(self, t: np.ndarray) -> np.ndarray:
        """``(N, 2)`` |wrist - shoulder| / (upper + forearm)."""
        S = self.shoulders(t)
        out = np.zeros(S.shape[:2])
        for slot, side in enumerate(("left", "right")):
            W = self.wrist_pose(t, side)[:, :3, 3]
            out[:, slot] = sk.reach_ratio(S[:, slot], W, self.profile["upper_arm_length_m"], self.profile["forearm_length_m"])
        return out


def finite_difference_rates(pose_fn: Callable[[np.ndarray], np.ndarray], t: np.ndarray, h_rot: float = 1e-4,
                            h_pos: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Body angular velocity ``(N, 3)`` (rad/s, in the moving frame) and world acceleration
    ``(N, 3)`` (m/s^2) of a pose function by central differences."""
    t = np.asarray(t, dtype=np.float64)
    Rm = pose_fn(t - h_rot)[:, :3, :3]
    Rp = pose_fn(t + h_rot)[:, :3, :3]
    omega = geo.so3_log(np.swapaxes(Rm, -1, -2) @ Rp) / (2 * h_rot)
    pm = pose_fn(t - h_pos)[:, :3, 3]
    p0 = pose_fn(t)[:, :3, 3]
    pp = pose_fn(t + h_pos)[:, :3, 3]
    acc = (pp - 2 * p0 + pm) / (h_pos * h_pos)
    return omega, acc
