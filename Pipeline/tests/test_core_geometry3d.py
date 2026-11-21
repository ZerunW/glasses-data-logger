"""Pure 3-D geometry used by the objects step: kNN, convex hulls and meshes, planes and polygons,
ICP, depth fusion, grasp detection."""

import math

import numpy as np
import pytest

from glassesdata_pipeline import geometry as geo
from glassesdata_pipeline.core import fusion as fu
from glassesdata_pipeline.core import grasp as gr
from glassesdata_pipeline.core import hull as hl
from glassesdata_pipeline.core import icp
from glassesdata_pipeline.core import plane as pl
from glassesdata_pipeline.core.knn import KNN, self_knn


def box_surface(size, n, seed=0):
    """Points and outward normals uniformly on the surface of a box centred at the origin."""
    size = np.asarray(size, dtype=float)
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size / 2
    Vh, F = hl.convex_hull_3d(V)
    return hl.sample_surface(Vh, F, n, seed)


# ---------------------------------------------------------------- kNN


def test_knn_flann_matches_brute_force():
    rng = np.random.default_rng(0)
    ref = rng.random((3000, 3)) + [1000.0, -2000.0, 5.0]  # large offset: FLANN runs centred float32
    q = rng.random((400, 3)) + [1000.0, -2000.0, 5.0]
    d_b, i_b = KNN(ref, "brute").query(q, 5)
    d_f, i_f = KNN(ref, "flann").query(q, 5)
    assert np.mean(i_b == i_f) > 0.99
    assert np.allclose(d_b[:, 0], np.linalg.norm(ref[i_b[:, 0]] - q, axis=1))
    assert np.all(np.diff(d_f, axis=1) >= -1e-12)
    d, i = self_knn(ref[:50], 3)
    assert d.shape == (50, 3) and not np.any(i == np.arange(50)[:, None]) and np.all(d > 0)
    assert KNN(ref[:2]).query(q[:3], 10)[0].shape == (3, 2)  # k clipped


# ---------------------------------------------------------------- meshes


def test_convex_hull_box_properties_and_obj_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    corners = np.array([[x, y, z] for x in (0, 2) for y in (0, 3) for z in (0, 4)], float)
    P = np.vstack([corners, rng.random((500, 3)) * [2, 3, 4], corners[:3]])  # + interior + duplicates
    V, F = hl.convex_hull_3d(P)
    assert len(V) == 8 and len(F) == 12 and hl.is_closed_manifold(F)
    vol, c = hl.mesh_volume_centroid(V, F)
    assert vol == pytest.approx(24.0) and np.allclose(c, [1, 1.5, 2])
    mass, com, I = hl.mesh_inertia(V, F, density=2.0)
    assert mass == pytest.approx(48.0)
    assert np.allclose(I, np.diag(mass / 12 * np.array([3**2 + 4**2, 2**2 + 4**2, 2**2 + 3**2])), atol=1e-9)
    assert np.all(hl.convex_signed_distance(V, F, P) <= 1e-9)
    hl.write_obj(tmp_path / "m.obj", V, F, "test")
    V2, F2 = hl.read_obj(tmp_path / "m.obj")
    assert np.allclose(V2, V, atol=1e-6) and np.array_equal(F2, F)
    pts, nrm = hl.sample_surface(V, F, 2000)
    assert np.all(np.abs(hl.convex_signed_distance(V, F, pts)) < 1e-9)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1)
    with pytest.raises(ValueError):
        hl.convex_hull_3d(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]))  # coplanar


def test_hull_of_grid_and_sphere():
    g = np.stack(np.meshgrid(*[np.arange(10.0)] * 3, indexing="ij"), -1).reshape(-1, 3)
    V, F = hl.convex_hull_3d(g)
    assert len(V) == 8 and hl.mesh_volume_centroid(V, F)[0] == pytest.approx(729.0)
    d = hl.fibonacci_directions(3000)
    V, F = hl.convex_hull_3d(d)
    assert hl.is_closed_manifold(F) and hl.mesh_volume_centroid(V, F)[0] == pytest.approx(4 / 3 * math.pi, rel=0.01)


def test_halfspace_intersection_and_trimmed_support_polytope():
    D = np.vstack([np.eye(3), -np.eye(3)])
    V, F = hl.halfspace_intersection(D, np.array([1, 2, 3, 1, 2, 3.0]), np.zeros(3))
    assert hl.mesh_volume_centroid(V, F)[0] == pytest.approx(2 * 4 * 6)
    # noisy box surface: the trimmed polytope is much closer to the true size than the hull
    size = np.array([0.065, 0.065, 0.11])
    P, N = box_surface(size, 3000)
    P = P + np.random.default_rng(2).normal(scale=0.003, size=P.shape)
    ext_hull = hl.mesh_extents(hl.hull_mesh(P)[0])
    V, F, info = hl.support_polytope(P, trim=24, extra_dirs=np.vstack([np.eye(3), -np.eye(3)]))
    ext_trim = hl.mesh_extents(V)
    assert info["method"] == "trimmed support polytope" and hl.is_closed_manifold(F)
    assert np.all(ext_hull - size > 0.012)  # +2 x ~2.5 sigma
    assert np.all(np.abs(ext_trim - size) < np.abs(ext_hull - size)) and np.all(np.abs(ext_trim - size) < 0.012)


# ---------------------------------------------------------------- planes and polygons


def test_plane_fit_ransac_and_horizontal_planes():
    rng = np.random.default_rng(3)
    floor = np.c_[rng.uniform(-2, 2, (3000, 2)), rng.normal(-1.45, 0.005, 3000)]
    table = np.c_[rng.uniform(0, 1, (2000, 2)), rng.normal(-0.70, 0.005, 2000)]
    wall = np.c_[np.full(1500, 2.0), rng.uniform(-2, 2, 1500), rng.uniform(-1.4, 1.0, 1500)]
    junk = rng.uniform(-2, 2, (800, 3))
    P = np.vstack([floor, table, wall, junk])
    n, off = pl.fit_plane(table)
    assert n[2] > 0.999 and -off / n[2] == pytest.approx(-0.70, abs=0.002)
    res = pl.ransac_plane(P, 0.015, max_tilt_deg=10)
    assert res["n_inliers"] >= 2900 and pl.tilt_deg(res["normal"]) < 1
    planes = pl.horizontal_planes(P, 0.015, max_planes=3, min_inliers=1000, max_tilt_deg=10)
    assert [round(p["z"], 2) for p in planes] == [-1.45, -0.7]  # the wall is not horizontal
    assert pl.plane_height(planes[1]["normal"], planes[1]["offset"], np.array([0.5, 0.5])) == pytest.approx(-0.7, abs=0.003)


def test_polygons_and_components():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    hull = pl.convex_polygon(np.vstack([sq, [[0.5, 0.5]]]))
    assert len(hull) == 4 and pl.polygon_area(hull) == pytest.approx(1.0)
    r = pl.convex_polygon_iou(sq, sq + [0.5, 0])
    assert r["iou"] == pytest.approx(0.5 / 1.5) and r["frac_a"] == pytest.approx(0.5)
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 1, (400, 2))
    b = rng.uniform(3, 4, (900, 2))
    m = pl.component_mask(np.vstack([a, b]), 0.1, seeds=np.array([[0.5, 0.5]]))
    assert m[:400].all() and not m[400:].any()
    assert pl.component_mask(np.vstack([a, b]), 0.1)[400:].all()  # no seed: the largest


# ---------------------------------------------------------------- ICP


def test_point_to_plane_icp_recovers_pose_and_dof():
    size = [0.16, 0.10, 0.07]
    M, N = box_surface(size, 4000, seed=5)
    T_true = geo.make_T(geo.so3_exp([0.05, -0.04, 0.12]), [0.012, -0.008, 0.01])
    S, _ = box_surface(size, 600, seed=6)
    S = S[S @ np.array([-1, 0, 1.0]) > -0.02]  # a partial view
    src = geo.transform(geo.inv_T(T_true), S) + np.random.default_rng(7).normal(scale=0.001, size=S.shape)
    res = icp.icp_point_to_plane(src, M, N, np.eye(4), max_correspondence=0.05)
    assert res["ok"] and res["fitness"] > 0.95
    assert np.linalg.norm(res["T"][:3, 3] - T_true[:3, 3]) < 0.002
    assert math.degrees(geo.rotation_angle(res["T"][:3, :3], T_true[:3, :3])) < 1.0
    # translation only: the rotation stays exactly as initialised
    T_t = geo.make_T(None, [0.01, 0.0, -0.005])
    res3 = icp.icp_point_to_plane(geo.transform(geo.inv_T(T_t), S), M, N, np.eye(4), dof="3")
    assert np.allclose(res3["T"][:3, :3], np.eye(3)) and np.linalg.norm(res3["T"][:3, 3] - T_t[:3, 3]) < 0.001
    # yaw + translation
    T_y = geo.make_T(geo.rot_z(0.1), [0.005, 0.004, 0.0])
    res4 = icp.icp_point_to_plane(geo.transform(geo.inv_T(T_y), S), M, N, np.eye(4), dof="4")
    assert abs(math.degrees(geo.rotation_angle(res4["T"][:3, :3], T_y[:3, :3]))) < 0.5
    assert abs(res4["T"][2, 0]) < 1e-9 and abs(res4["T"][2, 1]) < 1e-9  # no tilt
    # too few points
    assert not icp.icp_point_to_plane(S[:3], M, N)["ok"]


def test_align_similarity_and_average_poses():
    M, _ = box_surface([0.1, 0.06, 0.04], 2000, seed=8)
    s, R, t = 2.5, geo.rot_z(0.7), np.array([0.3, -0.2, 0.1])
    target = (s * M @ R.T + t)[: 1200]  # partial
    res = icp.align_similarity(M, target, n_yaw=8)
    assert res["s"] == pytest.approx(s, rel=0.01) and np.allclose(res["t"], t, atol=0.003)
    Ts = np.stack([geo.make_T(geo.so3_exp([0, 0, a]), [a, 0, 0]) for a in (-0.1, 0.0, 0.1, 0.5)])
    avg = icp.average_poses(Ts[:3])
    assert np.allclose(avg, np.eye(4), atol=1e-9)
    assert icp.average_poses(Ts)[0, 3] == pytest.approx(0.0) or icp.average_poses(Ts)[0, 3] == pytest.approx(0.1)


# ---------------------------------------------------------------- fusion


def test_point_filters_normals_and_completion():
    rng = np.random.default_rng(9)
    P = np.c_[rng.uniform(-0.1, 0.1, (2000, 2)), rng.normal(0.0, 0.001, 2000)]
    out = rng.uniform(-0.5, 0.5, (20, 3)) + [0, 0, 1.0]
    keep = fu.statistical_outlier_mask(np.vstack([P, out]), 12, 2.0)
    assert keep[:2000].mean() > 0.97 and not keep[2000:].any()
    N = fu.estimate_normals(P, 16, toward=[0, 0, 1.0])
    assert np.mean(N[:, 2]) > 0.99
    V, _ = fu.voxel_downsample(P, 0.02)
    assert 90 <= len(V) <= 130
    S = fu.mls_smooth(P + [0, 0, 0], 24, 1)
    assert np.std(S[:, 2]) < 0.4 * np.std(P[:, 2])
    top = np.c_[rng.uniform(-0.03, 0.03, (100, 2)), np.full(100, 0.1)]
    base = fu.extrude_to_plane(top, [0, 0, 1.0], 0.0, layers=1)
    assert len(base) == 200 and np.allclose(base[:100, 2], 0) and np.allclose(base[100:, 2], 0.05)
    R = geo.rot_z(0.4)[:2, :2]
    fp = rng.uniform([-0.08, -0.03], [0.08, 0.03], (500, 2)) @ R.T
    A = fu.footprint_axes(np.c_[fp, np.zeros(500)])
    ang = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    assert min(abs(ang - 22.92), abs(ang - 22.92 + 180), abs(ang - 22.92 - 180)) < 1.0
    assert np.allclose(A[:, 2], [0, 0, 1]) and np.linalg.det(A) == pytest.approx(1.0)


def test_tsdf_fuses_noisy_depth_of_a_box():
    """Render z-depth of a box on a plane from several views, add 3 % noise, fuse, compare."""
    cam = geo.Camera.fov_default(120, 160, 60.0)
    size = np.array([0.08, 0.06, 0.10])
    centre = np.array([0.0, 0.0, size[2] / 2])
    rays = cam.pixel_rays()
    rng = np.random.default_rng(10)
    vol = fu.TSDFVolume(centre - 0.1, centre + 0.1, 0.004, 0.03)
    for k, yaw in enumerate(np.linspace(-0.5, 0.5, 15)):
        eye = np.array([-0.6 * math.cos(yaw), -0.6 * math.sin(yaw), 0.55])
        T_wc = geo.look_at(eye, centre)
        o = T_wc[:3, 3]
        d = (rays.reshape(-1, 3) @ T_wc[:3, :3].T)
        # analytic ray-box intersection (slabs), z-depth = t because rays have z = 1 in the camera
        lo, hi = centre - size / 2, centre + size / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tn, tf = np.nanmax(np.minimum(t1, t2), axis=1), np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tn <= tf) & (tn > 0)
        depth = np.where(hit, tn, np.nan).reshape(cam.height, cam.width)
        mask = np.isfinite(depth)
        noisy = depth * (1 + rng.normal(scale=0.03, size=depth.shape))
        vol.integrate(noisy, fu.erode(mask, 1), cam, T_wc)
    P, N = vol.extract_surface(min_weight=4)
    from glassesdata_pipeline.synth.motion import box_signed_distance

    sd = box_signed_distance(P - centre, size)
    assert len(P) > 300 and abs(np.median(sd)) < 0.003 and np.percentile(np.abs(sd), 90) < 0.012  # 3 % noise = 2 cm per pixel
    # normals point out of the box
    assert np.mean(np.einsum("ij,ij->i", N, P - centre) > 0) > 0.8


# ---------------------------------------------------------------- grasps


def _hand_track(T, fps, grasp_at, release_at, move=True, close=0.06, open_=0.15):
    """Synthetic joints (T, 2, 21, 3): a rigid hand whose thumb/index aperture closes at
    ``grasp_at`` and opens at ``release_at``; it moves 0.2 m up in between when ``move``."""
    from glassesdata_pipeline import skeleton as sk

    rest = sk.rest_pose(None, "right")
    t = np.arange(T) / fps
    J = np.full((T, 2, 21, 3), np.nan)
    for i, ti in enumerate(t):
        s = np.clip((ti - grasp_at + 0.3) / 0.3, 0, 1) * (1 - np.clip((ti - release_at) / 0.3, 0, 1))
        ap = open_ + (close - open_) * s
        P = rest.copy()
        P[4] = P[8] + (P[4] - P[8]) / np.linalg.norm(P[4] - P[8]) * ap
        lift = 0.2 * np.clip((ti - grasp_at - 0.2) / 1.0, 0, 1) if move and ti < release_at else (0.2 if move and ti >= release_at else 0.0)
        J[i, 1] = P + [0.4, 0.0, lift]
    return J


def test_detect_grasps_needs_closing_and_moving():
    fps = 24.0
    J = _hand_track(144, fps, 2.0, 4.5, move=True)
    eps, rej = gr.detect_grasps(J, fps)
    assert len(eps) == 1 and eps[0]["hand"] == 1
    assert abs(eps[0]["onset"] - 48) <= 3 and abs(eps[0]["release"] - 108) <= 6
    J2 = _hand_track(144, fps, 2.0, 4.5, move=False)
    eps2, rej2 = gr.detect_grasps(J2, fps)
    assert eps2 == [] and rej2 and rej2[0]["reason"] == "hand did not move while closed"
    eps3, rej3 = gr.detect_grasps(J, fps, near_surface=lambda h, f, p: False)
    assert eps3 == [] and rej3[0]["reason"] == "not near a surface"
    assert np.allclose(gr.grasp_point(J[60], "thumb_index")[1], 0.5 * (J[60, 1, 4] + J[60, 1, 8]))


def test_hand_rigid_motion_and_contact_grasped():
    rng = np.random.default_rng(11)
    from glassesdata_pipeline import skeleton as sk

    S = sk.rest_pose(None, "right")
    Ts = [geo.make_T(geo.so3_exp([0.1 * k, -0.05 * k, 0.2 * k]), [0.01 * k, 0.02, -0.01 * k]) for k in range(10)]
    J = np.stack([geo.transform(T, S) for T in Ts]) + rng.normal(scale=0.002, size=(10, 21, 3))
    H = gr.hand_rigid_motion(J, range(10))
    for k in range(10):
        rel = H[k] @ geo.inv_T(H[0])
        true = Ts[k] @ geo.inv_T(Ts[0])
        assert math.degrees(geo.rotation_angle(rel[:3, :3], true[:3, :3])) < 3.0
    sm = gr.smooth_poses(H, np.ones(10, bool), 1.0)
    assert sm.shape == H.shape and np.all(np.isfinite(sm))
    T = 60
    d = np.full((T, 2, 5), 0.2)
    d[20:40, 1, 1:3] = 0.01  # two fingertips in contact
    eps = [{"hand": 1, "onset": 18, "release": 42}]
    g = gr.contact_grasped(d, eps, np.zeros(T), np.zeros((T, 2)))
    assert g[18:43, 1].all() and not g[:18].any() and not g[43:].any() and not g[:, 0].any()
    assert not gr.contact_grasped(np.full((T, 2, 5), 0.2), eps, np.zeros(T), np.zeros((T, 2))).any()


def test_grasp_speed_and_windows_use_timestamps():
    fps = 24.0
    J = _hand_track(144, fps, 2.0, 4.5, move=True)
    t = np.arange(144) / fps
    eps, _ = gr.detect_grasps(J, fps)
    eps_t, _ = gr.detect_grasps(J, fps, t=t)  # uniform timestamps: the same result
    assert [(e["onset"], e["release"]) for e in eps] == [(e["onset"], e["release"]) for e in eps_t]
    # a frame drop: 2 s of positions moving at 0.5 m/s, with frames 10-19 missing from the video
    keep = np.r_[0:10, 20:48]
    tt = t[keep]
    pts = np.column_stack([0.5 * tt, 0 * tt, 0 * tt])
    v = gr.speed(pts, fps, 2, tt)
    np.testing.assert_allclose(v, 0.5, rtol=1e-9)
    assert gr.speed(pts, fps, 2)[9] > 1.0  # frame counting overestimates across the drop
    # the move window is time based: a drop shortens it in frames
    J_drop = J[np.r_[0:60, 72:144]]
    eps_d, _ = gr.detect_grasps(J_drop, fps, t=np.r_[t[0:60], t[72:144]])
    assert len(eps_d) == 1 and abs(eps_d[0]["onset"] - 48) <= 3
