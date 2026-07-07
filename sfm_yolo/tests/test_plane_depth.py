"""Synthetic-scene test for the below-plane depth core.

Builds a metric depth map of a tilted road with a known-depth rectangular dip
and checks that :func:`pothole_depth_from_depthmap` recovers that depth.
"""

from __future__ import annotations

import numpy as np

from sfm_yolo.src.geometry.plane_depth import pothole_depth_from_depthmap


def _synthetic_depthmap(depth_true: float, *, noise_m: float = 0.0, seed: int = 0):
    W = H = 448
    fx = fy = 400.0
    cx = cy = 224.0
    n_up = np.array([0.0, -0.37, -0.93]); n_up /= np.linalg.norm(n_up)  # road up-normal
    gravity_cam = -n_up
    P0 = np.array([0.0, 0.0, 1.5])                                       # road ~1.5 m ahead
    c_road = float(n_up @ P0)

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rx = (us - cx) / fx
    ry = (vs - cy) / fy
    rz = np.ones_like(rx)
    ndotr = n_up[0] * rx + n_up[1] * ry + n_up[2] * rz

    bbox = (200.0, 200.0, 248.0, 248.0)
    in_hole = (us >= bbox[0]) & (us <= bbox[2]) & (vs >= bbox[1]) & (vs <= bbox[3])
    c_plane = np.where(in_hole, c_road - depth_true, c_road)             # floor lower by depth
    depth_map = (c_plane / ndotr).astype(np.float32)                    # Z along optical axis

    if noise_m > 0:
        rng = np.random.default_rng(seed)
        depth_map = depth_map + rng.normal(0, noise_m, depth_map.shape).astype(np.float32)

    return depth_map, bbox, dict(fx=fx, fy=fy, cx=cx, cy=cy, gravity_cam=gravity_cam)


def test_recovers_known_depth_clean():
    depth_true = 0.05
    dmap, bbox, intr = _synthetic_depthmap(depth_true)
    res = pothole_depth_from_depthmap(dmap, bbox, **intr)
    assert abs(res.depth_m - depth_true) < 0.01, res.as_dict()
    assert res.confidence > 0.3


def test_recovers_known_depth_noisy():
    depth_true = 0.07
    dmap, bbox, intr = _synthetic_depthmap(depth_true, noise_m=0.002, seed=1)
    res = pothole_depth_from_depthmap(dmap, bbox, **intr)
    assert abs(res.depth_m - depth_true) < 0.015, res.as_dict()


def test_flat_road_reads_near_zero():
    dmap, bbox, intr = _synthetic_depthmap(0.0)
    res = pothole_depth_from_depthmap(dmap, bbox, **intr)
    assert res.depth_m < 0.01, res.as_dict()


if __name__ == "__main__":
    for depth_true in (0.03, 0.05, 0.07, 0.10):
        dmap, bbox, intr = _synthetic_depthmap(depth_true, noise_m=0.002)
        res = pothole_depth_from_depthmap(dmap, bbox, **intr)
        print(f"true={depth_true*100:5.1f} cm -> recovered={res.depth_m*100:5.1f} cm "
              f"(conf={res.confidence:.2f}, {res.notes})")
