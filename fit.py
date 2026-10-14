"""Fit a reconstructed mesh's size and proportions to metric depth over many frames.

SAM3D builds the mesh from one image, so its proportions are off (a cup 1.5-1.9x too wide
but 1.2x too tall), and a single global scale cannot fix that. The tracked poses put every
frame's masked depth points (stereo-scaled) into the object's own frame. Fitting a per-axis
scale and an offset so the mesh surface passes through those points fixes size and aspect.

    python fit.py out6/e21_blue_cup [more dirs]  -> <dir>/result_slam_gravity_aligned/mesh_fit.obj
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def depth_read(path: Path) -> np.ndarray:
    inv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    with np.errstate(divide="ignore"):
        return 65535.0 / inv - 1.0


def object_points(d: Path, To: np.ndarray, K: np.ndarray, step: int = 3, per_frame: int = 400):
    """Masked depth points from every `step`-th frame, in the object frame."""
    rng = np.random.default_rng(0)
    fx, fy, cx, cy = K
    pts = []
    depths = sorted((d / "depth").glob("*.png"))
    masks = sorted((d / "masks/1").glob("*.png"))
    for t in range(0, min(len(To), len(depths), len(masks)), step):
        m = cv2.imread(str(masks[t]), cv2.IMREAD_GRAYSCALE) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # drop edge pixels
        z = depth_read(depths[t])
        v, u = np.nonzero(m & (z > 0.1) & (z < 2.0))
        if len(u) < 50:
            continue
        k = rng.choice(len(u), min(per_frame, len(u)), replace=False)
        u, v = u[k], v[k]
        zz = z[v, u]
        P = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], 1)
        T = np.linalg.inv(To[t])
        pts.append(P @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts)


def axes(V):
    """Principal axes of the mesh (columns): its long, middle and short directions."""
    c = V.mean(0)
    return c, np.linalg.eigh(np.cov((V - c).T))[1]


def stretch(X, c, R, s, o):
    """Scale points X by s along the principal axes R about c, then shift by o."""
    return ((X - c) @ R) * s @ R.T + c + o


def fit(V, F, pts):
    """Per-axis scale along the mesh's principal axes plus an offset that minimise robust
    point-to-surface distance. Returns (scale(3), offset(3), inlier share)."""
    mesh = trimesh.Trimesh(V, F, process=False)
    S = mesh.sample(30000)
    c, R = axes(V)
    # first pass: keep points near the current surface (drops hand / background leaks)
    d0, _ = cKDTree(S).query(pts)
    keep = d0 < max(0.03, np.percentile(d0, 70))
    P = pts[keep]

    def resid(x):
        s, o = x[:3], x[3:]
        tree = cKDTree(stretch(S, c, R, s, o))
        return tree.query(P)[0]

    x0 = np.r_[np.ones(3), np.zeros(3)]
    r = least_squares(resid, x0, loss="huber", f_scale=0.005, diff_step=1e-3,
                      bounds=([0.4] * 3 + [-0.1] * 3, [1.6] * 3 + [0.1] * 3))
    return r.x[:3], r.x[3:], keep.mean()


def main():
    for d in map(Path, sys.argv[1:]):
        res = d / "result_slam_gravity_aligned"
        npz = np.load(res / "result.npz")
        mesh = trimesh.load(res / "mesh.obj", force="mesh")
        V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)
        pts = object_points(d, npz["object_to_camera_transform"], npz["camera_intrinsics"])
        s, o, share = fit(V, F, pts)
        c, R = axes(V)
        out = trimesh.Trimesh(stretch(V, c, R, s, o), F, process=False)
        out.export(res / "mesh_fit.obj")
        (res / "mesh_fit.json").write_text(json.dumps({"scale": s.tolist(), "offset": o.tolist(),
                                                       "points": len(pts), "inliers": share}))
        print(f"{d.name}: {len(pts)} pts ({share:.0%} used) scale {np.round(s, 2)} "
              f"size cm {np.round(mesh.extents * 100)} -> {np.round(out.extents * 100)}")


if __name__ == "__main__":
    main()
