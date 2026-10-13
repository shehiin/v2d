"""Per-frame tracking confidence: IoU between the tracked mesh's silhouette and the SAM2 mask.

FoundationPose can drift or flip while SAM2 still segments the object correctly, so a
low overlap between "where the pose puts the object" and "where the object is" flags an
unreliable frame. CPU only; reads a finished pipeline output dir.

    python conf.py out/e12_blue_cup [more dirs]     -> writes <dir>/conf.npy, (T, 2)

Columns: IoU, and coverage = fraction of the SAM2 mask inside the silhouette. A hand
covering the object shrinks the mask but not the silhouette, so IoU drops on correct
poses during grasps; coverage stays high there and only drops when the pose is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

DOWN = 4  # work at 1/4 resolution


def silhouette(V, F, T, K, shape):
    """Filled mask of the mesh's projected triangles (camera at origin, z forward)."""
    P = V @ T[:3, :3].T + T[:3, 3]
    m = np.zeros(shape, np.uint8)
    if (P[:, 2] <= 0.05).any():
        return m
    uv = (P[:, :2] / P[:, 2:]) * [K[0], K[1]] + [K[2], K[3]]
    tri = np.round(uv[F] / DOWN).astype(np.int32)
    cv2.fillPoly(m, list(tri), 1)
    return m


def main():
    for d in map(Path, sys.argv[1:]):
        res = d / "result_slam_gravity_aligned"
        npz = np.load(res / "result.npz")
        To = npz["object_to_camera_transform"]
        K = npz["camera_intrinsics"]
        mesh = trimesh.load(res / "mesh.obj", force="mesh")
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        masks = sorted((d / "masks/1").glob("*.png"))
        out = np.zeros((len(To), 2))
        for t, f in enumerate(masks[:len(To)]):
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (m.shape[1] // DOWN, m.shape[0] // DOWN), interpolation=cv2.INTER_NEAREST) > 0
            s = silhouette(V, F, To[t], K, m.shape) > 0
            inter, union = (m & s).sum(), (m | s).sum()
            out[t] = (inter / union if union else 0.0, inter / m.sum() if m.sum() else 0.0)
        np.save(res / "conf.npy", out)
        print(f"{d.name}: median IoU {np.median(out[:, 0]):.2f}, coverage {np.median(out[:, 1]):.2f}")


if __name__ == "__main__":
    main()
