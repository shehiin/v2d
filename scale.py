"""Put MoGe depth on the stereo pair's metric scale.

MoGe's depth has the right shape but a wrong global scale (8 to 16% too far with
AnyCalib intrinsics, about 20% more with the real focal length). The ego stereo pair
(cams B, C, 75 mm baseline) is metric by construction. Per frame: triangulate stereo
points, express them in cam A, project into the undistorted cam A image, and take the
median ratio of stereo depth to MoGe depth at those pixels. The ratio is smoothed over
time and multiplied into every depth PNG (DepthImage encoding: 65535 / (d + 1)).

    python scale.py --out out/e12_blue_cup --episode 12 [--split public]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import stereo as ST


def depth_read(path: Path) -> np.ndarray:
    inv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    with np.errstate(divide="ignore"):
        return 65535.0 / inv - 1.0


def depth_write(path: Path, d: np.ndarray):
    px = np.clip(65535.0 / (d + 1.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), px)


def frame_ratio(pts_a: np.ndarray, K: np.ndarray, depth: np.ndarray) -> tuple[float, int]:
    """Median stereo/MoGe depth ratio over stereo points that land inside the image."""
    z = pts_a[:, 2]
    u = K[0, 0] * pts_a[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_a[:, 1] / z + K[1, 2]
    h, w = depth.shape
    ok = (z > 0.15) & (z < 2.0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    m = depth[v[ok].astype(int), u[ok].astype(int)]
    good = np.isfinite(m) & (m > 0.1)
    if good.sum() < 500:
        return float("nan"), int(good.sum())
    return float(np.median(z[ok][good] / m[good])), int(good.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="pipeline output dir with depth/")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--split", default="public")
    p.add_argument("--step", type=int, default=3, help="measure every n-th frame")
    a = p.parse_args()

    done = a.out / "depth_scale.json"
    if done.exists():
        return
    cam, ext = ST.calib()
    K = cam["a"][0]  # undistorted cam A keeps the calibrated K (batch.calib_undistort)
    vids = ST.ROOT.parent / a.split / "videos/chunk-000"
    vb = vids / f"observation.images.ego_cam_b/episode_{a.episode:06d}.mp4"
    vc = vids / f"observation.images.ego_cam_c/episode_{a.episode:06d}.mp4"
    files = sorted((a.out / "depth").glob("*.png"))
    idx = list(range(0, len(files), a.step))
    raw = {}
    for i in idx:
        r, n = frame_ratio(ST.points_in_a(vb, vc, i, cam, ext), K, depth_read(files[i]))
        if np.isfinite(r):
            raw[i] = r
    if not raw:
        raise SystemExit("no usable stereo frames")
    # robust temporal smoothing: running median, then interpolate to every frame
    keys = np.array(sorted(raw))
    vals = np.array([raw[k] for k in keys])
    med = np.array([np.median(vals[max(0, j - 3):j + 4]) for j in range(len(vals))])
    scale = np.interp(np.arange(len(files)), keys, med)
    for f, s in zip(files, scale):
        depth_write(f, depth_read(f) * s)
    done.write_text(json.dumps({"median": float(np.median(vals)), "per_frame": scale.tolist(),
                                "raw": {int(k): v for k, v in raw.items()}}))
    print(f"scale median {np.median(vals):.3f} range {vals.min():.3f}-{vals.max():.3f}")


if __name__ == "__main__":
    main()
