"""Metric depth for ego cam A from the ego stereo pair (cams B, C).

The stored ego videos are the sensor images rotated 180 degrees (rotation_deg = 180).
All geometry here works in sensor orientation: rotate frames back, rectify B/C, run
SGBM, then express the 3D points in cam A and project them into the stored cam A image.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "track_3" / "public"


def calib(root: Path = ROOT):
    c = json.loads((root / "meta" / "camera_calibration.json").read_text())["cameras"]
    cam = {k: (np.array(c[f"ego_cam_{k}"]["intrinsic_matrix"], np.float64),
               np.array(c[f"ego_cam_{k}"]["distortion_coefficients"], np.float64),
               tuple(c[f"ego_cam_{k}"]["resolution"])) for k in "abc"}
    ext = {k: np.array(c[f"ego_cam_{k}"]["extrinsics_to_stereo_left"], np.float64) for k in "ac"}
    return cam, ext


def frame(video: Path, i: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    if not ok:
        raise IndexError(i)
    return img


def rectifier(cam, ext, invert: bool):
    """Rectification maps for B (left) and C (right) in sensor orientation.
    ext['c'] relates the C and B frames; invert picks which direction it maps."""
    Kb, db, size = cam["b"]
    Kc, dc, _ = cam["c"]
    T = ext["c"] if not invert else np.linalg.inv(ext["c"])
    R, t = T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()
    Rb, Rc, Pb, Pc, Q, _, _ = cv2.stereoRectify(Kb, db, Kc, dc, size, R, t, alpha=0)
    mb = cv2.initUndistortRectifyMap(Kb, db, Rb, Pb, size, cv2.CV_32FC1)
    mc = cv2.initUndistortRectifyMap(Kc, dc, Rc, Pc, size, cv2.CV_32FC1)
    return mb, mc, Rb, Pb, Q


def row_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Median vertical offset of matched features (px) and median horizontal disparity."""
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(left, None)
    k2, d2 = orb.detectAndCompute(right, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.array([k1[x.queryIdx].pt for x in m])
    p2 = np.array([k2[x.trainIdx].pt for x in m])
    dy, dx = p1[:, 1] - p2[:, 1], p1[:, 0] - p2[:, 0]
    keep = np.abs(dy) < 20
    return float(np.median(np.abs(dy[keep]))), float(np.median(dx[keep]))


def points_in_a(video_b: Path, video_c: Path, i: int, cam, ext, num_disp: int = 128):
    """Metric 3D points (N, 3) in cam A coordinates from stereo frame i.
    In the stored orientation C is the left camera and ext['c'] maps C to B: p_B = T p_C."""
    Kc, dc, size = cam["c"]
    Kb, db, _ = cam["b"]
    T = ext["c"]
    R, t = T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()
    Rl, Rr, Pl, Pr, Q, _, _ = cv2.stereoRectify(Kc, dc, Kb, db, size, R, t, alpha=0)
    ml = cv2.initUndistortRectifyMap(Kc, dc, Rl, Pl, size, cv2.CV_32FC1)
    mr = cv2.initUndistortRectifyMap(Kb, db, Rr, Pr, size, cv2.CV_32FC1)
    L = cv2.remap(cv2.cvtColor(frame(video_c, i), cv2.COLOR_BGR2GRAY), *ml, cv2.INTER_LINEAR)
    Rt = cv2.remap(cv2.cvtColor(frame(video_b, i), cv2.COLOR_BGR2GRAY), *mr, cv2.INTER_LINEAR)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=5,
                                 P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(L, Rt).astype(np.float32) / 16.0
    xyz = cv2.reprojectImageTo3D(disp, Q)
    ok = (disp > 1.0) & np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0.1) & (xyz[..., 2] < 3.0)
    p_rect = xyz[ok]
    p_c = p_rect @ Rl          # rectified left -> C (Rl maps C to rectified)
    p_b = p_c @ T[:3, :3].T + T[:3, 3]
    A = ext["a"]               # p_B = A p_A
    p_a = (p_b - A[:3, 3]) @ A[:3, :3]
    return p_a
