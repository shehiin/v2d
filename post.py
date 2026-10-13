"""Clean a reconstructed object trajectory with physical priors from the public data.

Priors (all 20 public episodes, 33 objects):
  - every object is still for at least 10 frames at the start (median 41);
  - the dish rack never moves (under 0.4 cm, 1.5 deg);
  - true motion stays under ~24 deg and ~6.5 cm per frame.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

STATIC = {"plastic_dish_rack"}
HOLD = 10          # frames held still at the start
MAX_TURN = 30.0    # deg per frame; faster is a tracking jump
MAX_STEP = 0.08    # m per frame


def reject_jumps(pose: np.ndarray) -> np.ndarray:
    """Hold the last accepted pose through frames that jump faster than physics allows."""
    out = pose.copy()
    for t in range(1, len(out)):
        turn = np.degrees((R.from_quat(out[t - 1, 3:]).inv() * R.from_quat(pose[t, 3:])).magnitude())
        step = np.linalg.norm(pose[t, :3] - out[t - 1, :3])
        if turn > MAX_TURN:
            out[t, 3:] = out[t - 1, 3:]
        if step > MAX_STEP:
            out[t, :3] = out[t - 1, :3]
    return out


def smooth(pose: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over time for positions and (sign-aligned) quaternions."""
    if sigma <= 0:
        return pose
    out = pose.copy()
    out[:, :3] = gaussian_filter1d(pose[:, :3], sigma, axis=0, mode="nearest")
    q = pose[:, 3:].copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    out[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return out


def clean(pose: np.ndarray, name: str, *, hold=True, static=True, jumps=True, sigma=0.0) -> np.ndarray:
    """pose (T, 7) xyzw in the reconstruction world frame -> cleaned copy."""
    out = pose.copy()
    if jumps:
        out = reject_jumps(out)
    if static and name in STATIC:
        out[:] = out[min(HOLD, len(out) - 1)]
        return out
    out = smooth(out, sigma)
    if hold:
        out[:HOLD] = out[min(HOLD, len(out) - 1)]
    return out
