"""Score a policy rollout (kit evaluator parquet) against a public episode's ground truth.

The rollout's achieved object poses are mapped to video frames by reference_frame (training
at motion_speed 1.0 on a 20 fps task makes step k play frame k); frames without a row hold
the previous pose. Meshes are our reconstructions, as in the submission.

    .venv/bin/python t3/roll.py --episode 2 --rollout runs/e2/res_1048576/episode_000002.parquet \
        --meshes out7/e2_white_pot/result_slam_gravity_aligned out7/e2_white_pot_lid/...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh

import score as S

POSE = ["achieved_position_x", "achieved_position_y", "achieved_position_z",
        "achieved_quaternion_x", "achieved_quaternion_y", "achieved_quaternion_z", "achieved_quaternion_w"]


def rollout_poses(path: Path, frames: int, bodies: int) -> tuple[np.ndarray, float]:
    """(frames, bodies, 7) xyzw poses from world 0, and the fraction of frames with a row."""
    t = pq.read_table(path).to_pandas()
    t = t[t.environment_id == 0].sort_values(["step", "object_body_id"])
    pose = np.full((frames, bodies, 7), np.nan)
    for f, g in t.groupby("reference_frame", sort=True):
        if 0 <= f < frames:
            first = g[g.step == g.step.min()].sort_values("object_body_id")
            pose[f] = first[POSE].to_numpy()
    have = ~np.isnan(pose[:, 0, 0])
    for f in range(frames):                      # hold through frames the rollout skipped
        if not have[f]:
            pose[f] = pose[f - 1] if f else pose[np.argmax(have)]
    return pose, have.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--meshes", type=Path, nargs="+", required=True, help="one bundle dir per object, slot order")
    a = p.parse_args()
    names, ref = S.load_reference(a.episode)
    meshes = [trimesh.load(m / "mesh.obj", force="mesh") for m in a.meshes]
    meshes = [(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)) for m in meshes]
    pred, cover = rollout_poses(a.rollout, len(ref), len(names))
    scores = S.score_episode(a.episode, pred, meshes, ref, [S.load_scan(n) for n in names], names)
    print(f"episode {a.episode}: rollout covers {cover:.0%} of frames")
    for k, v in scores.items():
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main()
