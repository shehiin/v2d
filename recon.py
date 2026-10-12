"""Score a reconstruction bundle (result.npz + mesh.obj) against a public episode.

    .venv/bin/python t3/recon.py --episode 41 --result path/to/result_dir [--slot 0]

The bundle holds one object; --slot picks which object of the episode it is. That
object is scored alone: mixing in the other object's true pose would mix two world
frames (reconstruction vs mocap), so RPE needs both objects reconstructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import score as S


def bundle_poses(result_dir: Path):
    """(T, 7) xyzw object-in-world poses, (T,) validity, and the bundle mesh (V, F)."""
    npz = np.load(result_dir / "result.npz")
    T = npz["object_to_world_transform"]
    valid = npz["object_is_valid"].astype(bool)
    pose = np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=1)
    mesh = trimesh.load(result_dir / "mesh.obj", force="mesh")
    return pose, valid, (np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--slot", type=int, default=0)
    args = p.parse_args()

    names, ref = S.load_reference(args.episode)
    scans = [S.load_scan(n) for n in names]
    pose, valid, mesh = bundle_poses(args.result)
    if len(pose) != len(ref):
        raise SystemExit(f"bundle has {len(pose)} frames, reference has {len(ref)}")

    # hold the last valid pose through invalid frames, as a rollout would
    for t in range(len(pose)):
        if not valid[t]:
            pose[t] = pose[t - 1] if t else pose[np.argmax(valid)]

    s = args.slot
    scores = S.score_episode(args.episode, pose[:, None], [mesh], ref[:, [s]], [scans[s]])
    print(f"episode {args.episode} {names[args.slot]} valid={valid.mean():.0%}")
    for k, v in scores.items():
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main()
