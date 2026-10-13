"""Score a reconstruction bundle (result.npz + mesh.obj) against a public episode.

    .venv/bin/python t3/recon.py --episode 0 --result pot_dir lid_dir [--slots 0 1]

Each bundle holds one object; --slots says which object of the episode each is. A
single object is scored alone: mixing in another object's true pose would mix two world
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


def load(result: Path, frames: int):
    pose, valid, mesh = bundle_poses(result)
    if len(pose) != frames:
        raise SystemExit(f"{result}: {len(pose)} frames, reference has {frames}")
    # hold the last valid pose through invalid frames, as a rollout would
    for t in range(len(pose)):
        if not valid[t]:
            pose[t] = pose[t - 1] if t else pose[np.argmax(valid)]
    return pose, valid, mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--result", type=Path, nargs="+", required=True,
                   help="one bundle per slot in --slots order; bundles must share a camera track")
    p.add_argument("--slots", type=int, nargs="+", default=None)
    args = p.parse_args()

    names, ref = S.load_reference(args.episode)
    scans = [S.load_scan(n) for n in names]
    slots = args.slots or list(range(len(args.result)))
    loaded = [load(r, len(ref)) for r in args.result]
    pred = np.stack([l[0] for l in loaded], axis=1)
    scores = S.score_episode(args.episode, pred, [l[2] for l in loaded],
                             ref[:, slots], [scans[i] for i in slots])
    label = " + ".join(f"{names[i]} ({l[1].mean():.0%} valid)" for i, l in zip(slots, loaded))
    print(f"episode {args.episode}: {label}")
    for k, v in scores.items():
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main()
