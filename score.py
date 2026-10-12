"""Score a predicted object trajectory against a public Track 3 episode, locally.

Reuses the kit's own Kaggle scoring functions (metric_code/track_3/MPPE.py; all six
scorer files share the same code) so local numbers follow the same registration,
alignment and metric formulas as the leaderboard.

Two things differ from Kaggle, because the host keeps them private:
  - Object symmetries are not applied. Cups and the pot lid are scored with full
    orientation, so local scores are equal to or slightly worse than Kaggle's.
  - The host's per-object anchor and AUC surface points are rebuilt here from the
    public scan: anchor = area-weighted surface centroid, points = surface samples.

Usage:
    .venv/bin/python t3/score.py --episode 41 --mode oracle
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

WS = Path(__file__).resolve().parents[1]
KIT = WS / "v2d_submission_kit"
DATA = WS / "data" / "track_3" / "public"

sys.path.insert(0, str(KIT))
from v2dlb.trajectory import episode_parquet, read_episode  # noqa: E402

_spec = importlib.util.spec_from_file_location("t3metric", KIT / "metric_code" / "track_3" / "MPPE.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

POSE_METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm")
AUC_POINTS = 2000


def load_reference(episode: int):
    """Object names and true poses (T, B, 7) xyzw, invisible frames dropped."""
    traj = read_episode(episode_parquet(DATA, episode), episode)
    return traj.object_names, traj.pose_xyzw


def load_scan(name: str):
    """The public scan of one object as (vertices, faces), metres, in its pose frame."""
    folder = DATA / "mesh" / name
    path = folder / f"{name}.glb"
    if not path.exists():
        path = folder / f"{name}_visual.glb"
    mesh = trimesh.load(path, force="mesh")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def score_episode(episode, pred_poses, pred_meshes, ref_poses, ref_meshes):
    """Kaggle's registered Track 3 scoring for one episode, one rollout world.

    pred_poses, ref_poses: (T, B, 7) xyzw. pred_meshes, ref_meshes: list of (V, F) per slot.
    Returns {metric: value} for the five pose metrics plus cd_o_cm (mean over objects).
    """
    achieved = M._normalise_quaternions(pred_poses[:, None].copy(), "prediction", ValueError)
    reference = M._normalise_quaternions(ref_poses[:, None].copy(), "reference", ValueError)
    vertices, axes, orders, chamfers = [], [], [], []
    for slot, (cand, ref) in enumerate(zip(pred_meshes, ref_meshes)):
        seed = M._object_seed(episode, slot)
        anchor, _ = M._surface_moments(*ref)
        cand_anchor, rotation = M._register_object_body(cand, ref, anchor, seed)
        achieved[:, :, slot] = M._convert_object_body(achieved[:, :, slot], cand_anchor, rotation)
        reference[:, :, slot, :3] += np.einsum(
            "...ij,j->...i", M._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        vertices.append(M._sample_mesh_surface(*ref, AUC_POINTS, seed) - anchor)
        axes.append(np.zeros(3))
        orders.append(0)
        chamfers.append(M._object_chamfer_cm(ref, cand, seed))
    achieved = M._align_registered_scene(achieved, reference, axes)
    result = {m: M._registered_pose_metric(m, achieved, reference, vertices, axes, orders)
              for m in POSE_METRICS}
    result["cd_o_cm"] = float(np.mean(chamfers))
    return result


def frame_errors(episode, pred_poses, pred_meshes, ref_poses, ref_meshes):
    """Per-frame position error (cm) and orientation error (deg), shape (T, B), after the
    same mesh registration and first-frame alignment the scorer applies."""
    achieved = M._normalise_quaternions(pred_poses[:, None].copy(), "prediction", ValueError)
    reference = M._normalise_quaternions(ref_poses[:, None].copy(), "reference", ValueError)
    axes = []
    for slot, (cand, ref) in enumerate(zip(pred_meshes, ref_meshes)):
        anchor, _ = M._surface_moments(*ref)
        cand_anchor, rotation = M._register_object_body(cand, ref, anchor, M._object_seed(episode, slot))
        achieved[:, :, slot] = M._convert_object_body(achieved[:, :, slot], cand_anchor, rotation)
        reference[:, :, slot, :3] += np.einsum(
            "...ij,j->...i", M._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        axes.append(np.zeros(3))
    achieved = M._align_registered_scene(achieved, reference, axes)
    pos = 100 * np.linalg.norm(achieved[:, 0, :, :3] - reference[:, 0, :, :3], axis=-1)
    ang = np.degrees(M._quaternion_geodesic_angle(achieved[:, 0, :, 3:], reference[:, 0, :, 3:]))
    return pos, ang


def perturb(ref_poses: np.ndarray, rng: np.random.Generator,
            kind: str = "jitter", size: float = 0.01) -> np.ndarray:
    """Return a corrupted copy of the true poses (T, B, 7) xyzw, imitating reconstruction error.

    kind="jitter": independent noise per frame, std `size` metres (shaky tracking).
    kind="drift":  offset growing linearly to `size` metres by the last frame (camera drift).
    kind="scale":  motion from the first frame scaled by 1 + `size` (wrong depth scale).
    Only positions change; rotations stay exact.
    """
    pred = ref_poses.copy()
    pos = pred[..., :3]
    if kind == "jitter":
        pos += rng.normal(0.0, size, pos.shape)
    elif kind == "drift":
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pos += np.linspace(0.0, size, len(pos))[:, None, None] * direction
    elif kind == "scale":
        pos[:] = pos[:1] + (pos - pos[:1]) * (1.0 + size)
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--mode", choices=["oracle", "frozen", "perturb"], default="oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=["jitter", "drift", "scale"], default="jitter")
    parser.add_argument("--size", type=float, default=0.01)
    args = parser.parse_args()

    names, ref_poses = load_reference(args.episode)
    scans = [load_scan(n) for n in names]
    if args.mode == "oracle":
        pred = ref_poses.copy()
    elif args.mode == "frozen":
        pred = np.repeat(ref_poses[:1], len(ref_poses), axis=0)
    else:
        pred = perturb(ref_poses, np.random.default_rng(args.seed), args.kind, args.size)

    scores = score_episode(args.episode, pred, scans, ref_poses, scans)
    print(f"episode {args.episode} {names} frames={len(ref_poses)} mode={args.mode}")
    for key, value in scores.items():
        print(f"  {key:14s} {value:.4f}")


if __name__ == "__main__":
    main()
