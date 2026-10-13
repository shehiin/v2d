"""Score a predicted object trajectory against a public Track 3 episode, locally.

Reuses the kit's own Kaggle scoring functions (metric_code/track_3/MPPE.py; all six
scorer files share the same code) so local numbers follow the same registration,
alignment and metric formulas as the leaderboard.

Two things differ from Kaggle, because the host keeps them private:
  - Object symmetries (pass names=) use axes estimated from the public scans, since
    the host's axes are defined on its private scans.
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


# Kaggle README symmetries: cups spin freely about their axis, the lid has a half turn.
AXIAL = {"beige_cup", "blue_cup"}
HALF_TURN = {"white_pot_lid"}


def symmetry(name, mesh, anchor):
    """(axis, order) in the scan frame. The host defines axes on its private scans; here
    they come from the public scan's shape: the cup axis is the principal direction whose
    spread differs from the other two, the lid axis is its thinnest direction."""
    if name not in AXIAL and name not in HALF_TURN:
        return np.zeros(3), 0
    pts = M._sample_mesh_surface(*mesh, 20000, 0) - anchor
    w, V = np.linalg.eigh(pts.T @ pts / len(pts))
    if name in HALF_TURN:
        return V[:, 0], 2
    gaps = [abs(w[i] - w[(i + 1) % 3]) + abs(w[i] - w[(i + 2) % 3]) for i in range(3)]
    return V[:, int(np.argmax(gaps))], M._SYMMETRY_ANGLES


def score_episode(episode, pred_poses, pred_meshes, ref_poses, ref_meshes, names=None):
    """Kaggle's registered Track 3 scoring for one episode, one rollout world.

    pred_poses, ref_poses: (T, B, 7) xyzw. pred_meshes, ref_meshes: list of (V, F) per slot.
    names: object names per slot; enables the cup and lid symmetries.
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
        axis, order = symmetry(names[slot], ref, anchor) if names else (np.zeros(3), 0)
        axes.append(axis)
        orders.append(order)
        chamfers.append(M._object_chamfer_cm(ref, cand, seed))
    achieved = M._align_registered_scene(achieved, reference, axes)
    result = {m: M._registered_pose_metric(m, achieved, reference, vertices, axes, orders)
              for m in POSE_METRICS}
    result["cd_o_cm"] = float(np.mean(chamfers))
    return result


def frame_errors(episode, pred_poses, pred_meshes, ref_poses, ref_meshes, names=None):
    """Per-frame position error (cm) and orientation error (deg), shape (T, B), after the
    same mesh registration, symmetry and first-frame alignment the scorer applies."""
    achieved = M._normalise_quaternions(pred_poses[:, None].copy(), "prediction", ValueError)
    reference = M._normalise_quaternions(ref_poses[:, None].copy(), "reference", ValueError)
    axes, orders = [], []
    for slot, (cand, ref) in enumerate(zip(pred_meshes, ref_meshes)):
        anchor, _ = M._surface_moments(*ref)
        cand_anchor, rotation = M._register_object_body(cand, ref, anchor, M._object_seed(episode, slot))
        achieved[:, :, slot] = M._convert_object_body(achieved[:, :, slot], cand_anchor, rotation)
        reference[:, :, slot, :3] += np.einsum(
            "...ij,j->...i", M._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        axis, order = symmetry(names[slot], ref, anchor) if names else (np.zeros(3), 0)
        axes.append(axis)
        orders.append(order)
    achieved = M._align_registered_scene(achieved, reference, axes)
    pos = 100 * np.linalg.norm(achieved[:, 0, :, :3] - reference[:, 0, :, :3], axis=-1)
    ang = np.degrees(M._symmetry_orientation_errors(achieved, reference, axes, orders)[:, 0])
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

    scores = score_episode(args.episode, pred, scans, ref_poses, scans, names)
    print(f"episode {args.episode} {names} frames={len(ref_poses)} mode={args.mode}")
    for key, value in scores.items():
        print(f"  {key:14s} {value:.4f}")


if __name__ == "__main__":
    main()
