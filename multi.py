"""Inside the task-library-loader container: load one episode with all its objects.

The stock ego_recon loader builds a task with a single object. Track 3 scores every
object in an episode, and FlashCHORD and the retargeter take a list of object bodies,
so this subclass merges the per-object bundles of one episode into one sequence.
All bundles of an episode share camera, hands and gravity alignment (batch.py seeds
the second object from the first), so hands come from the first bundle and the
scene alignment is the first object's.

    python multi.py --bundles e21_blue_cup/result_slam_gravity_aligned e21_water_pitcher/... \
        --names blue_cup water_pitcher --sequence_name e21 --fps 20.254 \
        --mano_model_dir /data/mano --mesh_dir ... --object_model_root ... \
        --output_dir ... --no_ground_align --save
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v2d.task_library_loader.lib import ego_recon_loader as E
from v2d.task_library_loader.lib.dataset_loader_base import DatasetLoaderBase, SequenceInfo


@dataclass
class MultiSource(E.EgoReconSequenceSource):
    """First object's fields (used for hands and scene alignment) plus every object."""

    objects: list[E.EgoReconSequenceSource] = None
    bodies: list[str] = None


def bundle(result_dir: Path) -> E.EgoReconSequenceSource:
    npz_path = result_dir / "result.npz"
    with np.load(npz_path) as npz:
        scale = float(npz["object_scale"]) if "object_scale" in npz else 1.0
    return E.EgoReconSequenceSource(result_dir=result_dir, npz_path=npz_path,
                                    mesh_path=result_dir / "mesh.obj", object_scale=scale)


class MultiLoader(E.EgoReconDatasetLoader):

    def list_sequences(self, args: Any) -> list[SequenceInfo]:
        objs = [bundle(Path(b)) for b in args.bundles]
        bodies = [E.make_usd_safe(n) for n in args.names]
        first = objs[0]
        src = MultiSource(result_dir=first.result_dir, npz_path=first.npz_path,
                          mesh_path=first.mesh_path, object_scale=first.object_scale,
                          objects=objs, bodies=bodies)
        return [SequenceInfo(sequence_id=args.sequence_name, raw_motion_file=str(first.npz_path),
                             object_name="+".join(args.names), object_body_names=bodies,
                             source=src)]

    def load_object_data(self, sequence_info: SequenceInfo) -> dict[str, Any]:
        R, t = self._scene_align(sequence_info)  # first object's, applied to all
        out = {}
        for body, obj in zip(sequence_info.source.bodies, sequence_info.source.objects):
            with np.load(obj.npz_path) as npz:
                poses = npz["object_to_world_transform"].astype(np.float64)
            aligned = poses.copy()
            aligned[:, :3, :3] = np.einsum("ij,njk->nik", R, poses[:, :3, :3])
            aligned[:, :3, 3] = (R @ poses[:, :3, 3].T).T + t
            pos, aa = E.poses_to_root_position_and_axis_angle(aligned)
            out[body] = (aligned, pos, aa, None)
        return out

    def load_object_meshes(self, sequence_info: SequenceInfo, device: torch.device):
        src = sequence_info.source
        scales = {o.object_scale for o in src.objects}
        if len(scales) != 1:
            raise ValueError(f"objects need one shared mesh scale, got {scales}")
        paths = {b: str(o.mesh_path) for b, o in zip(src.bodies, src.objects)}
        return E.load_meshes_to_device(paths, device, vertex_scale=scales.pop())

    def get_fps(self) -> float:
        return float(self._args.fps)

    def get_object_mesh_paths(self, sequence_info: SequenceInfo) -> list[str]:
        src = sequence_info.source
        out = []
        for body, obj in zip(src.bodies, src.objects):
            one = SequenceInfo(sequence_id=sequence_info.sequence_id,
                               raw_motion_file=str(obj.npz_path), object_name=body,
                               object_body_names=[body], source=obj)
            out.append(str(self._ensure_object_asset(one)))
        return out

    def get_object_urdf_paths(self, sequence_info: SequenceInfo) -> list[str]:
        root = Path(self._args.object_model_root)
        return [str(root / f"{b}_rigid.urdf") for b in sequence_info.source.bodies]


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    DatasetLoaderBase.add_common_args(p, dataset_root=E.DEFAULT_DATASET_ROOT,
                                      object_model_root=E.EGO_URDF_DIR, mesh_dir=E.EGO_MESH_DIR,
                                      output_dir=E.LOADED_SAVE_DIR)
    p.add_argument("--bundles", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--sequence_name", required=True)
    p.add_argument("--fps", type=float, required=True, help="real video frame rate")
    p.add_argument("--no_ground_align", action="store_true")
    p.add_argument("--ground_z_offset", type=float, default=0.0)
    p.add_argument("--ground_yaw_offset_deg", type=float, default=0.0)
    a = p.parse_args()
    if len(a.bundles) != len(a.names):
        p.error("one name per bundle")
    return a


if __name__ == "__main__":
    MultiLoader().run(parse_args())
