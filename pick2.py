"""Choose an object's SAM3D mesh from several views, before the pipeline's own SAM3D step.

SAM3D reconstructs from one image, and a poor view (top-down, partly hidden) gives a
wrong shape. So: build candidates on up to six good frames (views.candidates), keep the
consensus one (smallest mean shape distance to the others after Sim(3) registration: wrong
shapes disagree with each other, right ones agree), and keep SAM3D's own depth-conditioned
size (FoundationPose's scale search inflated already-fat meshes). Writes the files the
pipeline checks, so it skips SAM3D, the transform and scale estimation.

    python pick.py <object output dir>        (called by the pipeline when V2D_MESH_PICK=1)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import views  # noqa: E402
import score as S  # noqa: E402

SEED = 0


def build(d: Path, t: int, weights: str, texture: bool = False) -> Path:
    """SAM3D candidate on frame t, transformed like the pipeline does. Seeded, so the textured
    rebuild of the winner has the candidate's geometry; texture baking is most of SAM3D's
    time and only the winner needs it (the pick compares geometry)."""
    from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
    from v2d.pipelines.run_ego_wilor import _apply_sam3d_transform
    out = d / "views" / f"s{t}" / ("tex" if texture else "")
    if (out / "mesh.obj").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    mask = d / f"masks/1/{t:06d}.png"
    run_image_to_mesh(image_path=str(d / f"frames/{t:06d}.png"), mask_path=str(mask),
                      mesh_path=str(out / "textured_mesh.obj"), transform_path=str(out / "mesh_transform.json"),
                      intrinsics_path=str(out / "mesh_intrinsics.json"), weights_dir=weights, seed=SEED,
                      with_texture_baking=texture, with_mesh_postprocess=True,
                      depth_path=str(d / f"depth/{t:06d}.png"),
                      depth_intrinsics_path=str(d / "intrinsics_stable.json"), depth_mask_path=str(mask))
    _apply_sam3d_transform(str(out / "textured_mesh.obj"), str(out / "mesh_transform.json"),
                           str(out / "mesh.obj"))
    return out


def consensus(dirs: list[Path]) -> Path:
    M = []
    for p in dirs:
        m = trimesh.load(p / "mesh.obj", force="mesh")
        M.append((np.asarray(m.vertices), np.asarray(m.faces)))
    n = len(M)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = S.M._object_chamfer_cm(M[i], M[j], 0)
    return dirs[int(np.argmin(D.sum(1)))]


def main():
    d = Path(sys.argv[1]).resolve()
    weights = str(Path(__file__).resolve().parents[1] / "video_to_data/reconstruction/data/weights/sam3d")
    dirs = [build(d, t, weights) for t in views.candidates(d)]
    pick = consensus(dirs) if len(dirs) > 2 else dirs[0]
    best = build(d, int(pick.name[1:]), weights, texture=True)
    mesh_dir = d / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    for f in best.iterdir():                     # SAM3D outputs incl. texture/material files
        if f.is_file():
            shutil.copy(f, mesh_dir / f.name)
    # the transformed mesh with its material, as both "pretransformed" and "scaled" (scale 1)
    m = trimesh.load(best / "mesh.obj")
    m.export(d / "mesh_pretransformed.obj")
    m.export(d / "mesh_scaled.obj")
    (d / "scale.json").write_text(json.dumps({"scale": 1.0}))
    (d / "views" / "pick.json").write_text(json.dumps({"frame": pick.name, "candidates": [p.name for p in dirs]}))
    print(f"{d.name}: picked {pick.name} of {[p.name for p in dirs]}")


if __name__ == "__main__":
    main()
