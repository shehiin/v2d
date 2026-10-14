"""Try SAM3D on several frames of one object, to see how much the view matters.

Candidate frames: large object mask, little overlap with the hand masks, not cut by the
image border; the best one from each sixth of the video. Each candidate's mesh is put in
metric camera coordinates with SAM3D's own transform, as the pipeline does.

    python views.py out6/e21_blue_cup [more dirs]   -> <dir>/views/f<frame>/mesh.obj
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def candidates(d: Path, n: int = 6) -> list[int]:
    obj = sorted((d / "masks/1").glob("*.png"))
    hands = [sorted((d / f"masks/{k}").glob("*.png")) for k in (2, 3) if (d / f"masks/{k}").is_dir()]
    score = np.zeros(len(obj))
    for t, f in enumerate(obj):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0
        if m.sum() < 500 or m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        ring = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        touch = sum((cv2.imread(str(h[t]), cv2.IMREAD_GRAYSCALE) > 0)[ring].sum() for h in hands if t < len(h))
        score[t] = m.sum() / (1.0 + touch / m.sum() * 10)
    thirds = np.array_split(np.arange(len(obj)), n)
    return [int(p[np.argmax(score[p])]) for p in thirds if score[p].max() > 0]


def main():
    from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
    from v2d.pipelines.run_ego_wilor import _apply_sam3d_transform
    weights = str(Path(__file__).resolve().parents[1] / "video_to_data/reconstruction/data/weights/sam3d")
    for d in map(Path, sys.argv[1:]):
        d = d.resolve()
        for t in candidates(d):
            out = d / "views" / f"f{t}"
            if (out / "mesh.obj").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            img, mask, depth = d / f"frames/{t:06d}.png", d / f"masks/1/{t:06d}.png", d / f"depth/{t:06d}.png"
            run_image_to_mesh(image_path=str(img), mask_path=str(mask), mesh_path=str(out / "raw.obj"),
                              transform_path=str(out / "transform.json"),
                              intrinsics_path=str(out / "intrinsics.json"), weights_dir=weights,
                              with_texture_baking=False, with_mesh_postprocess=True,
                              depth_path=str(depth), depth_intrinsics_path=str(d / "intrinsics_stable.json"),
                              depth_mask_path=str(mask))
            _apply_sam3d_transform(str(out / "raw.obj"), str(out / "transform.json"), str(out / "mesh.obj"))
            print(f"{d.name} frame {t}: done", flush=True)


if __name__ == "__main__":
    main()
