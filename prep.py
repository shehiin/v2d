"""On the GPU server: fetch Track 3 public cam A videos, parquets, meshes; simplify meshes.

    ~/ws/v2d/.venv/bin/python prep.py --root ~/ws/v2d/data
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HF = "https://huggingface.co/datasets/nvidia/video_to_data_challenge"
TREE = "https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/main/track_3/public"
FACES = 20000


def fetch(rel: str, root: Path):
    out = root / rel
    if out.exists() and out.stat().st_size > 0:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(out), f"{HF}/resolve/main/{rel}"], check=True)


def listing(sub: str) -> list[str]:
    raw = subprocess.run(["curl", "-sf", f"{TREE}/{sub}?recursive=1"], check=True,
                         capture_output=True, text=True).stdout
    return [x["path"] for x in json.loads(raw) if x["type"] == "file"]


def simplify(src: Path, dst: Path):
    import pymeshlab as ml
    import trimesh
    obj = src.with_suffix(".obj")
    if not obj.exists():
        trimesh.load(src, force="mesh").export(obj)
    ms = ml.MeshSet()
    ms.load_new_mesh(str(obj))
    ms.meshing_merge_close_vertices()
    if ms.current_mesh().face_number() > FACES:
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=FACES, preserveboundary=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(dst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root

    for rel in ["track_3/public/meta/info.json", "track_3/public/meta/episodes_metadata.jsonl",
                "track_3/public/meta/camera_calibration.json"]:
        fetch(rel, root)
    episodes = json.loads((root / "track_3/public/meta/info.json").read_text())["episode_indices"]
    for e in episodes:
        fetch(f"track_3/public/data/chunk-000/episode_{e:06d}.parquet", root)
        fetch(f"track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_{e:06d}.mp4", root)
    for rel in listing("mesh"):
        fetch(rel, root)

    for folder in sorted((root / "track_3/public/mesh").iterdir()):
        name = folder.name
        src = folder / f"{name}.glb"
        if not src.exists():
            src = folder / f"{name}_visual.glb"
        dst = folder / "s20k" / f"{name}.obj"
        if not dst.exists():
            simplify(src, dst)
        print(name, "ok")


if __name__ == "__main__":
    main()
