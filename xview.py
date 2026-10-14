"""Pick the best SAM3D candidate mesh by how well it fits the OTHER candidate views.

A mesh built from one frame always fits that frame; a correct shape also fits the other
views. For every (candidate mesh, candidate frame) pair, FoundationPose's scale search
registers the mesh in that frame and reports its best scale and fit score (mask IoU +
depth agreement). The mesh with the best mean score on frames other than its own wins,
scaled by the median of its per-frame scales (a multi-view size estimate).

    python xview.py out6/e21_blue_cup [more dirs]  -> <dir>/views/pick.json, views/mesh_pick.obj
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "video_to_data/reconstruction"
PAT = re.compile(r"Level (\d+)/(\d+): best_scale=([\d.]+)\s+score=([-\d.]+)")


def fit(d: Path, mesh: Path, t: int, out: Path) -> tuple[float, float]:
    """FoundationPose scale search of `mesh` in frame t: (best scale, score)."""
    code = ("from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale as r; "
            f"r(mesh_path='{mesh}', rgb_path='{d}/frames/{t:06d}.png', depth_path='{d}/depth/{t:06d}.png', "
            f"mask_path='{d}/masks/1/{t:06d}.png', intrinsics_path='{d}/intrinsics_stable.json', "
            f"weights_dir='{REC}/data/weights/foundation_pose', scale_path='{out}', "
            "lo=0.5, hi=2.0, n_samples=7, n_levels=3)")
    p = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", code], cwd=REC,
                       capture_output=True, text=True)
    hits = PAT.findall(p.stdout + p.stderr)
    if not hits:
        raise RuntimeError(f"no score for {mesh} frame {t}:\n{(p.stdout + p.stderr)[-1500:]}")
    _, _, s, score = hits[-1]
    return float(s), float(score)


def main():
    for d in map(Path, sys.argv[1:]):
        d = d.resolve()
        views = d / "views"
        frames = sorted(int(p.name[1:]) for p in views.glob("f*") if (p / "mesh.obj").exists())
        table = {}
        for a in frames:
            for b in frames:
                s, score = fit(d, views / f"f{a}/mesh.obj", b, views / f"f{a}/scale_on_{b}.json")
                table[f"{a}:{b}"] = (s, score)
                print(f"{d.name} mesh f{a} on frame {b}: scale {s:.3f} score {score:.3f}", flush=True)
        def cross(a):
            return np.mean([table[f"{a}:{b}"][1] for b in frames if b != a])
        best = max(frames, key=cross)
        scale = float(np.median([table[f"{best}:{b}"][0] for b in frames]))
        m = trimesh.load(views / f"f{best}/mesh.obj", force="mesh")
        m.apply_scale(scale)
        m.export(views / "mesh_pick.obj")
        (views / "pick.json").write_text(json.dumps({"frame": best, "scale": scale,
                                                     "cross": {a: cross(a) for a in frames},
                                                     "table": table}, indent=1))
        print(f"{d.name}: pick f{best} scale {scale:.3f} cross {[round(cross(a), 3) for a in frames]}")


if __name__ == "__main__":
    main()
