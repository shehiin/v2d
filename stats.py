"""What do the objects do in each public episode? Travel, peak speed, rotation, gaps.

    .venv/bin/python t3/stats.py
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

import score as L

FPS = 20.25

episodes = json.load(open(L.DATA / "meta" / "info.json"))["episode_indices"]
tasks = {json.loads(l)["episode_index"]: json.loads(l)["task"]
         for l in open(L.DATA / "meta" / "episodes_metadata.jsonl")}
print(f"{'ep':>3s} {'task':20s} {'object':18s} {'frames':>6s} {'gap':>4s} "
      f"{'range cm':>8s} {'path cm':>8s} {'vmax cm/s':>9s} {'rot deg':>7s}")
for e in episodes:
    traj = L.read_episode(L.episode_parquet(L.DATA, e), e)
    pose = traj.pose_xyzw
    for slot, name in enumerate(traj.object_names):
        p = pose[:, slot, :3]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.diff(traj.frame_index) / FPS
        rot = Rotation.from_quat(pose[:, slot, 3:])
        angle = np.degrees((rot[0].inv() * rot).magnitude()).max()
        print(f"{e:3d} {tasks[e]:20s} {name:18s} {len(p):6d} {traj.dropped_frames:4d} "
              f"{100 * np.linalg.norm(p - p[0], axis=1).max():8.1f} {100 * step.sum():8.1f} "
              f"{100 * (step / dt).max():9.1f} {angle:7.1f}")
