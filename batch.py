"""On the GPU server: run the baseline ego reconstruction for every (episode, object).

Two workers take whole episodes from a queue. Within an episode the second object
reuses the first object's camera, depth, SLAM and hand outputs, so only detection,
masks and pose tracking are redone. Each run picks the freest GPU when it starts.

    ~/ws/v2d/.venv/bin/python batch.py --root ~/ws/v2d [--workers 2] [--episodes 0 2 11]
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq

PROMPTS = {
    "beige_cup": "a beige cup",
    "blue_cup": "a blue cup",
    "mini_dust_pan": "a small dustpan",
    "mini_sweeper": "a small hand brush",
    "plastic_dish_rack": "a dish rack",
    "water_pitcher": "a water pitcher",
    "white_pot": "a white pot",
    "white_pot_lid": "a pot lid",
    "wooden_piece_1": "a flat wooden board",
    "wooden_piece_2": "a flat wooden board",
}
# outputs that depend on the object; everything else is shared within an episode
OBJECT_SPECIFIC = {"dino_detections.json", "object_track.json", "sam2_prompts.json",
                   "prompts_overlay.png", "masks", "masks_overlay.mp4",
                   "poses", "poses_smoothed"}
lock = threading.Lock()


def log(msg: str):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def objects(root: Path, episode: int) -> list[str]:
    path = root / f"data/track_3/public/data/chunk-000/episode_{episode:06d}.parquet"
    first = pq.read_table(path, columns=["observation.objects"]).column(0)[0].as_py()
    return [o["name"] for o in first]


def seed_from(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in OBJECT_SPECIFIC or item.name.startswith("result"):
            continue
        target = dst / item.name
        if target.exists():
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, target)


def run_one(root: Path, episode: int, name: str) -> bool:
    out = root / f"out/e{episode}_{name}"
    if (out / "result_slam_gravity_aligned" / "result.npz").exists():
        log(f"skip e{episode} {name} (done)")
        return True
    data = root / "data/track_3/public"
    gpu = subprocess.run([str(root / "bin/pickgpu")], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}",
               V2D_GPU=gpu, V2D_FP_W="1014", V2D_FP_H="760")
    cmd = [str(root / ".venv/bin/python"), "modules/v2d_pipelines/run_ego_reconstruction.py",
           "--video", str(data / f"videos/chunk-000/observation.images.ego_cam_a/episode_{episode:06d}.mp4"),
           "--object_prompt", PROMPTS[name],
           "--object_mesh", str(data / f"mesh/{name}/s20k/{name}.obj"),
           "--skip_object_scale_estimation",
           "--output_dir", str(out),
           "--reference_frame", "0", "--undistort", "--hand_tracking", "hamer",
           "--run_droid_slam", "--run_gravity_alignment"]
    log(f"start e{episode} {name} gpu{gpu}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".log"), "w") as f:
        code = subprocess.run(cmd, cwd=root / "video_to_data/reconstruction", env=env,
                              stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"{'ok' if code == 0 else 'FAIL'} e{episode} {name}")
    return code == 0


def worker(root: Path, jobs: queue.Queue):
    while True:
        try:
            episode = jobs.get_nowait()
        except queue.Empty:
            return
        names = objects(root, episode)
        first = None
        for name in names:
            if first is not None:
                seed_from(first, root / f"out/e{episode}_{name}")
            if run_one(root, episode, name) and first is None:
                first = root / f"out/e{episode}_{name}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--episodes", type=int, nargs="*")
    args = p.parse_args()
    import json
    episodes = args.episodes or json.loads(
        (args.root / "data/track_3/public/meta/info.json").read_text())["episode_indices"]
    jobs = queue.Queue()
    for e in episodes:
        jobs.put(e)
    threads = []
    for i in range(args.workers):
        t = threading.Thread(target=worker, args=(args.root, jobs))
        t.start()
        threads.append(t)
        time.sleep(120)  # stagger so workers don't pick the same GPU at once
    for t in threads:
        t.join()
    log("ALL DONE")


if __name__ == "__main__":
    main()
