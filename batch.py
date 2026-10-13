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
CFG = {"out": "out", "calib": False}
HEAVY = ("frames", "depth", "anycalib", "masks_overlay.mp4", "hamer_overlay.mp4",
         "hamer_aligned_overlay.mp4", "hamer_aligned_filled_overlay.mp4")


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
        # hard links: shared stages cost no extra disk
        if item.is_dir():
            shutil.copytree(item, target, copy_function=os.link)
        else:
            os.link(item, target)


def calib_undistort(root: Path, video: Path, out: Path):
    """Write the anycalib stage outputs from the real cam A calibration so the pipeline
    skips AnyCalib. The stored video is the sensor image rotated 180 degrees
    (ego rotation_deg = 180), so undistort in sensor orientation and rotate back."""
    import json
    import cv2
    import numpy as np
    dst = out / "anycalib"
    if (dst / "undistorted.mp4").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    cal = json.loads((root / "data/track_3/public/meta/camera_calibration.json").read_text())
    cam = cal["cameras"]["ego_cam_a"]
    K = np.array(cam["intrinsic_matrix"], dtype=np.float64)
    dist = np.array(cam["distortion_coefficients"], dtype=np.float64)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)
    tmp = dst / "undistorted.tmp.mp4"
    enc = subprocess.Popen([str(root / "bin/ffmpeg"), "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(tmp)],
                           stdin=subprocess.PIPE)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.rotate(cv2.remap(cv2.rotate(img, cv2.ROTATE_180), mx, my, cv2.INTER_LINEAR),
                         cv2.ROTATE_180)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    # intrinsics of the rotated, undistorted image
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": W - 1 - K[0, 2], "cy": H - 1 - K[1, 2],
            "width": W, "height": H}
    for name in ("intrinsics.json", "undistorted_intrinsics.json"):
        (dst / name).write_text(json.dumps(intr, indent=4))
    (dst / "distortion.json").write_text(json.dumps({"model": "none", "params": []}))
    tmp.rename(dst / "undistorted.mp4")


def cleanup(dirs: list[Path]):
    """Drop regenerable heavy intermediates once every object of an episode is done."""
    for d in dirs:
        for name in HEAVY:
            p = d / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()


def run_one(root: Path, episode: int, name: str, tries: int = 3) -> bool:
    """Run with retries; each try re-picks the freest GPU and resumes from cached stages."""
    for attempt in range(tries):
        if attempt:
            time.sleep(60)
        if run_try(root, episode, name):
            return True
    return False


def run_try(root: Path, episode: int, name: str) -> bool:
    out = root / CFG["out"] / f"e{episode}_{name}"
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
    if CFG["calib"]:
        calib_undistort(root, Path(cmd[3]), out)
    with open(out.with_suffix(".log"), "a") as f:
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
        dirs = [root / CFG["out"] / f"e{episode}_{n}" for n in names]
        first = None
        ok = True
        for name, d in zip(names, dirs):
            if first is not None:
                seed_from(first, d)
            done = run_one(root, episode, name)
            ok &= done
            if done and first is None:
                first = d
        if ok:
            cleanup(dirs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--out", default="out", help="output folder under root")
    p.add_argument("--calib", action="store_true", help="undistort with the real cam A calibration")
    args = p.parse_args()
    CFG.update(out=args.out, calib=args.calib)
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
