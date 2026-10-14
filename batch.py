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
    "wooden_piece_1": "a wooden plank",
    "wooden_piece_2": "a wooden plank",
    "wooden_spoon": "a wooden spoon",
}
# outputs that depend on the object; everything else is shared within an episode
OBJECT_SPECIFIC = {"dino_detections.json", "object_track.json", "sam2_prompts.json",
                   "prompts_overlay.png", "masks", "masks_overlay.mp4",
                   "poses", "poses_smoothed", "box_cands.json",
                   # SAM3D mesh outputs belong to one object
                   "mesh", "mesh_pretransformed.obj", "mesh_scaled.obj", "scale.json",
                   "material_0.png", "material.mtl", "views"}
# first prompt in boxes.py per object; preferred when confident enough
PREFERRED = {"water_pitcher": "a tall white jug", "white_pot": "a white saucepan with a long handle",
             "white_pot_lid": "a round white lid"}
lock = threading.Lock()
CFG = {"out": "out", "calib": False, "stereo": False, "keep": False, "src": None, "ref": "0", "sam3d": False}
# outputs a tracking-only rerun must redo (everything else is copied from --from)
TRACKING = {"poses", "poses_smoothed", "depth_scale.json"}
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
    skips AnyCalib. The calibration already applies to the stored (upright) images:
    the stereo check gives 1 px row error this way vs 7 to 12 px with a 180 rotation."""
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
    # keep roughly the field of view the pipeline saw with AnyCalib (f ~1000): reusing the
    # calibrated K (f 1150) crops the edges, where ego hands sit (e0: no hands in frame 0)
    N, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0.15, (W, H), centerPrincipalPoint=True)
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, N, (W, H), cv2.CV_32FC1)
    tmp = dst / "undistorted.tmp.mp4"
    enc = subprocess.Popen([str(root / "bin/ffmpeg"), "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(tmp)],
                           stdin=subprocess.PIPE)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    intr = {"fx": N[0, 0], "fy": N[1, 1], "cx": N[0, 2], "cy": N[1, 2],
            "width": W, "height": H}
    for name in ("intrinsics.json", "undistorted_intrinsics.json"):
        (dst / name).write_text(json.dumps(intr, indent=4))
    (dst / "distortion.json").write_text(json.dumps({"model": "none", "params": []}))
    tmp.rename(dst / "undistorted.mp4")


def iou(a, b) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a, b) -> float:
    """Fraction of box a's area that lies inside box b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return w * h / area if area > 0 else 0.0


def drop_groups(c: list[dict]) -> list[dict]:
    """Drop boxes that cover two separate candidates of the same object (e.g. one box around
    both wooden boards): a box holding two disjoint objects is a group, not an object."""
    def group(x):
        parts = [y for y in c if y is not x and inside(y["box"], x["box"]) > 0.9
                 and inside(x["box"], y["box"]) < 0.6]
        return any(iou(p["box"], q["box"]) < 0.1 for i, p in enumerate(parts) for q in parts[i + 1:])
    kept = [x for x in c if not group(x)]
    return kept or c


def choose(cands: dict[str, list[dict]]) -> dict[str, dict]:
    """Pick one candidate per object from boxes.py output.
    cands[name] is a list of {"box": [x0, y0, x1, y1], "score": s, "prompt": p}, best first.
    Returns {name: candidate}.
    Objects with the most confident top box choose first; a box another object already took
    (IoU > 0.8) is skipped, so two objects never track the same thing. Nesting (lid inside
    the pot's box) stays allowed. Within an object the first prompt in boxes.py is preferred
    when it clears 0.4: it encodes what the mesh needs (the pot's handle fixes its rotation)."""
    order = sorted(cands, key=lambda n: -(cands[n][0]["score"] if cands[n] else 0))
    taken, out = [], {}
    for name in order:
        c = drop_groups(cands[name])
        if not c:
            raise RuntimeError(f"no detection for {name}")
        preferred = [x for x in c if x["prompt"] == PREFERRED.get(name) and x["score"] >= 0.4]
        free = [x for x in preferred + c if all(iou(x["box"], t) <= 0.8 for t in taken)]
        out[name] = free[0] if free else c[0]
        taken.append(out[name]["box"])
    return out


def pick_boxes(root: Path, video: Path, names: list[str], dirs: list[Path]):
    """Detect candidates for all objects on the frame the pipeline will use, choose one box
    per object, and pre-write each dino_detections.json so the pipeline skips its own DINO."""
    if all((d / "dino_detections.json").exists() for d in dirs):
        return
    cands_path = dirs[0] / "box_cands.json"
    if not cands_path.exists():
        w = "/w"
        subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
                        "-v", f"{root}:{w}",
                        "v2d_grounding_dino:latest", "python", f"{w}/t3/boxes.py",
                        "--video", f"{w}/{video.relative_to(root)}",
                        "--weights", f"{w}/video_to_data/reconstruction/data/weights/grounding_dino",
                        "--out", f"{w}/{cands_path.relative_to(root)}", "--names", *names],
                       env=dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}"),
                       check=True, capture_output=True)
    import json
    picked = choose(json.loads(cands_path.read_text()))
    for name, d in zip(names, dirs):
        c = picked[name]
        x0, y0, x1, y1 = c["box"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "dino_detections.json").write_text(json.dumps(
            [{"label": c["prompt"], "confidence": c["score"],
              "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}], indent=2))
        log(f"box {d.name}: {c['prompt']!r} {c['score']:.2f}")


def stereo_depth(root: Path, episode: int, out: Path):
    """MoGe depth on our undistorted video with the calibrated intrinsics, then rescaled to
    the stereo pair's metric scale (scale.py). The pipeline sees depth/ and skips MoGe."""
    if (out / "depth_scale.json").exists():
        return
    env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}")
    if not (out / "depth").is_dir():
        # write to depth_tmp and rename when done, so an interrupted run never looks finished
        rec = root / "video_to_data/reconstruction"
        shutil.rmtree(out / "depth_tmp", ignore_errors=True)
        code = ("from v2d.moge.docker.run_video_to_depth import run_video_to_depth as r; "
                f"r(video_path='{out}/anycalib/undistorted.mp4', depth_folder='{out}/depth_tmp', "
                f"intrinsics_folder='{out}/intrinsics', weights_path='{rec}/data/weights/moge', "
                f"input_intrinsics_path='{out}/anycalib/undistorted_intrinsics.json')")
        subprocess.run([str(root / ".venv/bin/python"), "-c", code], cwd=rec, env=env,
                       check=True, capture_output=True)
        (out / "depth_tmp").rename(out / "depth")
    r = subprocess.run([str(root / ".venv/bin/python"), str(root / "t3/scale.py"),
                        "--out", str(out), "--episode", str(episode)],
                       cwd=root / "t3", env=env, check=True, capture_output=True, text=True)
    log(f"depth e{episode}: {r.stdout.strip()}")


def copy_run(src: Path, dst: Path):
    """Hard-link a finished object dir, minus tracking outputs, for a tracking-only rerun."""
    if dst.exists():
        return
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in TRACKING or item.name.startswith("result"):
            # the depth scale record stays with depth maps it already scaled
            if not (item.name == "depth_scale.json" and (src / "depth").is_dir()):
                continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, copy_function=os.link)
        else:
            os.link(item, dst / item.name)


def visible_frame(out: Path) -> int:
    """Frame where the object's SAM2 mask is largest: its least occluded view."""
    import cv2
    files = sorted((out / "masks/1").glob("*.png"))
    area = [int((cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0).sum()) for f in files]
    return int(max(range(len(area)), key=area.__getitem__))


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


PER_FRAME = ["frames", "depth", "masks/1", "masks/2", "masks/3", "poses", "poses_smoothed"]


def drop_partial(out: Path, video: Path):
    """The pipeline treats an existing stage folder as done, so a try killed mid-stage (e.g. out of
    memory) leaves a short folder that every retry then skips. Remove per-frame folders that hold
    fewer files than the video has frames."""
    import cv2
    n = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    for s in PER_FRAME:
        d = out / s
        if d.is_dir() and len(list(d.iterdir())) < n:
            log(f"{out.name}: {s} has {len(list(d.iterdir()))}/{n} frames, redoing it")
            shutil.rmtree(d)


def run_try(root: Path, episode: int, name: str) -> bool:
    out = root / CFG["out"] / f"e{episode}_{name}"
    if (out / "result_slam_gravity_aligned" / "result.npz").exists():
        log(f"skip e{episode} {name} (done)")
        return True
    data = root / "data/track_3/public"
    drop_partial(out, data / f"videos/chunk-000/observation.images.ego_cam_a/episode_{episode:06d}.mp4")
    gpu = subprocess.run([str(root / "bin/pickgpu")], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}",
               V2D_GPU=gpu, V2D_FP_W="1014", V2D_FP_H="760")
    if CFG["sam3d"]:
        env["V2D_MESH_PICK"] = str(root / "t3/pick.py")   # multi-view SAM3D mesh choice
    if CFG["ref"] == "visible" and (out / "masks/1").is_dir():
        env["V2D_FP_REF"] = str(visible_frame(out))
        log(f"e{episode} {name}: FoundationPose starts at frame {env['V2D_FP_REF']}")
    cmd = [str(root / ".venv/bin/python"), "modules/v2d_pipelines/run_ego_reconstruction.py",
           "--video", str(data / f"videos/chunk-000/observation.images.ego_cam_a/episode_{episode:06d}.mp4"),
           "--object_prompt", PROMPTS[name],
           *([] if CFG["sam3d"] else ["--object_mesh", str(data / f"mesh/{name}/s20k/{name}.obj"),
                                      "--skip_object_scale_estimation"]),
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
        try:
            run_episode(root, episode)
        except Exception as e:  # one bad episode must not take the worker down with it
            log(f"FAIL e{episode}: {type(e).__name__}: {e}")


def run_episode(root: Path, episode: int):
    names = objects(root, episode)
    dirs = [root / CFG["out"] / f"e{episode}_{n}" for n in names]
    if CFG["src"]:
        for n, d in zip(names, dirs):
            copy_run(root / CFG["src"] / f"e{episode}_{n}", d)
    if CFG["calib"]:
        # boxes need the exact frame the pipeline detects on: our own undistorted video
        data = root / "data/track_3/public/videos/chunk-000/observation.images.ego_cam_a"
        calib_undistort(root, data / f"episode_{episode:06d}.mp4", dirs[0])
        pick_boxes(root, dirs[0] / "anycalib/undistorted.mp4", names, dirs)
        if CFG["stereo"]:
            stereo_depth(root, episode, dirs[0])
    first = None
    ok = True
    for name, d in zip(names, dirs):
        if first is not None:
            seed_from(first, d)
        done = run_one(root, episode, name)
        ok &= done
        if done and first is None:
            first = d
    if ok and not CFG["keep"]:
        cleanup(dirs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--out", default="out", help="output folder under root")
    p.add_argument("--calib", action="store_true", help="undistort with the real cam A calibration")
    p.add_argument("--stereo", action="store_true", help="rescale MoGe depth with the stereo pair")
    p.add_argument("--keep", action="store_true", help="keep frames/depth for fast reruns")
    p.add_argument("--sam3d", action="store_true", help="reconstruct meshes with SAM3D instead of scans")
    p.add_argument("--from", dest="src", help="tracking-only rerun: reuse this output folder's stages")
    p.add_argument("--ref", default="0", choices=["0", "visible"],
                   help="FoundationPose start frame: 0, or each object's least occluded frame")
    args = p.parse_args()
    CFG.update(out=args.out, calib=args.calib, stereo=args.stereo, keep=args.keep, src=args.src, ref=args.ref,
               sam3d=args.sam3d)
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
