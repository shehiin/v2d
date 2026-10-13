"""Inside the Grounding DINO container: candidate boxes on one frame for every object.

    python boxes.py --video undistorted.mp4 --weights dir --out cands.json --names blue_cup water_pitcher

Writes {name: [{"box": [x0, y0, x1, y1], "score": s, "prompt": p}, ...]}, best first.
Several prompts per object: the one in the pipeline's prompt can rank the wrong object
first (e.g. "a water pitcher" picks the blue cup), so selection happens outside.
"""

from __future__ import annotations

import argparse
import json
import os

PROMPTS = {
    "beige_cup": ["a beige cup"],
    "blue_cup": ["a blue cup"],
    "mini_dust_pan": ["a small dustpan"],
    "mini_sweeper": ["a small hand brush"],
    "plastic_dish_rack": ["a dish rack"],
    "water_pitcher": ["a tall white jug", "a white plastic watering can", "a water pitcher"],
    "white_pot": ["a white saucepan with a long handle", "a white cooking pot with a handle",
                  "a white pot"],
    "white_pot_lid": ["a round white lid", "a pot lid"],
    "wooden_piece_1": ["a wooden plank", "a flat wooden board"],
    "wooden_piece_2": ["a wooden plank", "a flat wooden board"],
    "wooden_spoon": ["a wooden spoon"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--frame", type=int, default=0)
    a = p.parse_args()

    import cv2
    import groundingdino
    import groundingdino.datasets.transforms as T
    import torch
    from groundingdino.util.inference import load_model, predict
    from PIL import Image
    from torchvision.ops import box_convert

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"no frame {a.frame} in {a.video}")
    h, w = img.shape[:2]
    tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x, _ = tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), None)
    model = load_model(os.path.join(groundingdino.__path__[0], "config", "GroundingDINO_SwinT_OGC.py"),
                       os.path.join(a.weights, "groundingdino_swint_ogc.pth"))
    res = {}
    for name in a.names:
        cands = []
        for prompt in PROMPTS.get(name, [name.replace("_", " ")]):
            b, s, _ = predict(model=model, image=x, caption=prompt,
                              box_threshold=0.15, text_threshold=0.15)
            b = box_convert(b, "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
            cands += [{"box": [round(v, 1) for v in bb], "score": round(sc, 4), "prompt": prompt}
                      for bb, sc in zip(b.tolist(), s.tolist())]
        res[name] = sorted(cands, key=lambda c: -c["score"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
