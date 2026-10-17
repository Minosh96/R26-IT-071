"""
evaluate_models.py
==================
Offline sanity/accuracy check for the three Body Condition models, using the
labelled Test split (Data Set/Test/<DamageType>/<BodyPart>/*.jpg).

  * View Validator  – body-part folder is mapped to the expected view
  * Part Classifier – 13 body panels
  * Damage Classifier – Dent / Rust / Scratch / Undamaged

It also runs a "wrong view" test: every image is validated against every
OTHER view slot and we count how often the validator correctly rejects it.

Usage (from component2-body-condition/):
    venv\\Scripts\\python evaluate_models.py [--per-class 40] [--data "<path to Test>"]
"""
import argparse
import os
import random
import sys
import pathlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
from inference.predict import load_models, DAMAGE_TYPES, BODY_PARTS, preprocess_image  # noqa: E402
from inference.view_validator import VIEW_LABELS, VIEW_NAME_TO_IDX, CONFIDENCE_THRESHOLD  # noqa: E402
from PIL import Image  # noqa: E402

PART_TO_VIEW = {
    "Bonnet": "front", "Front_Bumper": "front",
    "Dicky_Door": "rear", "Rear_Bumper": "rear",
    "Left_Front_Door": "left", "Left_Rear_Door": "left",
    "Left_Front_Fender": "left", "Left_Rear_Fender": "left",
    "Right_Front_Door": "right", "Right_Rear_Door": "right",
    "Right_Front_Fender": "right", "Right_Rear_Fender": "right",
    "Roof": "roof",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA = r"C:\Users\Minosh\Desktop\Y4S1\4th Year Research\2-Minosh\Data Set\Test"


def collect(data_root, per_class, seed=42):
    rng = random.Random(seed)
    samples = []
    for dmg_dir in sorted(pathlib.Path(data_root).iterdir()):
        if not dmg_dir.is_dir() or dmg_dir.name not in DAMAGE_TYPES:
            continue
        for part_dir in sorted(dmg_dir.iterdir()):
            if not part_dir.is_dir() or part_dir.name not in BODY_PARTS:
                continue
            files = [f for f in part_dir.iterdir() if f.suffix.lower() in EXTS]
            rng.shuffle(files)
            for f in files[:per_class]:
                samples.append((f, dmg_dir.name, part_dir.name))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--per-class", type=int, default=15,
                    help="max images per (damage, part) folder")
    args = ap.parse_args()

    samples = collect(args.data, args.per_class)
    print(f"\nEvaluating on {len(samples)} images from {args.data}\n")

    models = load_models()
    dm, pm, vm = models["damage_model"], models["part_model"], models["view_model"]
    if vm is None:
        print("[FAIL] View validator did not load!")

    view_ok = part_ok = dmg_ok = 0
    reject_total = reject_ok = 0
    view_conf = defaultdict(lambda: defaultdict(int))
    part_conf = defaultdict(lambda: defaultdict(int))
    dmg_conf = defaultdict(lambda: defaultdict(int))
    per_view_total = defaultdict(int)
    per_view_ok = defaultdict(int)

    batch = 32
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        arr = np.concatenate([
            preprocess_image(Image.open(f).convert("RGB")) for f, _, _ in chunk
        ])
        dp = dm.predict(arr, verbose=0)
        pp = pm.predict(arr, verbose=0)
        vp = vm.predict(arr, verbose=0) if vm is not None else None

        for j, (f, dmg_true, part_true) in enumerate(chunk):
            d_pred = DAMAGE_TYPES[int(np.argmax(dp[j]))]
            p_pred = BODY_PARTS[int(np.argmax(pp[j]))]
            dmg_conf[dmg_true][d_pred] += 1
            part_conf[part_true][p_pred] += 1
            dmg_ok += d_pred == dmg_true
            part_ok += p_pred == part_true

            if vp is not None:
                v_true = PART_TO_VIEW[part_true]
                v_idx = int(np.argmax(vp[j]))
                v_pred = VIEW_LABELS[v_idx]
                conf = float(vp[j][v_idx])
                view_conf[v_true][v_pred] += 1
                per_view_total[v_true] += 1
                if v_idx == VIEW_NAME_TO_IDX[v_true]:
                    view_ok += 1
                    per_view_ok[v_true] += 1
                # Wrong-slot test: would the API reject this image in other slots?
                for other in ("front", "rear", "left", "right", "roof"):
                    if other == v_true:
                        continue
                    reject_total += 1
                    wrong_and_confident = (v_idx != VIEW_NAME_TO_IDX[other]
                                           and conf >= CONFIDENCE_THRESHOLD)
                    reject_ok += wrong_and_confident
        print(f"  processed {min(i + batch, len(samples))}/{len(samples)}", end="\r")

    n = len(samples)
    print("\n\n================ RESULTS ================")
    if vm is not None:
        print(f"View Validator accuracy   : {view_ok / n * 100:6.2f}%  ({view_ok}/{n})")
        for v in ("front", "rear", "left", "right", "roof"):
            t = per_view_total[v]
            if t:
                print(f"    {v:6s}: {per_view_ok[v] / t * 100:6.2f}%  ({per_view_ok[v]}/{t})  "
                      f"preds={dict(view_conf[v])}")
        print(f"Wrong-view rejection rate : {reject_ok / max(reject_total, 1) * 100:6.2f}%  "
              f"({reject_ok}/{reject_total})")
    print(f"Body Part accuracy (13)   : {part_ok / n * 100:6.2f}%  ({part_ok}/{n})")
    for p in BODY_PARTS:
        row = part_conf.get(p)
        if row:
            t = sum(row.values())
            print(f"    {p:20s}: {row.get(p, 0) / t * 100:6.2f}%  ({row.get(p, 0)}/{t})")
    print(f"Damage Type accuracy (4)  : {dmg_ok / n * 100:6.2f}%  ({dmg_ok}/{n})")
    for d in DAMAGE_TYPES:
        row = dmg_conf.get(d)
        if row:
            t = sum(row.values())
            print(f"    {d:10s}: {row.get(d, 0) / t * 100:6.2f}%  preds={dict(row)}")
    print("=========================================")


if __name__ == "__main__":
    main()
