"""
test_integration.py
===================
End-to-end test of the Body Condition component THROUGH THE API GATEWAY,
using the same routes the Flutter app calls:

  GET  {GW}/body/health
  POST {GW}/body/api/v1/validate-view?expected_view=<view>   (field: image)
  POST {GW}/body/analyze                                     (front/rear/left/right/roof)

Uses real labelled images from Data Set/Test/<Damage>/<Part>/.

Usage (from component2-body-condition/):
    venv\\Scripts\\python test_integration.py [--gateway http://127.0.0.1:9000]
"""
import argparse
import pathlib
import random
import sys

import requests

DATA = pathlib.Path(r"C:\Users\Minosh\Desktop\Y4S1\4th Year Research\2-Minosh\Data Set\Test")

# One sample per view slot: (damage folder, part folder)
VIEW_SAMPLES = {
    "front": ("Dent", "Bonnet"),
    "rear":  ("Scratch", "Rear_Bumper"),
    "left":  ("Rust", "Left_Front_Door"),
    "right": ("Undamaged", "Right_Rear_Door"),
    "roof":  ("Dent", "Roof"),
}

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


def pick(damage, part, seed=7):
    folder = DATA / damage / part
    files = []
    if folder.exists():
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        fallback = DATA / "Undamaged" / part
        if fallback.exists():
            files = sorted(p for p in fallback.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"No test images found for part '{part}' in {DATA}")
    return random.Random(seed).choice(files)


def validate(gw, path, view):
    with open(path, "rb") as f:
        r = requests.post(f"{gw}/body/api/v1/validate-view",
                          params={"expected_view": view},
                          files={"image": (path.name, f, "image/jpeg")}, timeout=60)
    return r


def analyze(gw, mapping):
    files = {k: (p.name, open(p, "rb"), "image/jpeg") for k, p in mapping.items()}
    try:
        return requests.post(f"{gw}/body/analyze", files=files, timeout=300)
    finally:
        for _, fh, _ in files.values():
            fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://127.0.0.1:9000")
    gw = ap.parse_args().gateway.rstrip("/")

    print(f"\n=== 1. Health via gateway ({gw}/body/health)")
    r = requests.get(f"{gw}/body/health", timeout=10)
    h = r.json()
    print("   ", h)
    check("HTTP 200", r.status_code == 200)
    check("status == ok (all 3 models loaded)", h.get("status") == "ok")

    samples = {v: pick(*VIEW_SAMPLES[v]) for v in VIEW_SAMPLES}

    print("\n=== 2. View validator – correct slot")
    for view, path in samples.items():
        res = validate(gw, path, view).json()
        print(f"    {view:5s} <- {path.parent.name:16s} predicted={res.get('predicted')} "
              f"conf={res.get('confidence')}%")
        check(f"{view} accepted", res.get("correct") is True, res.get("message", ""))

    print("\n=== 3. View validator – WRONG slot must show warning")
    wrong_cases = [("front", "rear"), ("rear", "front"), ("left", "right"),
                   ("right", "left"), ("roof", "front"), ("front", "roof")]
    for img_view, slot in wrong_cases:
        res = validate(gw, samples[img_view], slot).json()
        msg = res.get("message", "")
        print(f"    {img_view} image -> {slot} slot: valid={res.get('valid')} | {msg}")
        check(f"{img_view}-image rejected in {slot} slot",
              res.get("valid") is False and "Incorrect view" in msg)

    print("\n=== 4. Invalid input handling")
    r = validate(gw, samples["front"], "sideways")
    check("unknown expected_view -> 400", r.status_code == 400, r.text)
    r = requests.post(f"{gw}/body/api/v1/validate-view", params={"expected_view": "front"},
                      files={"image": ("x.jpg", b"not an image", "image/jpeg")}, timeout=30)
    check("corrupt image -> 4xx/5xx with message (no crash)",
          r.status_code >= 400 and "detail" in r.json(), r.text)
    
    # Non-car image rejection check
    import io
    from PIL import Image
    blank_io = io.BytesIO()
    Image.new("RGB", (224, 224), (200, 200, 200)).save(blank_io, format="JPEG")
    r_nc = requests.post(f"{gw}/body/api/v1/validate-view", params={"expected_view": "front"},
                         files={"image": ("blank.jpg", blank_io.getvalue(), "image/jpeg")}, timeout=30)
    check("non-car image rejected with clear car-related message",
          r_nc.status_code == 200 and r_nc.json().get("valid") is False and "car-related image" in r_nc.json().get("message", ""),
          r_nc.text)

    print("\n=== 5. Full /analyze – all views correct")
    r = analyze(gw, samples)
    check("HTTP 200", r.status_code == 200, r.text[:300])
    res = r.json()
    print(f"    score={res.get('final_body_condition_score')}  status={res.get('vehicle_status')}")
    for d in res.get("damaged_parts", []):
        print(f"      - {d['part']:20s} {d['damage_type']:8s} {d['category']:8s} "
              f"({d['confidence']}%) [{d['view']}]")
    check("all_views_valid", res.get("all_views_valid") is True, res.get("view_errors"))
    parts = {d["part"] for d in res.get("damaged_parts", [])}
    types = {d["damage_type"] for d in res.get("damaged_parts", [])}
    check("Bonnet damage detected (front)", "Bonnet" in parts, parts)
    check("Roof damage detected (roof)", "Roof" in parts, parts)
    check("Damage types are Dent/Rust/Scratch only",
          types <= {"Dent", "Rust", "Scratch"}, types)
    check("right (undamaged) view has no damage",
          res["view_analysis"]["right"]["status"] == "undamaged",
          res["view_analysis"]["right"])

    print("\n=== 6. Full /analyze – front & rear swapped")
    swapped = dict(samples, front=samples["rear"], rear=samples["front"])
    res = analyze(gw, swapped).json()
    errs = res.get("view_errors", [])
    for e in errs:
        print(f"      ! {e['message']}")
    check("all_views_valid is False", res.get("all_views_valid") is False)
    check("front & rear flagged", {e["view"] for e in errs} == {"front", "rear"},
          [e["view"] for e in errs])

    print(f"\n=== RESULT: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
