import os
import io
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener

# Register HEIC opener to support all Apple HEIC/HEIF images
register_heif_opener()

# Import the vehicle view validator
from inference.view_validator import load_view_model, validate_view, VIEW_LABELS

DAMAGE_TYPES = ["Dent", "Rust", "Scratch", "Undamaged"]
BODY_PARTS = [
    'Bonnet', 'Dicky_Door', 'Front_Bumper', 'Left_Front_Door', 'Left_Front_Fender',
    'Left_Rear_Door', 'Left_Rear_Fender', 'Rear_Bumper', 'Right_Front_Door',
    'Right_Front_Fender', 'Right_Rear_Door', 'Right_Rear_Fender', 'Roof'
]

# View weights for calculating the final score
VIEW_WEIGHTS = {
    "front": 0.25,
    "rear": 0.25,
    "left": 0.20,
    "right": 0.20,
    "roof": 0.10,
}

# Valid body parts associated with each view to avoid misclassification
VALID_PARTS_BY_VIEW = {
    "front": ["Bonnet", "Front_Bumper", "Left_Front_Fender", "Right_Front_Fender"],
    "rear": ["Dicky_Door", "Rear_Bumper", "Left_Rear_Fender", "Right_Rear_Fender"],
    "left": ["Left_Front_Door", "Left_Front_Fender", "Left_Rear_Door", "Left_Rear_Fender"],
    "right": ["Right_Front_Door", "Right_Front_Fender", "Right_Rear_Door", "Right_Rear_Fender"],
    "roof": ["Roof"]
}

# Base penalty per damage type
BASE_PENALTY = {
    "Scratch": 10.0,
    "Dent": 20.0,
    "Rust": 25.0,
    "Undamaged": 0.0
}

# Multipliers for severity categories
SEVERITY_MULTIPLIERS = {
    "Small": 0.5,
    "moderate": 1.0,
    "severe": 1.5
}

# ─── Dataset directory used to locate the saved weight files ────────────────
_DATASET_DIR = r"C:\Users\Minosh\Desktop\Y4S1\4th Year Research\2-Minosh\Data Set"
_COMPONENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_npz(filename):
    """Locate a .npz weights file in standard search locations."""
    search_paths = [
        os.path.join(_COMPONENT_DIR, filename),
        os.path.join(_DATASET_DIR, filename),
        filename,
        os.path.join("component2-body-condition", filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"NPZ weights file '{filename}' not found in any of: {search_paths}"
    )


def _build_damage_model():
    """
    Builds the MobileNetV3Small damage classifier (4 classes).
    Architecture mirrors train_damage_capped.py exactly.
    Augmentation layers (no trainable weights) are included for
    structural completeness; they run with training=False at inference.
    """
    aug = tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1),
    ], name="aug")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = aug(inputs, training=False)
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, weights=None,
        include_preprocessing=True, pooling=None
    )
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    h = layers.Dense(96, activation='relu',
                     kernel_regularizer=regularizers.l2(0.002))(x)
    h = layers.Dropout(0.55)(h)
    out = layers.Dense(len(DAMAGE_TYPES), activation='softmax',
                       name='damage_output')(h)
    return Model(inputs=inputs, outputs=out)


def _build_part_model():
    """
    Builds the EfficientNetV2B0 body-part classifier (13 classes).
    Architecture mirrors train_part_only.py exactly.
    """
    aug = tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.15),
    ], name="aug")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = aug(inputs, training=False)
    base = tf.keras.applications.EfficientNetV2B0(
        input_shape=(224, 224, 3), include_top=False, weights=None,
        include_preprocessing=True, pooling=None
    )
    base.trainable = True
    for layer in base.layers[:-100]:
        layer.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    h = layers.Dense(768, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    h = layers.Dropout(0.5)(h)
    out = layers.Dense(len(BODY_PARTS), activation='softmax',
                       name='part_output')(h)
    return Model(inputs=inputs, outputs=out)


def load_models():
    """
    Builds all three specialist models:
      1. Damage Classifier  – MobileNetV3Small  (damage_capped_weights.npz)
      2. Part Classifier    – EfficientNetV2B0  (part_only_weights.npz)
      3. View Validator     – MobileNetV3Small  (vehicle_view_weights.weights.h5)

    Returns dict: {"damage_model": ..., "part_model": ..., "view_model": ...}
    """
    # ── Damage model (MobileNetV3Small) ────────────────────────────────
    dmg_npz = _find_npz("damage_capped_weights.npz")
    print(f"[INFO] Loading damage weights from:\n       {dmg_npz}")
    damage_model = _build_damage_model()
    # Warm-up to create all variable shapes before set_weights
    damage_model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    data = np.load(dmg_npz, allow_pickle=False)
    w_list = [data[k] for k in sorted(data.files, key=lambda k: int(k.replace('arr_', '')))]
    damage_model.set_weights(w_list)
    print(f"[INFO] Damage model loaded ({len(w_list)} weight arrays). "
          f"Test prediction: {damage_model(np.zeros((1,224,224,3), dtype=np.float32), training=False).numpy()}")

    # ── Part model (EfficientNetV2B0) ─────────────────────────────────
    part_npz = _find_npz("part_only_weights.npz")
    print(f"[INFO] Loading part weights from:\n       {part_npz}")
    part_model = _build_part_model()
    part_model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
    data2 = np.load(part_npz, allow_pickle=False)
    w_list2 = [data2[k] for k in sorted(data2.files, key=lambda k: int(k.replace('arr_', '')))]
    part_model.set_weights(w_list2)
    print(f"[INFO] Part model loaded ({len(w_list2)} weight arrays). "
          f"Test prediction: {part_model(np.zeros((1,224,224,3), dtype=np.float32), training=False).numpy()}")

    # ── View Validator (MobileNetV3Small, 5-class) ─────────────────────
    # Isolated: a view-model failure must NOT take down damage/part analysis.
    # If it can't load (e.g. incompatible weights artifact), view_model stays
    # None and per-image view validation is skipped downstream.
    print("[INFO] Loading vehicle view validator model...")
    try:
        view_model = load_view_model()
    except Exception as e:
        view_model = None
        print(f"[WARN] View validator failed to load; view validation disabled. "
              f"Damage/part analysis will continue. Reason: {e}")

    return {"damage_model": damage_model, "part_model": part_model, "view_model": view_model}



def preprocess_image_bytes(image_bytes, view_name):
    if not image_bytes:
        raise ValueError(f"The {view_name} image is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        # Resize to model input size (224, 224)
        image_resized = image.resize((224, 224), Image.BILINEAR)
        image_array = np.array(image_resized, dtype=np.float32)
        # Add batch dimension (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, image
    except Exception as e:
        raise ValueError(
            f"The {view_name} file is not a valid or supported image format "
            f"(JPEG, PNG, HEIC, etc.). Error: {str(e)}"
        )


def annotate_image(image, damage_type, body_part, confidence, category):
    try:
        draw = ImageDraw.Draw(image)
        text = f"Part: {body_part}\nDamage: {damage_type}\nConf: {confidence:.1f}%\nSeverity: {category}"
        if damage_type == "Undamaged":
            text = "Status: Undamaged"
            color = (0, 200, 0)
        else:
            color = (255, 0, 0)

        # Draw small background box for text readability
        draw.rectangle([(5, 5), (180, 75)], fill=(255, 255, 255, 128))
        draw.text((10, 10), text, fill=color)
    except Exception as e:
        print(f"[WARN] Could not annotate image: {e}")
    return image


def annotate_image_multiple(image, damages):
    try:
        draw = ImageDraw.Draw(image)
        if not damages:
            text = "Status: Undamaged"
            color = (0, 200, 0)
            draw.rectangle([(5, 5), (180, 45)], fill=(255, 255, 255, 128))
            draw.text((10, 10), text, fill=color)
        else:
            color = (255, 0, 0)
            y_offset = 5
            for dmg in damages:
                text = (
                    f"Part: {dmg['part']}\nDamage: {dmg['damage_type']}\n"
                    f"Conf: {dmg['confidence']:.1f}%\nSeverity: {dmg['category']}"
                )
                draw.rectangle([(5, y_offset), (180, y_offset + 70)],
                                fill=(255, 255, 255, 128))
                draw.text((10, y_offset + 5), text, fill=color)
                y_offset += 80
    except Exception as e:
        print(f"[WARN] Could not annotate image: {e}")
    return image


def preprocess_image(image):
    """Resizes and converts a PIL Image to a model-ready numpy array."""
    image_resized = image.resize((224, 224), Image.BILINEAR)
    image_array = np.array(image_resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def get_crop_regions(image, view_name):
    """
    Returns a list of tuples (cropped_image, region_label) representing
    the regions to run inference on.
    """
    width, height = image.size
    regions = []

    # Always include the full image as the baseline region
    regions.append((image, "full"))

    if view_name == "front":
        # Upper portion (65% of height) for Bonnet
        regions.append((image.crop((0, 0, width, int(height * 0.65))), "upper"))
        # Lower portion (65% of height) for Front Bumper
        regions.append((image.crop((0, int(height * 0.35), width, height)), "lower"))
    elif view_name == "rear":
        # Upper portion (65% of height) for Dicky Door
        regions.append((image.crop((0, 0, width, int(height * 0.65))), "upper"))
        # Lower portion (65% of height) for Rear Bumper
        regions.append((image.crop((0, int(height * 0.35), width, height)), "lower"))
    elif view_name == "left" or view_name == "right":
        # Left portion (60% of width)
        regions.append((image.crop((0, 0, int(width * 0.6), height)), "left_side"))
        # Right portion (60% of width)
        regions.append((image.crop((int(width * 0.4), 0, width, height)), "right_side"))

    return regions


def get_damage_category(confidence):
    """Map prediction confidence of damage type to category: Small, moderate, severe."""
    if confidence < 0.50:
        return "Small"
    elif confidence < 0.75:
        return "moderate"
    else:
        return "severe"


def condition_label(score):
    if score >= 90:
        return "Excellent condition"
    elif score >= 75:
        return "Good condition"
    elif score >= 60:
        return "Fair condition"
    elif score >= 40:
        return "Poor condition"
    else:
        return "Severely damaged condition"


def predict_body_condition(models, views, conf_threshold=0.25):
    """
    Analyzes multiple views of a vehicle using three specialist models:
      1. View Validator  – validates each image shows the correct vehicle view
      2. Damage Classifier – detects damage type (Dent / Rust / Scratch / Undamaged)
      3. Part Classifier   – identifies which body part is damaged (13 classes)

    View validation runs FIRST for every image.  If a wrong view is detected
    with high confidence the view is flagged with a clear error and skipped in
    the damage pipeline so the overall score is not contaminated.

    Parameters
    ----------
    models : dict
        Keys: ``"damage_model"``, ``"part_model"``, ``"view_model"``.
    views : dict
        Mapping of view name → raw image bytes.
        Expected keys: front, rear, left, right, roof (or up).

    Returns
    -------
    dict
        Full analysis result compatible with the FastAPI response schema and
        the HTML frontend (index.html).  Now includes:
          ``view_validation`` – per-view validation status
          ``view_errors``     – list of views that had wrong images
          ``all_views_valid`` – bool
    """
    damage_model = models["damage_model"]
    part_model   = models["part_model"]
    view_model   = models.get("view_model")   # may be None if loading failed

    session_id    = str(uuid.uuid4())
    view_analysis = {}
    damaged_parts = []
    view_validation = {}   # per-view validation results
    view_errors     = []   # list of views with wrong images

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    total_penalty = 0.0

    for view_name, contents in views.items():
        # Clean view name
        normalized_view = view_name.lower().strip()
        # Map alternate names like 'up' to 'roof'
        if normalized_view == "up":
            normalized_view = "roof"

        if normalized_view not in VIEW_WEIGHTS:
            continue

        # Load the original image from bytes
        try:
            original_image = Image.open(io.BytesIO(contents))
            original_image = original_image.convert("RGB")
        except Exception as e:
            raise ValueError(
                f"The {view_name} file is not a valid or supported image format "
                f"(JPEG, PNG, HEIC, etc.). Error: {str(e)}"
            )

        # ── STEP 1: Validate the vehicle view ──────────────────────────────
        validation_result = None
        view_is_valid = True

        if view_model is not None:
            try:
                validation_result = validate_view(view_model, contents, normalized_view)
                view_validation[normalized_view] = validation_result

                if not validation_result["correct"]:
                    view_is_valid = False
                    # Only block if model is confident it's the wrong view
                    if not validation_result["uncertain"]:
                        view_errors.append({
                            "view":     normalized_view,
                            "expected": validation_result["expected"],
                            "detected": validation_result["predicted"],
                            "confidence": validation_result["confidence"],
                            "message":  validation_result["message"],
                        })
                        print(
                            f"[VIEW VALIDATION] ❌ {normalized_view.upper()}: "
                            f"Expected '{validation_result['expected']}', "
                            f"detected '{validation_result['predicted']}' "
                            f"({validation_result['confidence']:.1f}%). Skipping damage analysis."
                        )
                        # Add a placeholder in view_analysis so frontend knows this view had an error
                        view_analysis[normalized_view] = {
                            "status": "invalid_view",
                            "view_validation": validation_result,
                            "error": validation_result["message"],
                            "damage_type": None,
                            "body_part": None,
                            "confidence": 0.0,
                            "part_confidence": 0.0,
                            "category": None,
                            "score": 0.0,
                            "penalty": 0.0,
                            "visual_url": None,
                            "damages": [],
                        }
                        continue  # Skip damage analysis for this view
                    else:
                        # Uncertain – log a warning but continue
                        print(
                            f"[VIEW VALIDATION] ⚠️  {normalized_view.upper()}: "
                            f"Uncertain – looks like '{validation_result['predicted']}' "
                            f"({validation_result['confidence']:.1f}%). Proceeding with caution."
                        )
                else:
                    print(
                        f"[VIEW VALIDATION] ✅ {normalized_view.upper()}: "
                        f"'{validation_result['predicted']}' confirmed "
                        f"({validation_result['confidence']:.1f}%)."
                    )
            except Exception as ve:
                print(f"[VIEW VALIDATION] Warning – could not validate '{normalized_view}': {ve}")
                view_is_valid = True   # fallback: allow the image through

        # Save uploaded copy
        upload_path = os.path.join("uploads", f"{session_id}_{normalized_view}.jpg")
        original_image.save(upload_path)

        # Get crops (including the full image itself)
        crops = get_crop_regions(original_image, normalized_view)

        view_damages = []
        seen_parts   = set()

        for crop_img, region_label in crops:
            # Preprocess the crop
            image_array = preprocess_image(crop_img)

            # ── Run damage inference ────────────────────────────────────────
            damage_preds = damage_model.predict(image_array, verbose=0)
            damage_probs = damage_preds[0]  # shape (4,)

            # ── Run part inference ──────────────────────────────────────────
            part_preds = part_model.predict(image_array, verbose=0)
            part_probs = part_preds[0]      # shape (13,)

            dmg_idx     = int(np.argmax(damage_probs))
            damage_type = DAMAGE_TYPES[dmg_idx]

            if damage_type != "Undamaged":
                # Ensure the predicted part is valid for this view
                valid_parts   = VALID_PARTS_BY_VIEW.get(normalized_view, [])
                valid_indices = [BODY_PARTS.index(p) for p in valid_parts if p in BODY_PARTS]

                if valid_indices:
                    best_idx  = valid_indices[0]
                    best_prob = part_probs[best_idx]
                    for idx in valid_indices:
                        if part_probs[idx] > best_prob:
                            best_prob = part_probs[idx]
                            best_idx  = idx
                    part_idx = best_idx
                else:
                    part_idx = int(np.argmax(part_probs))

                body_part   = BODY_PARTS[part_idx]
                damage_conf = float(damage_probs[dmg_idx]) * 100
                part_conf   = float(part_probs[part_idx]) * 100

                # De-duplicate: keep only one damage per body part (highest confidence)
                if body_part in seen_parts:
                    for idx, existing in enumerate(view_damages):
                        if existing["part"] == body_part:
                            if damage_conf > existing["confidence"]:
                                category = get_damage_category(damage_probs[dmg_idx])
                                penalty  = BASE_PENALTY[damage_type] * SEVERITY_MULTIPLIERS[category]
                                view_damages[idx] = {
                                    "part": body_part,
                                    "damage_type": damage_type,
                                    "category": category,
                                    "confidence": round(damage_conf, 2),
                                    "part_confidence": round(part_conf, 2),
                                    "penalty": penalty,
                                    "view": normalized_view,
                                }
                            break
                else:
                    category = get_damage_category(damage_probs[dmg_idx])
                    penalty  = BASE_PENALTY[damage_type] * SEVERITY_MULTIPLIERS[category]
                    view_damages.append({
                        "part": body_part,
                        "damage_type": damage_type,
                        "category": category,
                        "confidence": round(damage_conf, 2),
                        "part_confidence": round(part_conf, 2),
                        "penalty": penalty,
                        "view": normalized_view,
                    })
                    seen_parts.add(body_part)

        # Calculate view penalty and score
        view_penalty = sum(d["penalty"] for d in view_damages)
        view_score   = max(0.0, 100.0 - view_penalty)

        # Apply weighted penalty to final score
        total_penalty += view_penalty * VIEW_WEIGHTS[normalized_view]

        # Annotate and save output image
        annotated_img   = annotate_image_multiple(original_image.copy(), view_damages)
        output_filename = f"result_{session_id}_{normalized_view}.jpg"
        output_path     = os.path.join("outputs", output_filename)
        annotated_img.save(output_path)

        # ── Populate view_analysis entry ────────────────────────────────────
        if view_damages:
            primary_dmg  = max(view_damages, key=lambda x: x["confidence"])
            status       = "damaged"
            damage_type  = primary_dmg["damage_type"]
            body_part    = primary_dmg["part"]
            damage_conf  = primary_dmg["confidence"]
            part_conf    = primary_dmg["part_confidence"]
            category     = primary_dmg["category"]
        else:
            status = "undamaged"
            # Fallback: run full-image inference to get default confidences
            image_array_full = preprocess_image(original_image)

            damage_preds_full = damage_model.predict(image_array_full, verbose=0)
            part_preds_full   = part_model.predict(image_array_full, verbose=0)
            damage_probs_full = damage_preds_full[0]
            part_probs_full   = part_preds_full[0]

            dmg_idx_full  = int(np.argmax(damage_probs_full))
            part_idx_full = int(np.argmax(part_probs_full))
            damage_type   = None
            body_part     = None
            damage_conf   = float(damage_probs_full[dmg_idx_full]) * 100
            part_conf     = float(part_probs_full[part_idx_full]) * 100
            category      = None

        # Append view_damages to global damaged_parts list
        for dmg in view_damages:
            damaged_parts.append({
                "part": dmg["part"],
                "damage_type": dmg["damage_type"],
                "category": dmg["category"],
                "confidence": dmg["confidence"],
                "view": normalized_view,
            })

        view_analysis[normalized_view] = {
            "status": status,
            "damage_type": damage_type,
            "body_part": body_part,
            "confidence": round(damage_conf, 2),
            "part_confidence": round(part_conf, 2),
            "category": category,
            "score": round(view_score, 2),
            "penalty": round(view_penalty, 2),
            "visual_url": f"/outputs/{output_filename}",
            "damages": [
                {
                    "body_part": d["part"],
                    "damage_type": d["damage_type"],
                    "confidence": d["confidence"],
                    "category": d["category"],
                    "penalty": round(d["penalty"], 2),
                }
                for d in view_damages
            ],
        }

    # ── Final score ─────────────────────────────────────────────────────────
    final_score = max(0.0, min(100.0, 100.0 - total_penalty))
    final_score = round(final_score, 2)

    # Classify overall vehicle damage category
    if len(damaged_parts) == 0 and len(view_errors) == 0:
        vehicle_status  = "undamaged vehicle"
        damage_category = "none"
    elif len(view_errors) > 0 and len(damaged_parts) == 0:
        vehicle_status  = "view validation failed"
        damage_category = "unknown"
    else:
        if final_score >= 80.0:
            vehicle_status  = "damaged vehicle (Small)"
            damage_category = "minor"
        elif final_score >= 60.0:
            vehicle_status  = "damaged vehicle (moderate)"
            damage_category = "moderate"
        else:
            vehicle_status  = "damaged vehicle (severe)"
            damage_category = "severe"

    # ── Compatibility mappings for HTML frontend (index.html) ────────────────
    html_results = {}
    for v_name, v_data in view_analysis.items():
        # index.html expects 'up' instead of 'roof'
        compat_key = "up" if v_name == "roof" else v_name

        if v_data.get("status") == "invalid_view":
            html_results[compat_key] = {
                "view_score": 0.0,
                "undamaged_probability": 0.0,
                "damaged_probability": 0.0,
                "damage_type": None,
                "penalty": 0.0,
                "error": v_data.get("error"),
            }
        elif v_data["status"] == "undamaged":
            undamaged_prob = v_data["confidence"]
            damaged_prob   = 100.0 - undamaged_prob
            html_results[compat_key] = {
                "view_score": v_data["score"],
                "undamaged_probability": round(undamaged_prob, 2),
                "damaged_probability": round(damaged_prob, 2),
                "damage_type": v_data["damage_type"],
                "penalty": v_data["penalty"],
            }
        else:
            damaged_prob   = v_data["confidence"]
            undamaged_prob = 100.0 - damaged_prob
            html_results[compat_key] = {
                "view_score": v_data["score"],
                "undamaged_probability": round(undamaged_prob, 2),
                "damaged_probability": round(damaged_prob, 2),
                "damage_type": v_data["damage_type"],
                "penalty": v_data["penalty"],
            }

    # Determine if all views passed validation
    all_views_valid = len(view_errors) == 0

    summary_parts = [f"Score: {final_score}/100", f"Found {len(damaged_parts)} damage(s)."]
    if view_errors:
        failed_views = ", ".join(e["view"] for e in view_errors)
        summary_parts.insert(0, f"⚠️ Wrong image(s) detected for: {failed_views}.")
    summary = (
        f"Analysis complete. Vehicle status: {vehicle_status}. "
        + " ".join(summary_parts)
    )

    return {
        "session_id": session_id,
        "vehicle_status": vehicle_status,
        "damage_category": damage_category,
        "final_body_condition_score": final_score,
        "body_score": final_score,
        "condition_score": final_score,
        "score": final_score,
        "condition": condition_label(final_score),
        "damaged_parts": damaged_parts,
        # View validation results
        "all_views_valid": all_views_valid,
        "view_errors": view_errors,
        "view_validation": view_validation,
        # Main analysis data
        "view_analysis": view_analysis,
        "results": html_results,
        "summary": summary,
        "models_used": {
            "damage_classifier": "MobileNetV3Small (damage_capped_model.weights.h5)",
            "part_classifier":   "EfficientNetV2B0 (part_only_model.weights.h5)",
            "view_validator":    "MobileNetV3Small (vehicle_view_weights.weights.h5)",
        },
    }
