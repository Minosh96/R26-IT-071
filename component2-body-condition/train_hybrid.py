import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# Use tf.keras.applications directly to avoid import errors in some environments
MobileNetV3Small = tf.keras.applications.MobileNetV3Small
EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0
import yaml

# CONFIGURATION
DATA_YAML = "car-damage-demo-100-1/data.yaml"
BASE_DATA_DIR = "car-damage-demo-100-1"
CLASSIFICATION_DIR = "data_classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12 # Small number for quick restoration
CLASS_NAMES = ["dent", "rust", "scratch", "undamaged"]

def prepare_classification_dataset():
    """
    Converts YOLO format (images/labels) to Classification format (folders).
    """
    print("--- Preparing Classification Dataset ---")
    if os.path.exists(CLASSIFICATION_DIR):
        shutil.rmtree(CLASSIFICATION_DIR)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(CLASSIFICATION_DIR, split), exist_ok=True)
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(CLASSIFICATION_DIR, split, cls), exist_ok=True)

        img_dir = os.path.join(BASE_DATA_DIR, split, "images")
        lbl_dir = os.path.join(BASE_DATA_DIR, split, "labels")

        if not os.path.exists(img_dir): continue

        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
            # Find corresponding label
            lbl_file = os.path.splitext(img_file)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            target_class = "undamaged"
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                classes = [int(line.split()[0]) for line in lines if line.strip()]
                if 2 in classes:
                    target_class = "dent"
                elif 7 in classes:
                    target_class = "scratch"
                else:
                    # Part-only annotations fallback to prefix severity mappings
                    prefix = img_file.split('_')[0].lower()
                    if prefix in ['major', 'moderate']:
                        target_class = "dent"
                    elif prefix in ['minor', 'damage']:
                        target_class = "scratch"
                    else:
                        target_class = "undamaged"

            
            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(CLASSIFICATION_DIR, split, target_class, img_file)
            )
    print("Dataset prepared!")

def build_and_train(model_type, save_path):
    print(f"\n--- Training {model_type} ---")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(CLASSIFICATION_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(CLASSIFICATION_DIR, "valid"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Note: main.py expects models with built-in preprocessing or raw 0-255 inputs.
    # main.py comment: "Both models were trained with include_preprocessing=True"
    
    if model_type == "MobileNetV3":
        base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    else:
        base_model = EfficientNetV2B0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        # Add a simple preprocessing layer if needed, but main.py says it's already included
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(CLASS_NAMES), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model.save(save_path)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    prepare_classification_dataset()
    
    build_and_train("MobileNetV3", "vehicle_damage_type_mobilenetv3_best.h5")
    build_and_train("EfficientNetV2B0", "efficientnetv2b0_damage_type_best.h5")
    
    print("\n--- Regeneration Complete! ---")
    print("You can now run: uvicorn main:app --host 0.0.0.0 --port 8080")
