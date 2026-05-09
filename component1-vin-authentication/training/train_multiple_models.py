import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV3Large
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
FINE_TUNE_EPOCHS = 10

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_datasets(preprocess_func):
    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    print("Class names:", train_ds.class_names)

    train_ds = train_ds.map(lambda x, y: (preprocess_func(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_func(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_model(base_model_class, model_name):
    base_model = base_model_class(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nTraining {model_name}...")
    return model


def fine_tune_resnet50(model, train_ds, val_ds):
    print("\nStarting fine-tuning for ResNet50...")

    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS
    )

    return model


def train_and_save(model_name, base_model_class, preprocess_func, save_name, fine_tune=False):
    train_ds, val_ds = load_datasets(preprocess_func)

    model = build_model(base_model_class, model_name)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    if fine_tune:
        model = fine_tune_resnet50(model, train_ds, val_ds)

    save_path = os.path.join(MODEL_SAVE_DIR, save_name)
    model.save(save_path)

    print(f"{model_name} saved to {save_path}")


if __name__ == "__main__":
    train_and_save(
        model_name="EfficientNetB0",
        base_model_class=EfficientNetB0,
        preprocess_func=efficientnet_preprocess,
        save_name="vin_tampering_efficientnetb0.keras",
        fine_tune=False
    )

    train_and_save(
        model_name="ResNet50 Fine-Tuned",
        base_model_class=ResNet50,
        preprocess_func=resnet_preprocess,
        save_name="vin_tampering_resnet50_finetuned.keras",
        fine_tune=True
    )

    train_and_save(
        model_name="MobileNetV3Large",
        base_model_class=MobileNetV3Large,
        preprocess_func=mobilenetv3_preprocess,
        save_name="vin_tampering_mobilenetv3large.keras",
        fine_tune=False
    )