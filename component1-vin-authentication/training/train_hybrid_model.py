import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50, MobileNetV3Large
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_datasets():
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

    train_ds = train_ds.map(lambda x, y: (resnet_preprocess(x), y))
    val_ds = val_ds.map(lambda x, y: (resnet_preprocess(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_hybrid_model():
    # Input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Base models
    resnet = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    mobilenet = MobileNetV3Large(weights="imagenet", include_top=False, input_tensor=inputs)

    # Freeze both initially
    resnet.trainable = False
    mobilenet.trainable = False

    # Extract features
    resnet_features = layers.GlobalAveragePooling2D()(resnet.output)
    mobilenet_features = layers.GlobalAveragePooling2D()(mobilenet.output)

    # Combine features
    combined = layers.Concatenate()([resnet_features, mobilenet_features])

    # Classification head
    x = layers.Dense(256, activation="relu")(combined)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining Hybrid Model...")
    return model, resnet


def fine_tune(model, resnet, train_ds, val_ds):
    print("\nStarting fine-tuning...")

    resnet.trainable = True

    for layer in resnet.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )


def main():
    train_ds, val_ds = load_datasets()

    model, resnet = build_hybrid_model()

    # Initial training
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Fine-tuning
    fine_tune(model, resnet, train_ds, val_ds)

    # Save model
    save_path = os.path.join(MODEL_SAVE_DIR, "vin_tampering_hybrid.keras")
    model.save(save_path)

    print(f"\nHybrid model saved to {save_path}")


if __name__ == "__main__":
    main()