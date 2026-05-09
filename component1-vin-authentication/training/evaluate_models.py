import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TEST_DIR = "data/test"

models_to_evaluate = [
    ("MobileNetV2", "models/vin_tampering_mobilenetv2.keras", mobilenet_preprocess),
    ("EfficientNetB0", "models/vin_tampering_efficientnetb0.keras", efficientnet_preprocess),
    ("ResNet50", "models/vin_tampering_resnet50.keras", resnet_preprocess),
    ("MobileNetV3Large", "models/vin_tampering_mobilenetv3large.keras", mobilenetv3_preprocess),
    ("ResNet50 Fine-Tuned", "models/vin_tampering_resnet50_finetuned.keras", resnet_preprocess),
    ("Hybrid Model", "models/vin_tampering_hybrid.keras", resnet_preprocess),
]


def load_test_dataset(preprocess_func):
    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    print("Class names:", test_ds.class_names)

    test_ds = test_ds.map(lambda x, y: (preprocess_func(x), y))
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return test_ds


for model_name, model_path, preprocess_func in models_to_evaluate:
    if not os.path.exists(model_path):
        print(f"\n{model_name} model not found: {model_path}")
        continue

    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(model_path)

    test_ds = load_test_dataset(preprocess_func)

    loss, accuracy = model.evaluate(test_ds)

    print(f"{model_name} Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Test Loss: {loss:.4f}")