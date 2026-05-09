import os
from ultralytics import YOLO

def test_model():
    model_path = "damage_model.pt"
    data_yaml = "Vehicle-Dent_Rust_Scratch-2/data.yaml"

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    if not os.path.exists(data_yaml):
        print(f"Dataset config {data_yaml} not found.")
        return

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)

    print(f"Running validation on dataset: {data_yaml}")
    metrics = model.val(data=data_yaml)

    print("\n=============================================")
    print("FINAL TEST METRICS (ACCURACY)")
    print("=============================================")
    map50 = metrics.box.map50 * 100
    map50_95 = metrics.box.map * 100
    print(f"Model Accuracy (mAP@0.50)      : {map50:.2f}%")
    print(f"Model Accuracy (mAP@0.50-0.95) : {map50_95:.2f}%")
    print("=============================================\n")

if __name__ == "__main__":
    test_model()
