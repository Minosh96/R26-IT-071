import os
import shutil
from ultralytics import YOLO

def train_model():
    """
    Train a YOLOv8 model for vehicle damage detection.
    Classes: scratch, dent, rust, repaint, misalignment.
    """
    
    # 1. Configuration
    DATA_YAML = "data.yaml"
    MODEL_VARIANT = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
    EPOCHS = 50
    IMG_SIZE = 640
    PROJECT_NAME = "vehicle_damage_detection"
    RUN_NAME = "damage_model_v1"

    # 2. Safety Checks
    if not os.path.exists(DATA_YAML):
        # Check if it's inside a subfolder (common with Roboflow downloads)
        possible_yaml = os.path.join(os.getcwd(), "Vehicle-Dent_Rust_Scratch-2", "data.yaml") # common name
        if os.path.exists(possible_yaml):
            DATA_YAML = possible_yaml
        else:
            print(f"\n[!] ERROR: '{DATA_YAML}' not found.")
            print("Please ensure your dataset is downloaded and the YAML file is in the root directory.")
            print("Run 'python download_dataset.py' first if you haven't.")
            return

    # 3. Initialize YOLOv8 model
    print(f"\n[INFO] Initializing model: {MODEL_VARIANT}")
    model = YOLO(MODEL_VARIANT) 

    # 4. Start Training
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=DATA_YAML, 
        epochs=EPOCHS, 
        imgsz=IMG_SIZE, 
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True
    )

    print("\n[SUCCESS] Training completed.")

    # Print Accuracy/Metrics
    print("\n=============================================")
    print("FINAL TRAINING METRICS (ACCURACY)")
    print("=============================================")
    try:
        # YOLOv8 metrics object
        if hasattr(results, 'box'):
            map50 = results.box.map50 * 100
            map50_95 = results.box.map * 100
            print(f"Model Accuracy (mAP@0.50)      : {map50:.2f}%")
            print(f"Model Accuracy (mAP@0.50-0.95) : {map50_95:.2f}%")
        elif hasattr(results, 'results_dict'):
            # Fallback for some ultralytics versions
            map50 = results.results_dict.get('metrics/mAP50(B)', 0) * 100
            map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0) * 100
            print(f"Model Accuracy (mAP@0.50)      : {map50:.2f}%")
            print(f"Model Accuracy (mAP@0.50-0.95) : {map50_95:.2f}%")
        else:
            print(f"Results metrics: {results}")
    except Exception as e:
        print(f"Could not parse accuracy metrics: {e}")
    print("=============================================\n")


    # 5. Extract and save the best model
    # YOLOv8 saves at: project/name/weights/best.pt
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, "damage_model.pt")
        print(f"[INFO] Best model exported to: {os.path.abspath('damage_model.pt')}")
    else:
        print(f"[WARNING] Could not find best model weights at {best_model_path}")
        print("Check the 'runs' or 'vehicle_damage_detection' folder.")

if __name__ == "__main__":
    train_model()
