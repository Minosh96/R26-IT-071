import os
import cv2
from ultralytics import YOLO
import argparse

def run_inference(model_path, source, save_results=True):
    """
    Run inference using the trained YOLOv8 model.
    """
    if not os.path.exists(model_path):
        print(f"[!] ERROR: Model file '{model_path}' not found. Run training first.")
        return

    # Load the model
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # Run prediction
    print(f"[INFO] Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=0.25, # Confidence threshold
        save=save_results,
        stream=False
    )

    # Process results
    for result in results:
        # Show specific detections
        count = len(result.boxes)
        print(f"[INFO] Detected {count} damage instances.")
        
        # You can access specific classes and coordinates here
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            print(f"  - {name}: {conf:.2f}")

    print(f"\n[SUCCESS] Inference finished. Results saved to 'runs/detect/predict'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Vehicle Damage Prediction")
    parser.add_argument("--model", type=str, default="damage_model.pt", help="Path to model file")
    parser.add_argument("--source", type=str, required=True, help="Path to image, folder, or video")
    
    args = parser.parse_args()
    
    run_inference(args.model, args.source)
