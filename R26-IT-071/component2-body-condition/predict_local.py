import sys
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def predict(image_path):
    model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"Analyzing {image_path} using {model_path}...")
    
    model = YOLO(model_path)
    results = model.predict(image_path, save=True, project="outputs", name="predictions")
    
    print("\n--- Results ---")
    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"Detected: {label} ({conf:.2f})")
            
    print(f"\nSaved annotated image to: outputs/predictions/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_local.py path/to/your/image.jpg")
    else:
        predict(sys.argv[1])
