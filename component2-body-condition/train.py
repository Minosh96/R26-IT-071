import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def train_model():
    """
    Trains the YOLOv8 model for car damage detection.
    """
    # 1. Load a pre-trained model (starting with nano for speed)
    model = YOLO('yolov8n.pt') 
    
    # 2. Path to data.yaml
    # Assuming dataset is downloaded in the same folder
    # You might need to update this path after running download_dataset.py
    project_slug = os.getenv("ROBOFLOW_PROJECT")
    version = os.getenv("ROBOFLOW_VERSION")
    data_path = f"{project_slug}-{version}/data.yaml"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run download_dataset.py first.")
        return

    print(f"Starting training on {data_path}...")
    
    # 3. Start training
    results = model.train(
        data=data_path,
        epochs=50,       # Adjust based on need
        imgsz=640,       # Image size
        batch=16,        # Adjust based on RAM/VRAM
        name='car_damage_model',
        project='runs'
    )
    
    print("Training complete!")
    print(f"Results saved in: {results.save_dir}")

    # 4. Export the best model
    # Usually results.save_dir/weights/best.pt
    # For simplicity, we manually copy/rename it to damage_model.pt after training
    
if __name__ == "__main__":
    train_model()
