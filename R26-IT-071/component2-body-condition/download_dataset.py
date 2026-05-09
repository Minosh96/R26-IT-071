import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download_dataset():
    # Load credentials from .env file
    load_dotenv()
    
    API_KEY = os.getenv("ROBOFLOW_API_KEY")
    # Roboflow expects workspace and project names to be slugified
    raw_workspace = os.getenv("ROBOFLOW_WORKSPACE")
    WORKSPACE = raw_workspace.lower().replace(" ", "-") if raw_workspace else None
    
    raw_project = os.getenv("ROBOFLOW_PROJECT")
    PROJECT = raw_project.lower().replace(" ", "-") if raw_project else None
    
    VERSION = int(os.getenv("ROBOFLOW_VERSION", 1))

    if not API_KEY or not WORKSPACE or not PROJECT:
        print("\n[!] ERROR: Missing Roboflow credentials in .env file.")
        print("Please ensure ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and ROBOFLOW_PROJECT are set.\n")
        return

    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        
        print(f"Downloading dataset '{PROJECT}' version {VERSION}...")
        dataset = project.version(VERSION).download("yolov8")
        
        print("\n[SUCCESS] Dataset downloaded successfully!")
        print(f"Location: {dataset.location}")
        
        # Check if data.yaml exists in the downloaded folder and move/update it if necessary
        # Usually Roboflow puts it inside the downloaded folder.
        
    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")

if __name__ == "__main__":
    download_dataset()
