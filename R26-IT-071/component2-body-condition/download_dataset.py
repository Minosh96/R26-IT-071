import os
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_data():
    """
    Downloads the dataset from Roboflow using credentials from .env
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version = int(os.getenv("ROBOFLOW_VERSION", 1))

    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in .env")
        return

    print(f"Connecting to Roboflow: {workspace}/{project_name} (v{version})...")
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov8")
        
        print(f"Dataset downloaded successfully to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    download_data()
