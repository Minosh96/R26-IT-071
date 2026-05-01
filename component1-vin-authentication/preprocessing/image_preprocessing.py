import cv2
import numpy as np

def preprocess_image(image):
    """
    Apply preprocessing steps like resizing, normalization, and noise reduction.
    """
    # Resize
    resized = cv2.resize(image, (224, 224))
    
    # Convert to grayscale or keep RGB based on model requirements
    # normalized = resized / 255.0
    
    return resized

if __name__ == "__main__":
    print("Image preprocessing module initialized.")
