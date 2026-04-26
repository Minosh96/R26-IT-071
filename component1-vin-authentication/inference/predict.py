def predict_vin(image_bytes):
    import cv2
    import numpy as np
    import random

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return "Invalid Image"

    results = ["Original", "Altered", "Need Review"]
    prediction = random.choice(results)

    confidence = round(random.uniform(0.75, 0.95), 2)

    return {
        "label": prediction,
        "confidence": confidence
    }