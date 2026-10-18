import requests
import sys

BASE = "http://127.0.0.1:8080/api/v1/validate-view"

def test_api():
    # Create simple dummy image using PIL
    from PIL import Image
    import io
    b = io.BytesIO()
    Image.new("RGB", (224, 224), (100, 100, 100)).save(b, "JPEG")
    b.seek(0)
    image_bytes = b.read()

    for view in ["front", "rear", "left", "right", "roof"]:
        url = f"{BASE}?expected_view={view}"
        res = requests.post(url, files={"image": ("test.jpg", image_bytes, "image/jpeg")})
        print(f"[{view}] URL: {url}")
        print(f"[{view}] Status: {res.status_code}")
        print(f"[{view}] Body: {res.json()}")
        print("-" * 50)

if __name__ == "__main__":
    test_api()
