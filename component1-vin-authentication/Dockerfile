FROM python:3.11-slim

# opencv-python needs libGL / glib at runtime even though we never open a window
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in the ImageNet MobileNetV2 weights used by the "is this a VIN?" check,
# so the container doesn't download them on its first request.
RUN python -c "import tensorflow as tf; tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))"

COPY . .

EXPOSE 8000

# Single worker: the model is loaded once into memory at startup: extra workers
# would each load their own copy and multiply RAM usage for no benefit here.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
