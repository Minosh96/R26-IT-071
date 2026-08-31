FROM python:3.11-slim

# opencv-python needs libGL / glib at runtime even though we never open a window
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Single worker: the model is loaded once into memory at startup: extra workers
# would each load their own copy and multiply RAM usage for no benefit here.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
