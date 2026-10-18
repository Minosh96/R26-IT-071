FROM python:3.11-slim

# opencv-python and ultralytics need libGL / glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# main.py's own __main__ block hardcodes port 8000 - don't use it here either.
# Single worker: both .h5 models are loaded once at startup and kept warm.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
