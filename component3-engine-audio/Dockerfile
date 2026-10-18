FROM python:3.11-slim

# librosa/soundfile need libsndfile + ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads

EXPOSE 5003

# Single worker: YAMNet + the SVM model are loaded once at startup and kept warm.
# Long timeout because model load + audio inference can take a while on a cold container.
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "--workers", "1", "--timeout", "120", "api.app:app"]
