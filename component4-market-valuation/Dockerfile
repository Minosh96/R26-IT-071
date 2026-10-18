FROM python:3.11-slim

WORKDIR /app

# LightGBM needs the OpenMP runtime, which the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5004

CMD ["gunicorn", "--bind", "0.0.0.0:5004", "--workers", "2", "--timeout", "60", "api.app:app"]
