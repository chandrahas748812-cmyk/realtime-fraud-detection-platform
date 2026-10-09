# Real-time fraud detection platform - inference API image.
FROM python:3.11-slim

# Keep Python output unbuffered and avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY src/ ./src/

EXPOSE 8000

# Serve the FastAPI inference API with uvicorn.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
