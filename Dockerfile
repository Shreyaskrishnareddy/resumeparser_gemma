FROM python:3.11-slim

WORKDIR /app

# System deps for DOC parsing and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# One process + threads (gthread): avoids a multi-process SQLite WAL init race
# while keeping /health and the UI responsive while a 2-4 min parse runs in
# another thread. --timeout 600 because Gemma 4 (free tier) is slow per resume.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "600"]
