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

# --timeout 600: Gemma 4 31B (free tier) can take 2-4 min per resume; the
# default 120s would kill the worker mid-request.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "600"]
