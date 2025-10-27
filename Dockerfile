# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bullseye AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime user
RUN useradd -m appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Create required dirs and set permissions
RUN mkdir -p /var/lib/chatfleet/faiss /var/lib/chatfleet/uploads && \
    chown -R appuser:appuser /var/lib/chatfleet

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
