FROM python:3.12-slim-bullseye AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
ARG BUILD_VERSION=dev
ARG BUILD_COMMIT=local
ENV CHATFLEET_BUILD_VERSION=${BUILD_VERSION} \
    CHATFLEET_BUILD_COMMIT=${BUILD_COMMIT}

# Runtime user
RUN useradd -m appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints-linux-cpu.txt ./
COPY scripts/install_python_deps.sh ./scripts/install_python_deps.sh
RUN chmod +x ./scripts/install_python_deps.sh && \
    PIP_BIN=pip PYTHON_BIN=python ./scripts/install_python_deps.sh
COPY . .

# Create required dirs and set permissions
RUN mkdir -p /var/lib/chatfleet/faiss /var/lib/chatfleet/uploads && \
    chown -R appuser:appuser /var/lib/chatfleet

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
