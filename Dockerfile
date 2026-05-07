# Reproducible runtime image for rlib.
#
# Build:    docker build -t rlib:3.0.0 .
# Run:      docker run --rm -it -v "$(pwd)":/workspace rlib:3.0.0 \
#               python examples/cartpole_a2c.py
#
# This image is CPU-only. For GPU training, base on a CUDA-enabled image
# such as `nvidia/cuda:12.1.0-runtime-ubuntu22.04` and install the matching
# `torch` wheel from <https://pytorch.org/get-started/locally/>.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages needed for Atari, OpenCV and rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first so we can cache them when only source changes.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project and install in editable mode with classic
# control + Atari extras pre-enabled.
COPY . .
RUN pip install -e ".[classic,atari]"

CMD ["python", "-c", "import rlib; print('rlib', rlib.__version__, 'ready')"]
