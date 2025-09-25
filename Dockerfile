# ==============================
# Stage 1: Base image with CUDA
# ==============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl ffmpeg \
    build-essential cmake pkg-config \
    libsndfile1 libgl1 libglib2.0-0 \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==============================
# Stage 2: Install Python deps
# ==============================

# Copy requirements (can be frozen with `pip freeze > requirements.txt`)
COPY requirements.txt .

# Install base deps
RUN pip install --no-cache-dir -r requirements.txt

# Install heavy optional deps (guarded by try/except in your code)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    transformers sentence-transformers \
    supabase duckduckgo-search redis \
    chromadb prometheus-client \
    faster-whisper TTS pillow \
    audiocraft openai

# ==============================
# Stage 3: Copy app
# ==============================
COPY . .

# Expose FastAPI port
EXPOSE 8000

# ==============================
# Stage 4: Run app
# ==============================
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]