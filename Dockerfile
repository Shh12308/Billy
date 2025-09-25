# ===============================
# Base image
# ===============================
FROM python:3.11-bullseye

# ===============================
# System dependencies for ML / audio / vision
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libjpeg-dev \
    zlib1g-dev \
    cmake \
    libffi-dev \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Upgrade pip
# ===============================
RUN python -m pip install --upgrade pip setuptools wheel

# ===============================
# Set working directory
# ===============================
WORKDIR /app

# ===============================
# Copy requirements
# ===============================
COPY requirements.txt .

# ===============================
# Install core dependencies first
# ===============================
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic requests httpx python-dotenv supabase openai transformers sentence-transformers chromadb redis prometheus-client duckduckgo-search

# ===============================
# Install PyTorch separately (pre-built wheel to avoid build issues)
# ===============================
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ===============================
# Install audio/voice/music/other ML libs
# ===============================
RUN pip install --no-cache-dir TTS audiocraft bitsandbytes accelerate huggingface_hub

# ===============================
# Install Whisper directly from GitHub
# ===============================
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git

# ===============================
# Copy app source
# ===============================
COPY . .

# ===============================
# Expose FastAPI default port
# ===============================
EXPOSE 8000

# ===============================
# Default command
# ===============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
