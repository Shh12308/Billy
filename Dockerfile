# ===============================
# Base image
# ===============================
FROM python:3.11-slim

# ===============================
# System dependencies for ML / audio / vision
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    wget \
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
# Install Python dependencies
# ===============================
# Remove audiocraft from requirements.txt if present
# and install it separately to avoid setup.py error
# ===============================
# Install Python dependencies
# ===============================
RUN sed -i '/audiocraft/d' requirements.txt \
 && pip install --no-cache-dir -r requirements.txt \
 # Force correct torch version for bitsandbytes
 && pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
 && pip install --no-cache-dir bitsandbytes==0.47.0 \
 && pip install --no-cache-dir audiocraft
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
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
