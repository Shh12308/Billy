# -------------------------------
# Base image
# -------------------------------
FROM python:3.11-slim

# -------------------------------
# System dependencies for ML / audio / vision
# -------------------------------
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

# -------------------------------
# Upgrade pip
# -------------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy requirements
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy app source
# -------------------------------
COPY . .

# -------------------------------
# Expose FastAPI default port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Default command
# -------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
