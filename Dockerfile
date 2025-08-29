# ===== Base image with CUDA (for GPU) =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ===== System dependencies =====
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3 (some libs need "python")
RUN ln -s /usr/bin/python3 /usr/bin/python

# ===== Python env =====
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# ===== Copy app =====
COPY . .

# ===== Expose port =====
EXPOSE 10000

# ===== Start FastAPI (Render needs 0.0.0.0) =====
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]