# ===== Base image (CPU only) =====
FROM python:3.11-slim

# ===== System dependencies =====
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===== Set working directory =====
WORKDIR /app

# ===== Copy requirements and install =====
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ===== Copy app code =====
COPY . .

# ===== Expose FastAPI port =====
EXPOSE 8000

# ===== Start FastAPI =====
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]