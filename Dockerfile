# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# ---------- System deps ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Workdir ----------
WORKDIR /app

# ---------- Install Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy app ----------
COPY . .

# ---------- Expose ----------
EXPOSE 8080

# ---------- Run ----------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
