FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose FastAPI port (Paperspace defaults to 7860, but 8000 is fine if you set it in the UI)
EXPOSE 8000

# Run FastAPI (adjust "main:app" if your app file is named differently)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
