FROM python:3.10-slim

WORKDIR /app

# System deps for audio/image
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy and install lightweight requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
