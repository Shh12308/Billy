FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for audio & image processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
