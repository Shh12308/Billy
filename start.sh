#!/bin/bash
# ==========================================
# ZYNARA MEGA AI — RUNPOD AUTO INSTALL SCRIPT
# (Assumes: already inside /workspace/Zynara)
# ==========================================

set -e

echo "🚀 Zynara Mega AI Installer Starting..."

# ------------------------------------------
# System update
# ------------------------------------------
echo "🔄 Updating system packages..."
apt update && apt upgrade -y

# ------------------------------------------
# Base dependencies
# ------------------------------------------
echo "📦 Installing system dependencies..."
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    wget \
    curl \
    unzip \
    ca-certificates \
    build-essential

# ------------------------------------------
# Sanity check
# ------------------------------------------
if [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found"
    echo "➡️ Make sure you are inside the Zynara project directory"
    exit 1
fi

# ------------------------------------------
# Python virtual environment
# ------------------------------------------
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ------------------------------------------
# Upgrade pip
# ------------------------------------------
pip install --upgrade pip setuptools wheel

# ------------------------------------------
# Install Python dependencies
# ------------------------------------------
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# ------------------------------------------
# Create runtime directories
# ------------------------------------------
echo "📁 Creating runtime directories..."
mkdir -p /tmp/generated_images
mkdir -p /tmp/zynara
mkdir -p logs

# ------------------------------------------
# Environment variables (.env)
# ------------------------------------------
echo "🔑 Writing .env file..."

cat <<EOF > .env
APP_NAME=Zynara Mega AI
APP_AUTHOR=GoldYLocks
APP_DESCRIPTION=Multi-modal AI backend

PORT=7860

OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXX

SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your-supabase-key

# Optional services
REDIS_URL=redis://localhost:6379/0
ELEVEN_API_KEY=XXXXXXXXXXXXXXXXXXXX

DISABLE_MULTIMODAL=0
ALLOWED_ORIGINS=*

IMAGES_DIR=/tmp/generated_images
EOF

echo "✅ .env created"

# ------------------------------------------
# Export env vars for current shell
# ------------------------------------------
export $(grep -v '^#' .env | xargs)

# ------------------------------------------
# Optional: Redis (LOCAL ONLY)
# ------------------------------------------
echo "🧠 Installing Redis (optional)..."
apt install -y redis-server || true
systemctl enable redis || true
systemctl start redis || true

# ------------------------------------------
# Launch server
# ------------------------------------------
echo "🚀 Starting Zynara Mega AI backend..."

nohup venv/bin/uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1 \
    > logs/server.log 2>&1 &

sleep 3

# ------------------------------------------
# Show final info
# ------------------------------------------
PUBLIC_IP=$(curl -s ifconfig.me || echo "<RUNPOD_PUBLIC_IP>")

echo "=========================================="
echo "✅ ZYNARA MEGA AI IS LIVE"
echo "=========================================="
echo "🌍 Public API URL:"
echo "   http://${PUBLIC_IP}:${PORT}"
echo ""
echo "📘 Swagger Docs:"
echo "   http://${PUBLIC_IP}:${PORT}/docs"
echo ""
echo "📂 Logs:"
echo "   $(pwd)/logs/server.log"
echo "=========================================="
