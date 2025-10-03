from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import date
from supabase import create_client
from PIL import Image
import os
import io
import base64

app = FastAPI()

# --------------------
# Supabase Setup
# --------------------
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", 20))

# --------------------
# Usage tracking
# --------------------
def check_usage(user_id: str) -> bool:
    today = date.today()
    resp = supabase.table("usage").select("*").eq("user_id", user_id).execute()
    if resp.data:
        u = resp.data[0]
        if u["last_reset"] != str(today):
            supabase.table("usage").update({"count":1, "last_reset":str(today)}).eq("user_id", user_id).execute()
            return True
        if u["count"] >= DAILY_LIMIT:
            return False
        supabase.table("usage").update({"count": u["count"]+1}).eq("user_id", user_id).execute()
        return True
    else:
        supabase.table("usage").insert({"user_id": user_id, "count":1, "last_reset":str(today)}).execute()
        return True

# --------------------
# Models
# --------------------
class PromptRequest(BaseModel):
    user_id: str
    prompt: str

# --------------------
# Chat Endpoint
# --------------------
@app.post("/chat")
def chat(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    # Lightweight AI simulation
    return {"reply": f"🤖 [Free AI] I hear you said: '{req.prompt}'. Here's my thought: Sounds interesting!"}

# --------------------
# Image Captioning (Lightweight)
# --------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    width, height = img.size
    fmt = img.format if img.format else "JPEG"
    return {
        "caption": f"An image with size {width}x{height}, format {fmt}.",
        "info": {
            "width": width,
            "height": height,
            "format": fmt
        }
    }

# --------------------
# Image Filters
# --------------------
@app.post("/filter/grayscale")
async def grayscale(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": img_b64}

@app.post("/filter/thumbnail")
async def thumbnail(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img.thumbnail((128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"thumbnail_base64": img_b64}

# --------------------
# Health Check
# --------------------
@app.get("/")
def root():
    return {"message": "✅ Free AI running on Render (chat + image tools ready)!"}
