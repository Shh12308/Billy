from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from datetime import date
from supabase import create_client, Client
from PIL import Image
import os
import io
import base64

app = FastAPI()

# --------------------
# Supabase Setup
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", 20))

# --------------------
# Usage tracking
# --------------------
def check_usage(user_id: str) -> bool:
    today = str(date.today())
    resp = supabase.table("usage").select("*").eq("user_id", user_id).execute()

    if resp.error:
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp.error}")

    if resp.data:
        u = resp.data[0]
        # Reset if it's a new day
        if u["last_reset"] != today:
            supabase.table("usage").update({
                "count": 1,
                "last_reset": today
            }).eq("user_id", user_id).execute()
            return True

        # Enforce daily limit
        if u["count"] >= DAILY_LIMIT:
            return False

        supabase.table("usage").update({
            "count": u["count"] + 1
        }).eq("user_id", user_id).execute()
        return True
    else:
        # First-time user entry
        supabase.table("usage").insert({
            "user_id": user_id,
            "count": 1,
            "last_reset": today
        }).execute()
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
async def chat(req: PromptRequest):
    if not check_usage(req.user_id):
        raise HTTPException(status_code=429, detail="Daily free limit reached.")
    # Lightweight AI simulation
    return {"reply": f"🤖 [Free AI] You said: '{req.prompt}'. Sounds interesting!"}

# --------------------
# Image Captioning (Lightweight)
# --------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        width, height = img.size
        fmt = img.format if img.format else "JPEG"
        return {
            "caption": f"An image with size {width}x{height}, format {fmt}.",
            "info": {"width": width, "height": height, "format": fmt}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

# --------------------
# Image Filters
# --------------------
@app.post("/filter/grayscale")
async def grayscale(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": img_b64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

@app.post("/filter/thumbnail")
async def thumbnail(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        img.thumbnail((128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"thumbnail_base64": img_b64}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# --------------------
# Health Check
# --------------------
@app.get("/")
async def root():
    return {"message": "✅ Free AI running on Render (chat + image tools ready)!"}
