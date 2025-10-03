from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import date
from supabase import create_client
import os
import uuid

# Optional multimodal
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = BlipForConditionalGeneration = None

app = FastAPI()
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
# Requests
# --------------------
class PromptRequest(BaseModel):
    user_id: str
    prompt: str

@app.post("/chat")
def chat(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    # Lightweight dummy AI, replace with your LLM / OpenAI
    return {"reply": f"🤖 Response to: {req.prompt}"}

# --------------------
# Image captioning
# --------------------
BLIP_MODEL = None
BLIP_PROC = None
def init_blip():
    global BLIP_MODEL, BLIP_PROC
    if BLIP_MODEL is None and BlipProcessor and BlipForConditionalGeneration:
        BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        BLIP_PROC = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
init_blip()

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if BLIP_MODEL is None:
        return {"caption": "BLIP not available."}
    from PIL import Image
    img = Image.open(file.file).convert("RGB")
    inputs = BLIP_PROC(images=img, return_tensors="pt")
    out = BLIP_MODEL.generate(**inputs)
    caption = BLIP_PROC.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}

# --------------------
# Simple health check
# --------------------
@app.get("/")
def root():
    return {"message": "✅ Billy Airunning"}
