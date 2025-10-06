from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import date
from supabase import create_client
from PIL import Image
import os
from gtts import gTTS
import tempfile
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import whisper

# ---------- Supabase Setup ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", 20))

# ---------- AI Model Setup ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting on device: {DEVICE}")

print("🧠 Loading BLIP model...")
BLIP_PROC = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

print("🎧 Loading Whisper tiny model...")
ASR_MODEL = whisper.load_model("tiny", device=DEVICE)

print("💬 Loading text generation model...")
CHAT_MODEL = pipeline(
    "text-generation", model="distilgpt2", device=0 if DEVICE == "cuda" else -1
)

# ---------- FastAPI App ----------
app = FastAPI(title="Zynara AI", description="Free AI API running on Render 🚀")

# ---------- Helper: Usage Tracking ----------
def check_usage(user_id: str) -> bool:
    """Track daily usage per user in Supabase."""
    today = str(date.today())
    resp = supabase.table("usage").select("*").eq("user_id", user_id).execute()

    if resp.data:
        u = resp.data[0]
        if u["last_reset"] != today:
            supabase.table("usage").update(
                {"count": 1, "last_reset": today}
            ).eq("user_id", user_id).execute()
            return True
        if u["count"] >= DAILY_LIMIT:
            return False
        supabase.table("usage").update(
            {"count": u["count"] + 1}
        ).eq("user_id", user_id).execute()
        return True
    else:
        supabase.table("usage").insert(
            {"user_id": user_id, "count": 1, "last_reset": today}
        ).execute()
        return True


# ---------- Models ----------
class PromptRequest(BaseModel):
    user_id: str
    prompt: str


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "✅ Zynara AI API is live on Render!"}


@app.post("/chat")
def chat(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    output = CHAT_MODEL(req.prompt, max_length=80, num_return_sequences=1)
    return {"reply": output[0]["generated_text"].strip()}


@app.post("/caption")
async def caption(user_id: str, file: UploadFile = File(...)):
    if not check_usage(user_id):
        return {"error": "Daily free limit reached."}
    img = Image.open(file.file).convert("RGB")
    inputs = BLIP_PROC(images=img, return_tensors="pt").to(DEVICE)
    out = BLIP_MODEL.generate(**inputs)
    caption = BLIP_PROC.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}


@app.post("/asr")
async def asr(user_id: str, file: UploadFile = File(...)):
    if not check_usage(user_id):
        return {"error": "Daily free limit reached."}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = ASR_MODEL.transcribe(tmp_path)
    os.remove(tmp_path)
    return {"transcript": result["text"].strip()}


@app.post("/tts")
def tts(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    tts = gTTS(req.prompt, lang="en")
    out_path = f"/tmp/{req.user_id}.mp3"
    tts.save(out_path)
    return {"audio_file": out_path}
