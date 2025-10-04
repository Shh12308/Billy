from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import date
from supabase import create_client
from PIL import Image
import os

# ---------- Supabase Setup ----------
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", 20))

# ---------- AI Imports ----------
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import whisper
from TTS.api import TTS

# Load lightweight models at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BLIP...")
BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
BLIP_PROC = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

print("Loading Whisper tiny...")
ASR_MODEL = whisper.load_model("tiny", device=DEVICE)

print("Loading TTS...")
TTS_MODEL = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=DEVICE=="cuda")

# Dummy LLM (replace with OpenAI API if desired)
CHAT_MODEL = pipeline("text-generation", model="distilgpt2", device=0 if DEVICE=="cuda" else -1)


# ---------- FastAPI App ----------
app = FastAPI()


# ---------- Usage Tracking ----------
def check_usage(user_id: str) -> bool:
    today = date.today()
    resp = supabase.table("usage").select("*").eq("user_id", user_id).execute()
    if resp.data:
        u = resp.data[0]
        if u["last_reset"] != str(today):
            supabase.table("usage").update({"count": 1, "last_reset": str(today)}).eq("user_id", user_id).execute()
            return True
        if u["count"] >= DAILY_LIMIT:
            return False
        supabase.table("usage").update({"count": u["count"] + 1}).eq("user_id", user_id).execute()
        return True
    else:
        supabase.table("usage").insert({"user_id": user_id, "count": 1, "last_reset": str(today)}).execute()
        return True


# ---------- Models ----------
class PromptRequest(BaseModel):
    user_id: str
    prompt: str


@app.get("/")
def root():
    return {"message": "✅ Free AI running on Render"}


@app.post("/chat")
def chat(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    output = CHAT_MODEL(req.prompt, max_length=80, num_return_sequences=1)
    return {"reply": output[0]["generated_text"]}


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
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    result = ASR_MODEL.transcribe(temp_path)
    return {"transcript": result["text"]}


@app.post("/tts")
def tts(req: PromptRequest):
    if not check_usage(req.user_id):
        return {"error": "Daily free limit reached."}
    out_path = f"/tmp/{req.user_id}.wav"
    TTS_MODEL.tts_to_file(text=req.prompt, file_path=out_path)
    return {"audio_file": out_path}
