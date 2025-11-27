# app.py — Mixtral + DALL·E 3 + ElevenLabs TTS + Groq Whisper STT + OpenRouter moderation + Supabase
import os
import time
import uuid
import logging
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixtral-full-server")

# ---------------- Environment ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
DALL_E_MODEL = "dall-e-3"

# ---------------- FastAPI ----------------
app = FastAPI(title="Mixtral AI Full Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Supabase helpers ----------------
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY: return
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)
        r.raise_for_status()

async def get_history(user_id: str, limit: int = 64):
    if not SUPABASE_KEY: return []
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{SUPABASE_URL}/history", headers=SB_HEADERS,
                             params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit})
        r.raise_for_status()
        return [{"role": r["role"], "content": r["content"]} for r in r.json()]

# ---------------- Rate Limiter ----------------
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

def get_client_id(request: Optional[Request], x_api_key: Optional[str]):
    if x_api_key: return f"key:{x_api_key}"
    if request and request.client: return f"ip:{request.client.host}"
    return "anon"

def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

# ---------------- Groq Chat ----------------
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

async def _call_groq(messages: List[Dict], model: str = GROQ_MODEL):
    if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY not set")
    payload = {"model": model, "messages": messages}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def provider_chat(messages: List[Dict]):
    try:
        j = await _call_groq(messages)
        choices = j.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text") or "(empty)"
        return "(empty)"
    except Exception as e:
        logger.exception("Groq API error")
        return f"(Groq error: {str(e)})"

# ---------------- OpenRouter Moderation ----------------
async def moderate_text(text: str):
    if not OPENROUTER_API_KEY: return True, None
    url = "https://openrouter.ai/api/v1/moderations"
    payload = {"model": "openai-moderation-latest", "input": text}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        res = r.json()
        categories = res.get("results", [{}])[0].get("categories", {})
        flagged = any(categories.get(cat, False) for cat in categories)
        return not flagged, "Flagged" if flagged else None

# ---------------- Image Generation ----------------
async def generate_image(prompt: str):
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    payload = {"model": DALL_E_MODEL, "prompt": prompt, "size": "1024x1024"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["url"]

# ---------------- TTS (ElevenLabs) ----------------
async def generate_tts(text: str, voice: str = "Bella"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.7}}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.content  # binary audio

# ---------------- STT (Groq Whisper) ----------------
async def transcribe_audio(file: UploadFile):
    url = "https://api.groq.com/whisper/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": (file.filename, await file.read(), file.content_type)}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, files=files)
        r.raise_for_status()
        return r.json().get("text", "")

# ---------------- Routes ----------------
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None

@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...)):
    allowed, reason = await moderate_text(prompt)
    if not allowed: raise HTTPException(400, f"Blocked: {reason}")

    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": prompt})

    out = await provider_chat(messages)
    await save_message(user_id, "user", prompt)
    await save_message(user_id, "assistant", out)
    return {"response": out}

@app.post("/image")
async def image(prompt: str = Form(...)):
    url = await generate_image(prompt)
    return {"url": url}

@app.post("/tts")
async def tts(prompt: str = Form(...)):
    audio = await generate_tts(prompt)
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    text = await transcribe_audio(file)
    return {"text": text}

@app.get("/stream")
async def stream_chat(prompt: str):
    messages = [{"role":"user","content": prompt}]
    async def event_generator():
        out = await provider_chat(messages)
        yield {"data": out}
        yield {"data": "[DONE]"}
    return EventSourceResponse(event_generator())
