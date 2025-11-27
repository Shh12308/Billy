# app.py — Billy AI Full Server: Chat + Stream + Image + TTS + STT + Moderation + History
import os
import time
import uuid
import logging
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, Form, Header, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-ai-server")

# ---------------- Environment ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

# ---------------- FastAPI App ----------------
app = FastAPI(title="Billy AI Full Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Rate limiter ----------------
def get_client_id(request: Optional[Request], x_api_key: Optional[str]):
    if x_api_key:
        return f"key:{x_api_key}"
    if request and request.client:
        return f"ip:{request.client.host}"
    return "anon"

def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

# ---------------- Moderation ----------------
async def moderate_text(text: str):
    if not text:
        return True, None
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based safety"
    if OPENROUTER_API_KEY:
        url = "https://openrouter.ai/api/v1/moderations"
        payload = {"model": "openai-moderation-latest", "input": text}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                r = await client.post(url, headers=headers, json=payload)
                data = r.json()
                categories = data.get("results", [{}])[0].get("categories", {})
                if any(categories.get(c, False) for c in categories):
                    return False, "Blocked by OpenRouter moderation"
            except Exception:
                logger.exception("Moderation error, fallback permissive")
    return True, None

# ---------------- History ----------------
async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY: return
    payload = {"id": str(uuid.uuid4()), "user_id": user_id, "role": role, "content": content, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)

async def get_history(user_id: str, limit: int = 64):
    if not SUPABASE_KEY: return []
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{SUPABASE_URL}/history", headers=SB_HEADERS,
                             params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit})
        return [{"role": h["role"], "content": h["content"]} for h in r.json()]

# ---------------- Groq Chat ----------------
async def provider_chat(messages: List[Dict], model: str = GROQ_MODEL):
    if not GROQ_API_KEY:
        return "(Groq API key not set)"
    cleaned_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]
    payload = {"model": model, "messages": cleaned_messages}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        if r.status_code == 400 and "decommissioned" in r.text:
            payload["model"] = "llama-3.1-8b-instant"
            r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text") or "(empty response)"
        return "(empty response)"

# ---------------- SSE Stream ----------------
@app.get("/stream")
async def stream_chat(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    async def event_generator():
        out = await provider_chat(messages)
        yield f"data: {out}\n\n"
        yield "data: [DONE]\n\n"
    return EventSourceResponse(event_generator())

# ---------------- Chat ----------------
@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)
    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")
    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history] + [{"role": "user", "content": prompt}]
    out = await provider_chat(messages)
    await save_message(user_id, "user", prompt)
    await save_message(user_id, "assistant", out)
    return {"response": out}

# ---------------- Image Generation ----------------
@app.post("/image")
async def generate_image(prompt: str = Form(...)):
    if not GROQ_API_KEY:
        return {"error": "No Groq key"}
    url = f"{GROQ_BASE}/images/generations"
    payload = {"prompt": prompt, "n": 1, "size": "1024x1024"}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return {"url": data.get("data", [{}])[0].get("url")}

# ---------------- TTS ----------------
@app.post("/tts")
async def text_to_speech(text: str = Form(...)):
    # Example using OpenRouter TTS endpoint
    if not OPENROUTER_API_KEY:
        return {"error": "No OpenRouter key"}
    url = "https://openrouter.ai/api/v1/audio/speech"
    payload = {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": text}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return {"audio_base64": r.json().get("audio")}

# ---------------- STT ----------------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not OPENROUTER_API_KEY:
        return {"error": "No OpenRouter key"}
    url = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    files = {"file": (file.filename, await file.read(), file.content_type)}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, files=files)
        r.raise_for_status()
        return {"text": r.json().get("text")}

# ---------------- Root ----------------
@app.get("/")
async def root():
    return {"status": "ok", "features": ["chat", "stream", "image", "tts", "stt"]}
