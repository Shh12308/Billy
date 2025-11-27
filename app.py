import os
import time
import uuid
import logging
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, Form, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-ai-server")

# ---------------- Environment / Keys ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"
GROQ_IMAGE_ENDPOINT = f"{GROQ_BASE}/images/generations"
GROQ_TTS_ENDPOINT = f"{GROQ_BASE}/audio/speech"
GROQ_STT_ENDPOINT = f"{GROQ_BASE}/audio/transcriptions"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

# ---------------- Rate Limiting ----------------
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

# ---------------- FastAPI ----------------
app = FastAPI(title="Billy AI Full Suite")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Helpers ----------------
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

async def moderate_text(text: str):
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based safety"

    if OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://openrouter.ai/api/v1/moderations",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                    json={"model":"openai-moderation-latest","input":text}
                )
                res = r.json()
                categories = res.get("results",[{}])[0].get("categories",{})
                flagged = any(categories.get(cat, False) for cat in categories)
                if flagged:
                    return False, "Blocked by OpenRouter moderation"
        except Exception:
            logger.exception("OpenRouter moderation failed — permissive fallback")
    return True, None

# ---------------- Supabase ----------------
async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY:
        return
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
    if not SUPABASE_KEY:
        return []
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit},
        )
        r.raise_for_status()
        return [{"role": r["role"], "content": r["content"]} for r in r.json()]

# ---------------- Groq AI ----------------
async def groq_chat(messages: List[Dict]):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_MODEL, "messages": messages}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        return j.get("choices",[{}])[0].get("message",{}).get("content","(empty)")

async def groq_image(prompt: str):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "size": "512x512"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GROQ_IMAGE_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("data",[{}])[0].get("url","")

async def groq_tts(text: str, voice: str="alloy"):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"voice": voice, "input": text, "format":"mp3"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GROQ_TTS_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        path = f"/tmp/{uuid.uuid4()}.mp3"
        with open(path,"wb") as f:
            f.write(r.content)
        return path

async def groq_stt(audio_file: UploadFile):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (audio_file.filename, await audio_file.read(), audio_file.content_type)}
        r = await client.post(GROQ_STT_ENDPOINT, headers=headers, files=files)
        r.raise_for_status()
        data = r.json()
        return data.get("text","")

# ---------------- Routes ----------------
@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None):
    client_id = get_client_id(request, None)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history if h.get("role") in ("user","assistant")]
    messages.append({"role":"user","content":prompt})
    out = await groq_chat(messages)

    try:
        await save_message(user_id, "user", prompt)
        await save_message(user_id, "assistant", out)
    except Exception:
        logger.exception("Failed to save history")

    return {"response": out}

@app.get("/stream")
async def stream_chat(prompt: str):
    messages = [{"role":"user","content":prompt}]
    async def event_gen():
        out = await groq_chat(messages)
        yield {"data": out}
        yield {"data": "[DONE]"}
    return EventSourceResponse(event_gen())

@app.post("/image")
async def image(prompt: str = Form(...)):
    url = await groq_image(prompt)
    return {"image_url": url}

@app.post("/tts")
async def tts(prompt: str = Form(...)):
    path = await groq_tts(prompt)
    return FileResponse(path, media_type="audio/mpeg", filename="output.mp3")

@app.post("/stt")
async def stt(audio_file: UploadFile = File(...)):
    text = await groq_stt(audio_file)
    return {"transcription": text}

@app.get("/history/{user_id}")
async def history(user_id: str):
    return {"history": await get_history(user_id)}

@app.get("/metrics")
async def metrics():
    return {"rate_limit_store_size": len(_rate_limit_store)}
