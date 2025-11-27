# app.py — Billy AI Full Stack (Chat stream + Image + TTS + STT + Moderation + History)
import os
import time
import uuid
import json
import base64
import logging
import asyncio
import tempfile
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, Form, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-full-server")

# ---------------- Environment ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"
GROQ_STT_ENDPOINT = f"{GROQ_BASE}/audio/transcriptions"   # assumed Groq STT path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")           # DALL·E
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")           # ElevenLabs TTS
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")   # moderation

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

# ---------------- Rate limiting ----------------
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

# ---------------- App ----------------
app = FastAPI(title="Billy AI Full Stack")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------- Helpers ----------------
def get_client_id(request: Optional[Request], x_api_key: Optional[str]):
    if x_api_key:
        return f"key:{x_api_key}"
    if request and getattr(request, "client", None):
        return f"ip:{request.client.host}"
    return "anon"

def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

async def openrouter_moderate(text: str):
    if not OPENROUTER_API_KEY:
        return {"error": "no_key"}
    try:
        url = "https://openrouter.ai/api/v1/moderations"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "openai-moderation-latest", "input": text}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.exception("OpenRouter moderation failure")
        return {"error": "moderation_failed", "exception": str(e)}

async def moderate_text(text: str):
    if not text:
        return True, None
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based safety"
    if OPENROUTER_API_KEY:
        res = await openrouter_moderate(text)
        results = res.get("results", [{}])
        categories = results[0].get("categories", {}) if results else {}
        flagged = any(categories.get(k, False) for k in categories)
        if flagged:
            return False, "Blocked by OpenRouter moderation"
    return True, None

# ---------------- Supabase helpers ----------------
async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY or not SUPABASE_URL:
        return
    payload = {"id": str(uuid.uuid4()), "user_id": user_id, "role": role, "content": content, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)
            r.raise_for_status()
        except Exception:
            logger.exception("Failed to save message to Supabase (non-fatal)")

async def get_history(user_id: str, limit: int = 64):
    if not SUPABASE_KEY or not SUPABASE_URL:
        return []
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{SUPABASE_URL}/history", headers=SB_HEADERS, params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit})
        r.raise_for_status()
        return [{"role": h["role"], "content": h["content"]} for h in r.json()]

# ---------------- Groq chat (non-stream) ----------------
async def call_groq_chat(messages: List[Dict], model: str = GROQ_MODEL):
    if not GROQ_API_KEY:
        return "(Groq API key not set — fallback response.)"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        if r.status_code == 400 and "decommissioned" in r.text:
            # fallback model
            logger.info("Model decommissioned — falling back")
            payload["model"] = "llama-3.1-8b-instant"
            r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        choices = j.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text") or "(empty)"
        return "(empty response)"

# ---------------- Groq streaming chat (SSE) ----------------
@app.get("/stream")
async def stream_chat(prompt: str):
    """
    Streamed chat endpoint (SSE). The client should connect with EventSource:
    new EventSource('/stream?prompt=hello')
    """
    messages = [{"role": "user", "content": prompt}]

    async def event_generator():
        if not GROQ_API_KEY:
            # send fallback slowly for UX
            fallback = "(Groq key missing — fallback response.)"
            for ch in fallback:
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
            return

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": GROQ_MODEL, "messages": messages, "stream": True}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", GROQ_CHAT_ENDPOINT, headers=headers, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        # Groq streaming typically sends lines like: data: {...} or data: [DONE]
                        if not line:
                            continue
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(raw)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                # stream token / characters
                                yield f"data: {delta}\n\n"
                        except Exception:
                            # If parsing failed, ignore and continue
                            continue
            except Exception as e:
                logger.exception("Error during Groq streaming")
                # send an error marker to client
                yield f"data: [STREAM_ERROR]\n\n"

    return EventSourceResponse(event_generator())

# ---------------- Normal chat (non-stream) ----------------
@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history if h.get("role") and h.get("content")]
    messages.append({"role": "user", "content": prompt})

    out = await call_groq_chat(messages)
    # Save asynchronously (non-blocking for response)
    asyncio.create_task(save_message(user_id, "user", prompt))
    asyncio.create_task(save_message(user_id, "assistant", out))
    return {"response": out}

# ---------------- Image generation (OpenAI DALL·E 3) ----------------
@app.post("/image")
async def image_gen(prompt: str = Form(...)):
    """
    Generates an image via OpenAI Images API (DALL·E). Returns JSON: { "url": "<image url>" }
    If OPENAI_API_KEY missing, returns a safe placeholder image URL.
    """
    if not OPENAI_API_KEY:
        # Placeholder fallback
        return {"url": f"https://via.placeholder.com/1024.png?text={prompt.replace(' ','+')}"}

    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "dall-e-3", "prompt": prompt, "size": "1024x1024"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        # OpenAI may return image URL or b64; handle common case
        # prefer data[0].url if present
        img = None
        try:
            img = data.get("data", [])[0].get("url")
        except Exception:
            pass
        if not img:
            # maybe base64 content
            b64 = data.get("data", [])[0].get("b64_json")
            if b64:
                # write temporary file and return file URL path
                raw = base64.b64decode(b64)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(raw)
                tmp.flush()
                tmp.close()
                return {"url": f"/tmp/{os.path.basename(tmp.name)}"}
        return {"url": img}

# ---------------- TTS (ElevenLabs) ----------------
@app.post("/tts")
async def tts_text(prompt: str = Form(...), voice: str = Form("alloy")):
    """
    Returns JSON: { "audio_base64": "..." }
    ElevenLabs endpoint returns binary audio. We'll return base64 for frontend convenience.
    """
    if not ELEVEN_API_KEY:
        # fallback: return silent audio base64 so frontend can handle
        silent = base64.b64encode(b"\0" * 20000).decode()
        return {"audio_base64": silent}

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": prompt, "voice_settings": {"stability": 0.6, "similarity_boost": 0.6}}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        audio_bytes = r.content
        return {"audio_base64": base64.b64encode(audio_bytes).decode()}

# ---------------- STT (Groq Whisper) ----------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Accepts multipart file upload (audio). Returns JSON: { "text": "<transcription>" }
    """
    if not GROQ_API_KEY:
        return {"text": "Transcription fallback: (no Groq key)"}

    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        try:
            r = await client.post(GROQ_STT_ENDPOINT, headers=headers, files=files)
            r.raise_for_status()
            data = r.json()
            # Groq response shape may vary; attempt common keys
            text = data.get("text") or data.get("transcription") or ""
            return {"text": text}
        except Exception as e:
            logger.exception("Groq STT error")
            return {"text": f"STT error: {e}"}

# ---------------- Utility endpoints ----------------
@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64):
    return {"history": await get_history(user_id, limit)}

@app.get("/metrics")
async def metrics():
    return {"rate_limit_store_size": len(_rate_limit_store)}

@app.get("/")
async def root():
    return {"status": "ok", "features": ["chat","stream","image","tts","stt"]}
