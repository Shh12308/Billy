# app.py — Billy AI Full Stack + JWT Auth (Register/Login) + SSE streaming + Image + TTS + STT
import os
import time
import uuid
import json
import base64
import logging
import asyncio
import tempfile
from typing import Optional, List, Dict, Any

import httpx
import bcrypt
import jwt  # PyJWT
from fastapi import FastAPI, Request, Form, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-full-server")

# ---------- Env / Config ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"
GROQ_STT_ENDPOINT = f"{GROQ_BASE}/audio/transcriptions"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", 60 * 60 * 24 * 7))  # 7 days

RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

ADMIN_KEY = os.getenv("ADMIN_KEY", "admin123")

# ---------- FastAPI app ----------
app = FastAPI(title="Billy AI Full Stack (Auth + Media + Stream)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- In-memory fallbacks (used only if Supabase not configured) ----------
_in_memory_users: Dict[str, Dict[str, Any]] = {}  # username -> {id, username, password_hash}
_in_memory_history: Dict[str, List[Dict[str, str]]] = {}  # user_id -> list of messages


# ---------- Utilities: passwords, JWT ----------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False

def create_jwt_token(user_id: str, username: str) -> str:
    payload = {"sub": user_id, "username": username, "exp": int(time.time()) + JWT_EXPIRES_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------- User persistence (Supabase if available, else in-memory) ----------
async def create_user_supabase(username: str, password_hash: str) -> Dict[str, Any]:
    if not SUPABASE_KEY or not SUPABASE_URL:
        raise RuntimeError("Supabase not configured")
    payload = {"username": username, "password_hash": password_hash}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{SUPABASE_URL}/users", headers=SB_HEADERS, json=payload)
        r.raise_for_status()
        return r.json()

async def get_user_by_username_supabase(username: str) -> Optional[Dict[str, Any]]:
    if not SUPABASE_KEY or not SUPABASE_URL:
        raise RuntimeError("Supabase not configured")
    params = {"username": f"eq.{username}"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{SUPABASE_URL}/users", headers=SB_HEADERS, params=params)
        r.raise_for_status()
        arr = r.json()
        return arr[0] if arr else None

# wrappers that use supabase when available, else memory
async def create_user(username: str, password_hash: str) -> Dict[str, Any]:
    if SUPABASE_KEY and SUPABASE_URL:
        try:
            return await create_user_supabase(username, password_hash)
        except Exception:
            logger.exception("Supabase user create failed, falling back to memory")
    # fallback
    user_id = str(uuid.uuid4())
    _in_memory_users[username] = {"id": user_id, "username": username, "password_hash": password_hash}
    return _in_memory_users[username]

async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    if SUPABASE_KEY and SUPABASE_URL:
        try:
            return await get_user_by_username_supabase(username)
        except Exception:
            logger.exception("Supabase user lookup failed, falling back to memory")
    return _in_memory_users.get(username)


# ---------- History save / get (Supabase if available, else memory) ----------
async def save_message(user_id: str, role: str, content: str):
    if SUPABASE_KEY and SUPABASE_URL:
        payload = {"id": str(uuid.uuid4()), "user_id": user_id, "role": role, "content": content, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)
                r.raise_for_status()
            except Exception:
                logger.exception("Supabase save history failed — falling back to memory")
    # memory fallback
    _in_memory_history.setdefault(user_id, []).append({"role": role, "content": content})

async def get_history(user_id: str, limit: int = 64) -> List[Dict[str, str]]:
    if SUPABASE_KEY and SUPABASE_URL:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{SUPABASE_URL}/history", headers=SB_HEADERS,
                                     params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit})
                r.raise_for_status()
                return [{"role": h["role"], "content": h["content"]} for h in r.json()]
        except Exception:
            logger.exception("Supabase get_history failed — falling back to memory")
    return _in_memory_history.get(user_id, [])[-limit:]


# ---------- Rate limiter ----------
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


# ---------- Moderation ----------
async def openrouter_moderate(text: str):
    if not OPENROUTER_API_KEY:
        return {"error": "no_key"}
    try:
        url = "https://openrouter.ai/api/v1/moderations"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "openai-moderation-latest", "input": text}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    except Exception:
        logger.exception("OpenRouter moderation failed")
        return {"error": "moderation_failed"}

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


# ---------- Groq chat helpers ----------
async def call_groq_chat(messages: List[Dict], model: str = GROQ_MODEL):
    if not GROQ_API_KEY:
        return "(Groq key missing — fallback response.)"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        if r.status_code == 400 and "decommissioned" in r.text:
            payload["model"] = "llama-3.1-8b-instant"
            r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        choices = j.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text") or "(empty)"
        return "(empty response)"


# ---------- Groq streaming -> SSE (works on Railway) ----------
@app.get("/stream")
async def stream_chat(prompt: str, authorization: Optional[str] = Header(None)):
    """
    SSE streaming endpoint. Browser should call:
      new EventSource('/stream?prompt=Hello')
    This implementation streams token deltas from Groq and yields them as SSE 'data' events.
    """
    messages = [{"role": "user", "content": prompt}]

    async def event_generator():
        # If no Groq key, return a friendly fallback slowly
        if not GROQ_API_KEY:
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
                            # adapt to Groq streaming response structure
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                # SSE text must be 'data: <text>\n\n'
                                yield f"data: {delta}\n\n"
                        except Exception:
                            # ignore parse errors and continue
                            continue
            except Exception as e:
                logger.exception("Error during Groq streaming")
                # Notify client of stream error
                yield f"data: [STREAM_ERROR]\n\n"

    return EventSourceResponse(event_generator())


# ---------- Normal chat (protected) ----------
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Dependency: pass Authorization: Bearer <token>. Returns user dict or raises 401."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = authorization.split(" ", 1)[1]
    payload = decode_jwt_token(token)
    user = await get_user_by_username(payload.get("username"))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/chat")
async def chat(prompt: str = Form(...), user: Dict = Depends(get_current_user)):
    # rate limit per user
    client_id = f"user:{user['username']}"
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")

    history = await get_history(user_id=user["id"], limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history if h.get("role") and h.get("content")]
    messages.append({"role": "user", "content": prompt})
    out = await call_groq_chat(messages)
    # save asynchronously
    asyncio.create_task(save_message(user["id"], "user", prompt))
    asyncio.create_task(save_message(user["id"], "assistant", out))
    return {"response": out}


# ---------- Image generation (OpenAI DALL·E 3) ----------
@app.post("/image")
async def image_gen(prompt: str = Form(...), user: Dict = Depends(get_current_user)):
    rate_limited(f"user:{user['username']}")
    if not OPENAI_API_KEY:
        return {"url": f"https://via.placeholder.com/1024.png?text={prompt.replace(' ', '+')}"}
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "dall-e-3", "prompt": prompt, "size": "1024x1024", "n": 1}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        # data may have data[0].url or data[0].b64_json
        img = None
        try:
            img = data.get("data", [])[0].get("url")
        except Exception:
            pass
        if not img:
            b64 = data.get("data", [])[0].get("b64_json")
            if b64:
                raw = base64.b64decode(b64)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(raw)
                tmp.flush()
                tmp.close()
                return {"url": f"/tmp/{os.path.basename(tmp.name)}"}
        return {"url": img}


# ---------- TTS (ElevenLabs) ----------
@app.post("/tts")
async def tts_text(prompt: str = Form(...), voice: str = Form("alloy"), user: Dict = Depends(get_current_user)):
    rate_limited(f"user:{user['username']}")
    if not ELEVEN_API_KEY:
        # return silent fallback audio base64
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


# ---------- STT (Groq Whisper) ----------
@app.post("/stt")
async def stt(file: UploadFile = File(...), user: Dict = Depends(get_current_user)):
    rate_limited(f"user:{user['username']}")
    if not GROQ_API_KEY:
        return {"text": "Transcription fallback (no Groq key)"}
    files = {"file": (file.filename, await file.read(), file.content_type)}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(GROQ_STT_ENDPOINT, headers=headers, files=files)
            r.raise_for_status()
            data = r.json()
            text = data.get("text") or data.get("transcription") or ""
            return {"text": text}
        except Exception as e:
            logger.exception("Groq STT error")
            return {"text": f"STT error: {e}"}


# ---------- Auth: register / login ----------
class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/register")
async def register(req: RegisterRequest):
    # Basic validation
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="username and password required")
    existing = await get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="user already exists")
    hashed = hash_password(req.password)
    user = await create_user(req.username, hashed)
    token = create_jwt_token(user["id"], user["username"])
    return {"user": {"id": user["id"], "username": user["username"]}, "token": token}

@app.post("/login")
async def login(req: LoginRequest):
    user = await get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_jwt_token(user["id"], user["username"])
    return {"user": {"id": user["id"], "username": user["username"]}, "token": token}


# ---------- Utility endpoints ----------
@app.get("/history/{user_id}")
async def history(user_id: str, limit: int = 64, admin_key: Optional[str] = Header(None)):
    # Admin-protect if admin key configured
    if ADMIN_KEY and admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return {"history": await get_history(user_id, limit)}

@app.get("/metrics")
async def metrics(admin_key: Optional[str] = Header(None)):
    if ADMIN_KEY and admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return {"rate_limit_store_size": len(_rate_limit_store)}

@app.get("/")
async def root():
    return {"status": "ok", "features": ["auth","chat","stream","image","tts","stt"]}
