# app.py - Multi-provider AI Server with Supabase

import os
import json
import time
import logging
import asyncio
import uuid
from typing import Optional, Dict, List, Any, AsyncGenerator

import httpx
from fastapi import (
    FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect,
    Form, UploadFile, File, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from databases import Database

load_dotenv()

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-ai-server")

# ---- Config ----
PROVIDER = os.getenv("PROVIDER", "mock").lower()  # 'openai', 'hf', or 'mock'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
PROVIDER_TIMEOUT = int(os.getenv("PROVIDER_TIMEOUT", "60"))

RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))

ADMIN_KEY = os.getenv("ADMIN_KEY")  # simple admin auth for /metrics and admin endpoints

# ---- Supabase / Database ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

DATABASE_URL = f"{SUPABASE_URL}?apikey={SUPABASE_KEY}"
db = Database(DATABASE_URL)

# ---- In-memory rate-limit store ----
_rate_limit_store: Dict[str, List[float]] = {}

# ---- FastAPI app ----
app = FastAPI(title="Multi-Provider AI Server (Supabase)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Startup / Shutdown ----
@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await db.disconnect()
    logger.info("Disconnected from Supabase.")

# ---- Helpers ----
def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

async def moderate_text(text: str) -> (bool, Optional[str]):
    if not text:
        return True, None
    banned = ["bomb", "kill", "terror", "child abuse"]
    low = text.lower()
    if any(word in low for word in banned):
        return False, "Blocked by heuristic"
    if OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/moderations"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "omni-moderation-latest", "input": text}
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                j = r.json()
                if j.get("results", [{}])[0].get("flagged", False):
                    return False, "Blocked by OpenAI moderation"
        except Exception:
            logger.exception("Moderation API error (permissive).")
            pass
    return True, None

def get_client_id(request: Optional[Request] = None, x_api_key: Optional[str] = None):
    if x_api_key:
        return f"key:{x_api_key}"
    if request and request.client:
        return f"ip:{request.client.host}"
    return "anon"

def require_admin(x_admin_key: Optional[str] = Header(None)):
    if not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin access not configured")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

# ---- Supabase chat history functions ----
async def save_message(user_id: str, role: str, content: str):
    query = """
    INSERT INTO history(id, user_id, created_at, role, content)
    VALUES (:id, :user_id, :created_at, :role, :content)
    """
    await db.execute(query, values={
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "role": role,
        "content": content
    })

async def get_history(user_id: str, limit: int = 64):
    query = """
    SELECT role, content FROM history
    WHERE user_id = :user_id
    ORDER BY created_at DESC
    LIMIT :limit
    """
    rows = await db.fetch_all(query, values={"user_id": user_id, "limit": limit})
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

# ---- Provider Layer ----
# [Same as original: openai_chat_completion, hf_chat_completion, mock_chat_completion, provider_chat, provider_embeddings]
# You can copy over these functions directly. They don't need database changes.

# ---- Routes ----

@app.get("/")
async def root():
    return {"status": "ok", "provider": PROVIDER, "default_model": DEFAULT_MODEL}

@app.get("/health")
async def health():
    configured = True
    if PROVIDER == "openai" and not OPENAI_API_KEY:
        configured = False
    if PROVIDER == "hf" and not HF_API_KEY:
        configured = False
    return {"ok": True, "provider": PROVIDER, "configured": configured, "default_model": DEFAULT_MODEL}

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    parameters: Optional[dict] = None

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)
    allowed, reason = await moderate_text(req.prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")
    model = req.model or DEFAULT_MODEL
    messages = [{"role":"user", "content": req.prompt}]
    out = await provider_chat(model, messages, stream=False, params=req.parameters)
    await save_message(client_id, "user", req.prompt)
    await save_message(client_id, "assistant", str(out))
    return {"source": PROVIDER, "model": model, "text": out}

@app.post("/chat")
async def chat(user_id: Optional[str] = Form("guest"), prompt: Optional[str] = Form(None), request: Request = None, x_api_key: Optional[str] = Header(None)):
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    client_id = get_client_id(request, x_api_key) if request else (f"key:{x_api_key}" if x_api_key else user_id)
    rate_limited(client_id)
    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")
    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": prompt})
    out = await provider_chat(DEFAULT_MODEL, messages, stream=False)
    await save_message(user_id, "user", prompt)
    await save_message(user_id, "assistant", str(out))
    return {"source": PROVIDER, "model": DEFAULT_MODEL, "response": out}

@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64, x_api_key: Optional[str] = Header(None)):
    return {"user_id": user_id, "history": await get_history(user_id, limit)}

@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Only text files (utf-8) supported")
    await save_message(user_id, "system", f"[uploaded:{file.filename}]\n{text}")
    return {"ok": True, "filename": file.filename, "size": len(content)}

@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    total_msgs = await db.fetch_val("SELECT COUNT(*) FROM history")
    return {"provider": PROVIDER, "total_history_messages": total_msgs, "rate_limit_store_size": len(_rate_limit_store)}

# SSE and WebSocket streaming routes can remain almost unchanged.
# Just make sure to `await save_message()` instead of SQLite commit.
