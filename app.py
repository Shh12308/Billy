# app.py — Mixtral via Groq + Hive moderation + Supabase REST
import os
import time
import uuid
import logging
import asyncio
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ---------------- Logging & Config ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixtral-groq-server")

# Provider / model config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
GROQ_BASE = os.getenv("GROQ_BASE", "https://api.groq.com/openai/v1")  # OpenAI-compatible base
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"  # POST

# Hive moderation
HIVE_API_KEY = os.getenv("HIVE_API_KEY")

# Supabase REST
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://orozxlbnurnchwodzfdt.supabase.co/rest/v1")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # optional for testing
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

# Admin / rate-limits
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin123")
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))

# ---------------- Rate-limiter ----------------
_rate_limit_store: Dict[str, List[float]] = {}
def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

def get_client_id(request: Optional[Request], x_api_key: Optional[str]):
    if x_api_key:
        return f"key:{x_api_key}"
    if request and request.client:
        return f"ip:{request.client.host}"
    return "anon"

def require_admin(x_admin_key: Optional[str]):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")

# ---------------- Supabase helpers (REST) ----------------
async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY:
        logger.debug("SUPABASE_KEY not set; skipping save_message.")
        return
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload, timeout=20.0)
        r.raise_for_status()

async def get_history(user_id: str, limit: int = 64):
    if not SUPABASE_KEY:
        return []
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit},
            timeout=20.0
        )
        r.raise_for_status()
        rows = r.json()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

async def get_total_messages():
    if not SUPABASE_KEY:
        return 0
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"select": "count", "count": "exact", "limit": 1},
            timeout=20.0
        )
        r.raise_for_status()
        cr = r.headers.get("content-range", "0/0")
        # content-range like "0-0/123"
        parts = cr.split("/")
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

# ---------------- Hive moderation ----------------
async def hive_moderate(text: str):
    if not HIVE_API_KEY:
        return {"error": "no_key"}
    headers = {"Authorization": f"Token {HIVE_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "models": ["text_moderation"]}
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.thehive.ai/api/v2/task/sync", headers=headers, json=payload, timeout=20.0)
        r.raise_for_status()
        return r.json()

async def moderate_text(text: str):
    # quick rule-based checks first
    if not text:
        return True, None
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    low = text.lower()
    if any(w in low for w in banned):
        return False, "Blocked by rule-based safety"
    # Hive.ai check if available
    if HIVE_API_KEY:
        try:
            res = await hive_moderate(text)
            status = res.get("status", [])
            if status and isinstance(status, list) and "response" in status[0]:
                classes = status[0]["response"].get("output_text", {}).get("classes", [])
                for c in classes:
                    if c.get("score", 0) > 0.65:
                        return False, f"Blocked by Hive: {c.get('class')} ({c.get('score'):.2f})"
        except Exception:
            logger.exception("Hive moderation error — permissive fallback")
            return True, None
    return True, None

# ---------------- Groq (Mixtral) provider ----------------
async def groq_chat_completion(messages: List[Dict], model: str = GROQ_MODEL, temperature: float = 0.2, max_tokens: int = 512):
    """
    Calls Groq's OpenAI-compatible chat completions endpoint.
    Endpoint: POST https://api.groq.com/openai/v1/chat/completions
    Auth: Authorization: Bearer <GROQ_API_KEY>
    See Groq docs (OpenAI-compatible).  [oai_citation:2‡console.groq.com](https://console.groq.com/docs/api-reference?utm_source=chatgpt.com)
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be set in env")

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def provider_chat(messages: List[Dict]):
    # messages = [{"role":"user","content":"..."}] or chat history
    try:
        j = await groq_chat_completion(messages)
        # OpenAI-compatible response format: choices[0].message.content
        choices = j.get("choices", [])
        if choices:
            # Some Groq responses put text under 'message' -> 'content' (chat format)
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if content is None:
                # fallback to 'text' or 'output_text'
                content = choices[0].get("text") or j.get("text") or ""
            return content
        # fallback: some Groq responses might use 'output_text'
        if "output_text" in j:
            return j["output_text"]
        return "(empty response)"
    except Exception as e:
        logger.exception("Groq API error")
        # Provide a friendly fallback so app doesn't crash
        return f"(Groq error: {str(e)})"

# ---------------- FastAPI app ----------------
app = FastAPI(title="Mixtral (Groq) Server + Hive moderation")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    parameters: Optional[dict] = None

@app.get("/")
async def root():
    return {"status": "ok", "provider": "groq", "model": GROQ_MODEL}

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(req.prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    messages = [{"role": "user", "content": req.prompt}]
    out = await provider_chat(messages)
    # save history best-effort
    try:
        await save_message(client_id, "user", req.prompt)
        await save_message(client_id, "assistant", out)
    except Exception:
        logger.exception("Failed to save history (non-fatal).")
    return {"text": out}

@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role":"user","content": prompt})
    out = await provider_chat(messages)
    try:
        await save_message(user_id, "user", prompt)
        await save_message(user_id, "assistant", out)
    except Exception:
        logger.exception("Failed to save history (non-fatal).")
    return {"response": out}

@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64):
    return {"history": await get_history(user_id, limit)}

@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    total = await get_total_messages()
    return {"total_messages": total, "rate_limit_store_size": len(_rate_limit_store)}

@app.post("/stream")
async def stream_chat(prompt: str = Form(...)):
    messages = [{"role":"user","content": prompt}]
    async def event_generator():
        out = await provider_chat(messages)
        yield {"data": out}
    return EventSourceResponse(event_generator())
