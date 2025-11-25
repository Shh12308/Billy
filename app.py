# app.py — Multi-provider AI Server (Supabase REST + OpenAI/HF/Mock)
import os
import json
import time
import uuid
import logging
import httpx
import asyncio
from typing import Optional, List, Dict, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai  # pip install openai

load_dotenv()

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-ai-server")

# ---------------- Config ----------------
PROVIDER = os.getenv("PROVIDER", "mock").lower()  # 'openai', 'hf', 'mock'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
ADMIN_KEY = os.getenv("ADMIN_KEY")

# ---------------- Supabase REST ----------------
SUPABASE_URL = "https://orozxlbnurnchwodzfdt.supabase.co/rest/v1"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY must be set!")

SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# ---------------- Rate-limiting ----------------
_rate_limit_store: Dict[str, List[float]] = {}

def rate_limited(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

def get_client_id(request: Request, x_api_key: Optional[str]):
    if x_api_key:
        return f"key:{x_api_key}"
    return f"ip:{request.client.host}"

def require_admin(x_admin_key: Optional[str]):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")

# ---------------- Supabase REST helpers ----------------
async def save_message(user_id: str, role: str, content: str):
    payload = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)
        r.raise_for_status()

async def get_history(user_id: str, limit: int = 64):
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={
                "user_id": f"eq.{user_id}",
                "order": "created_at.asc",
                "limit": limit
            },
        )
        r.raise_for_status()
        rows = r.json()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

async def get_total_messages():
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"select": "count", "count": "exact", "limit": 1}
        )
        r.raise_for_status()
        return r.headers.get("content-range", "0").split("/")[1]

# ---------------- Moderation ----------------
async def moderate_text(text: str):
    banned = ["bomb", "kill", "terror"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by simple moderation"
    return True, None

# ---------------- Provider layer ----------------
async def provider_chat(model: str, messages: List[Dict], stream=False, params=None):
    if PROVIDER == "openai":
        openai.api_key = OPENAI_API_KEY
        if not stream:
            resp = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                **(params or {})
            )
            return resp.choices[0].message.content
        else:
            async def stream_gen() -> AsyncGenerator[str, None]:
                async for chunk in await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    stream=True,
                    **(params or {})
                ):
                    yield chunk.choices[0].delta.get("content", "")
            return stream_gen()
    elif PROVIDER == "hf":
        # Simple Hugging Face text generation (mock streaming)
        prompt = messages[-1]["content"]
        return f"(HF response to: {prompt})"
    else:
        # Mock
        return f"(mock response to: {messages[-1]['content']})"

# ---------------- FastAPI ----------------
app = FastAPI(title="Multi-Provider AI Server (Supabase REST)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"status": "ok", "provider": PROVIDER, "model": DEFAULT_MODEL}

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
        raise HTTPException(400, f"Moderation blocked: {reason}")

    out = await provider_chat(req.model or DEFAULT_MODEL, [{"role": "user", "content": req.prompt}], stream=False)
    await save_message(client_id, "user", req.prompt)
    await save_message(client_id, "assistant", out)
    return {"text": out}

@app.post("/chat")
async def chat(
    user_id: str = Form("guest"),
    prompt: str = Form(...),
    request: Request = None,
    x_api_key: Optional[str] = Header(None)
):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)
    history.append({"role": "user", "content": prompt})
    out = await provider_chat(DEFAULT_MODEL, history)

    await save_message(user_id, "user", prompt)
    await save_message(user_id, "assistant", out)
    return {"response": out}

@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64):
    return {"history": await get_history(user_id, limit)}

@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    total = await get_total_messages()
    return {"total_messages": total, "rate_limit": len(_rate_limit_store)}

# ---------- SSE streaming endpoint ----------
@app.post("/stream")
async def stream_chat(prompt: str = Form(...), request: Request = None):
    messages = [{"role": "user", "content": prompt}]
    async def event_generator():
        gen = await provider_chat(DEFAULT_MODEL, messages, stream=True)
        async for chunk in gen:
            yield {"data": chunk}
    return EventSourceResponse(event_generator())
