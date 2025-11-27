# app.py — Groq Chat API + OpenRouter moderation + Supabase History
import os
import time
import uuid
import logging
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixtral-groq-server")

# -------- Groq Provider --------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"

async def groq_chat(messages: List[Dict], model: str = GROQ_MODEL):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except Exception:
            logger.error(f"Groq response text: {r.text}")
            raise
        return r.json()

async def provider_chat(messages: List[Dict]):
    try:
        j = await groq_chat(messages)
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Groq API error")
        return f"(Groq error: {e})"

# -------- OpenRouter moderation --------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def openrouter_moderate(text: str):
    if not OPENROUTER_API_KEY:
        return {"error": "no_key"}

    url = "https://openrouter.ai/api/v1/moderations"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": "openai-moderation-latest", "input": text}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            return r.json()
        except:
            logger.warning(f"OpenRouter moderation non-JSON: {r.text}")
            return {"error": "bad_json"}

async def moderate_text(text: str):
    if not text:
        return True, None

    # Fast word ban
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based filter"

    if OPENROUTER_API_KEY:
        res = await openrouter_moderate(text)
        try:
            categories = res["results"][0]["categories"]
            if any(categories.get(cat, False) for cat in categories):
                return False, "Blocked by OpenRouter moderation"
        except:
            return True, None  # fail-open

    return True, None

# -------- Supabase --------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

async def save_message(user_id, role, content):
    if not SUPABASE_KEY:
        return
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            json={
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "role": role,
                "content": content,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

async def get_history(user_id, limit=32):
    if not SUPABASE_KEY:
        return []
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit},
        )
        r.raise_for_status()
        return [
            {"role": row["role"], "content": row["content"]}
            for row in r.json()
        ]

# -------- Rate-limit --------
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_store = {}

def get_client_id(request: Request, key: Optional[str]):
    if key:
        return f"key:{key}"
    return f"ip:{request.client.host}"

def apply_rate_limit(cid):
    now = time.time()
    entries = _rate_store.get(cid, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Rate limit exceeded")
    entries.append(now)
    _rate_store[cid] = entries

# -------- FastAPI app --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateReq(BaseModel):
    prompt: str
    model: Optional[str] = None

@app.post("/generate")
async def generate(req: GenerateReq, request: Request, x_api_key: Optional[str] = Header(None)):
    cid = get_client_id(request, x_api_key)
    apply_rate_limit(cid)

    ok, reason = await moderate_text(req.prompt)
    if not ok:
        raise HTTPException(400, f"Moderation: {reason}")

    out = await provider_chat([{"role": "user", "content": req.prompt}])
    return {"text": out}

# -------- The IMPORTANT endpoint your front-end uses --------
@app.post("/chat")
async def chat(
    user_id: str = Form("guest"),
    prompt: str = Form(...),
    request: Request = None,
    x_api_key: Optional[str] = Header(None)
):
    cid = get_client_id(request, x_api_key)
    apply_rate_limit(cid)

    ok, reason = await moderate_text(prompt)
    if not ok:
        raise HTTPException(400, f"Moderation: {reason}")

    history = await get_history(user_id, limit=12)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": prompt})

    out = await provider_chat(messages)

    try:
        await save_message(user_id, "user", prompt)
        await save_message(user_id, "assistant", out)
    except:
        pass

    return {"response": out}

@app.get("/")
def home():
    return {"status": "ok", "model": GROQ_MODEL}
