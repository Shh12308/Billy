# app.py — Mixtral via Groq + OpenRouter moderation + Supabase REST
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

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mixtral-groq-server")

# ---------------- Groq Provider ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_CHAT_ENDPOINT = f"{GROQ_BASE}/chat/completions"


# ============================================================
# SAFE VERSION — NEVER sends bad payloads, auto-cleans messages
# ============================================================
async def _call_groq(messages: List[Dict], model: str):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    # ---------------- VALIDATE MODEL ----------------
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Invalid model name: {model}")

    # ---------------- VALIDATE MESSAGES ----------------
    cleaned_messages = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(f"Skipping non-dict message at index {idx}: {msg}")
            continue

        role = msg.get("role")
        content = msg.get("content")

        if role not in ("system", "user", "assistant"):
            logger.warning(f"Invalid role in message {idx}: {role}. Skipping.")
            continue

        if not isinstance(content, str) or not content.strip():
            logger.warning(f"Empty/invalid content in message {idx}. Skipping.")
            continue

        cleaned_messages.append({"role": role, "content": content})

    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "(empty)"}]

    payload = {
        "model": model,
        "messages": cleaned_messages
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

        except httpx.HTTPStatusError as e:
            logger.error("Groq rejected payload with 400:")
            logger.error(payload)
            logger.error(f"Groq response: {e.response.text}")
            raise


# ---------------- Wrapper for fallback ----------------
async def groq_chat_completion(messages: List[Dict], model: str = GROQ_MODEL):
    try:
        return await _call_groq(messages, model)

    except httpx.HTTPStatusError as e:
        logger.error(f"Groq HTTP error: {e}")

        resp_text = e.response.text or ""
        if ("model" in resp_text and "decommissioned" in resp_text):
            fallback_model = "llama-3.1-8b-instant"
            if model != fallback_model:
                logger.info(f"Retrying Groq with fallback model '{fallback_model}'")
                return await _call_groq(messages, fallback_model)

        raise


async def provider_chat(messages: List[Dict]):
    try:
        j = await groq_chat_completion(messages)
        choices = j.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content") or choices[0].get("text") or "(empty response)"
        return "(empty response)"
    except Exception as e:
        logger.exception("Groq API error")
        return f"(Groq error: {str(e)})"


# ---------------- OpenRouter moderation ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def openrouter_moderate(text: str):
    if not OPENROUTER_API_KEY:
        return {"error": "no_key"}
    url = "https://openrouter.ai/api/v1/moderations"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "openai-moderation-latest", "input": text}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            return r.json()
        except Exception:
            logger.warning(f"OpenRouter returned non-JSON: {r.text}")
            return {"error": "bad_json", "text": r.text}

async def moderate_text(text: str):
    if not text:
        return True, None

    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based safety"

    if OPENROUTER_API_KEY:
        try:
            res = await openrouter_moderate(text)
            categories = res.get("results", [{}])[0].get("categories", {})
            flagged = any(categories.get(cat, False) for cat in categories)
            if flagged:
                return False, "Blocked by OpenRouter moderation"
        except Exception:
            logger.exception("OpenRouter moderation error — permissive fallback")
            return True, None

    return True, None


# ---------------- Supabase helpers ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json",
}

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
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{SUPABASE_URL}/history", headers=SB_HEADERS, json=payload)
        r.raise_for_status()

async def get_history(user_id: str, limit: int = 64):
    if not SUPABASE_KEY:
        return []
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"user_id": f"eq.{user_id}", "order": "created_at.asc", "limit": limit},
        )
        r.raise_for_status()
        return [{"role": r["role"], "content": r["content"]} for r in r.json()]

async def get_total_messages():
    if not SUPABASE_KEY:
        return 0
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            f"{SUPABASE_URL}/history",
            headers=SB_HEADERS,
            params={"select": "count", "count": "exact", "limit": 1},
        )
        r.raise_for_status()
        cr = r.headers.get("content-range", "0/0")
        parts = cr.split("/")
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0


# ---------------- Rate-limiter ----------------
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, List[float]] = {}

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


# ---------------- FastAPI app ----------------
app = FastAPI(title="Mixtral (Groq) Server + OpenRouter moderation")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    parameters: Optional[dict] = None

@app.get("/")
async def root():
    return {"status": "ok", "provider": "groq", "model": GROQ_MODEL}


# ---------------- /generate ----------------
@app.post("/generate")
async def generate(req: GenerateRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(req.prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    messages = [{"role": "user", "content": req.prompt}]
    out = await provider_chat(messages)

    try:
        await save_message(client_id, "user", req.prompt)
        await save_message(client_id, "assistant", out)
    except Exception:
        logger.exception("Failed to save history (non-fatal).")

    return {"text": out}


# ---------------- /chat ----------------
@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)

    # Clean history so Groq gets only valid messages
    messages = [
        {"role": h["role"], "content": h["content"]}
        for h in history
        if isinstance(h, dict)
        and h.get("role") in ("system", "user", "assistant")
        and isinstance(h.get("content"), str)
    ]

    messages.append({"role": "user", "content": prompt})
    out = await provider_chat(messages)

    try:
        await save_message(user_id, "user", prompt)
        await save_message(user_id, "assistant", out)
    except Exception:
        logger.exception("Failed to save history (non-fatal).")

    return {"response": out}


# ---------------- /history ----------------
@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64):
    return {"history": await get_history(user_id, limit)}


# ---------------- /metrics ----------------
@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    if x_admin_key != os.getenv("ADMIN_KEY", "admin123"):
        raise HTTPException(403, "Invalid admin key")
    total = await get_total_messages()
    return {"total_messages": total, "rate_limit_store_size": len(_rate_limit_store)}


# ---------------- /stream ----------------
@app.post("/stream")
async def stream_chat(prompt: str = Form(...)):
    messages = [{"role":"user","content": prompt}]
    async def event_generator():
        out = await provider_chat(messages)
        yield {"data": out}
    return EventSourceResponse(event_generator())
