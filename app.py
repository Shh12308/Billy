# app.py — Mistral 7B (quantized) + Hive moderation + Supabase REST
import os
import time
import uuid
import logging
import asyncio
from typing import Optional, List, Dict, AsyncGenerator

import httpx
from fastapi import FastAPI, Request, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Transformer model imports (may require bitsandbytes + accelerate)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ------------- Config & Logging -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mistral-server")

PROVIDER = "mistral-local"
DEFAULT_MODEL = os.getenv("LOCAL_MODEL", "mistralai/Mistral-7B-Instruct-v0")
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin123")

# Supabase REST config (use env in production)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://orozxlbnurnchwodzfdt.supabase.co/rest/v1")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # REQUIRED in production; for testing you can hardcode
if not SUPABASE_KEY:
    logger.warning("SUPABASE_KEY not set — history saving will fail unless provided.")

SB_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}" if SUPABASE_KEY else "",
    "Content-Type": "application/json"
}

# Hive moderation key (REQUIRED if you want live moderation)
HIVE_API_KEY = os.getenv("HIVE_API_KEY")
if not HIVE_API_KEY:
    logger.warning("HIVE_API_KEY not set — hive moderation will not work until set.")

# Local model options via env
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"  # if true, don't use GPU
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"  # try quantized load
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

# ------------- Rate limiting (in-memory) -------------
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

# ------------- Supabase REST helpers -------------
async def save_message(user_id: str, role: str, content: str):
    if not SUPABASE_KEY:
        logger.debug("SUPABASE_KEY not set — skipping save_message")
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
        return int(r.headers.get("content-range", "0").split("/")[-1])

# ------------- Hive moderation -------------
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
    # quick rules-first check
    if not text:
        return True, None
    low = text.lower()
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in low for w in banned):
        return False, "Rule-based block"
    # hive.ai check if key available
    if HIVE_API_KEY:
        try:
            res = await hive_moderate(text)
            # safe-guard: if hive returns error structure, allow by default
            status = res.get("status", [])
            if status and isinstance(status, list) and "response" in status[0]:
                classes = status[0]["response"].get("output_text", {}).get("classes", [])
                for c in classes:
                    if c.get("score", 0) > 0.65:
                        return False, f"Blocked by Hive: {c.get('class')} ({c.get('score'):.2f})"
        except Exception as e:
            logger.exception("Hive moderation error — permissive fallback")
            return True, None
    return True, None

# ------------- Load Mistral 7B (try quantized) -------------
model = None
tokenizer = None
generator = None
model_ready = False

def try_load_model():
    global model, tokenizer, generator, model_ready
    try:
        logger.info(f"Attempting to load model: {DEFAULT_MODEL} (load_in_4bit={LOAD_IN_4BIT}, force_cpu={FORCE_CPU})")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=True)

        load_kwargs = {}
        # prefer GPU auto if available and not forced to CPU
        if LOAD_IN_4BIT:
            # load in 4bit (requires bitsandbytes)
            load_kwargs.update({
                "load_in_4bit": True,
                "device_map": "auto" if (torch.cuda.is_available() and not FORCE_CPU) else {"": "cpu"},
                "torch_dtype": torch.float16
            })
        else:
            load_kwargs.update({
                "device_map": "auto" if (torch.cuda.is_available() and not FORCE_CPU) else {"": "cpu"},
                "torch_dtype": torch.float16 if torch.cuda.is_available() and not FORCE_CPU else torch.float32
            })

        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, **load_kwargs)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        model_ready = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load local Mistral model — falling back to mock responses.")
        model_ready = False

# attempt load in background to avoid blocking import (but we load synchronously here)
try_load_model()

# ------------- Provider (uses local model if ready, else fallback mock) -------------
async def provider_chat(messages: List[Dict], max_new_tokens: int = MAX_NEW_TOKENS):
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    if model_ready and generator:
        loop = asyncio.get_event_loop()
        # run generation in executor to avoid blocking event loop
        def gen():
            out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, temperature=0.7)
            # pipeline returns list with 'generated_text'
            return out[0]["generated_text"]
        result = await loop.run_in_executor(None, gen)
        return result
    else:
        # fallback: simple canned reply
        return f"(local model not available — fallback response to: {messages[-1]['content']})"

# ------------- FastAPI app -------------
app = FastAPI(title="Mistral 7B Server (quantized) with Hive moderation")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str
    parameters: Optional[dict] = None

@app.get("/")
async def root():
    return {"status": "ok", "model": DEFAULT_MODEL, "model_ready": model_ready}

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(req.prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    out = await provider_chat([{"role":"user","content": req.prompt}])
    # save history (best-effort)
    try:
        await save_message(client_id, "user", req.prompt)
        await save_message(client_id, "assistant", out)
    except Exception:
        logger.exception("Failed saving history (non-fatal).")
    return {"text": out}

@app.post("/chat")
async def chat(user_id: str = Form("guest"), prompt: str = Form(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)

    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(400, f"Moderation blocked: {reason}")

    history = await get_history(user_id, limit=12)
    history.append({"role":"user","content": prompt})
    out = await provider_chat(history)
    try:
        await save_message(user_id, "user", prompt)
        await save_message(user_id, "assistant", out)
    except Exception:
        logger.exception("Failed saving history (non-fatal).")
    return {"response": out}

@app.get("/history/{user_id}")
async def history_endpoint(user_id: str, limit: int = 64):
    return {"history": await get_history(user_id, limit)}

@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    total = await get_total_messages()
    return {"total_messages": total, "rate_limit": len(_rate_limit_store)}

@app.post("/stream")
async def stream_chat(prompt: str = Form(...)):
    async def event_generator():
        out = await provider_chat([{"role":"user","content": prompt}], max_new_tokens=MAX_NEW_TOKENS)
        yield {"data": out}
    return EventSourceResponse(event_generator())
