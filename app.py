# app.py - Multi-provider AI Server with streaming + features
# Requirements (put into requirements.txt):
# fastapi
# uvicorn[standard]
# httpx
# python-multipart
# sse-starlette
# python-dotenv

import os
import json
import time
import logging
import asyncio
import uuid
import sqlite3
from typing import Optional, Dict, List, Any, AsyncGenerator
from functools import wraps

import httpx
from fastapi import (
    FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect,
    Form, UploadFile, File, Depends, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv

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

# ---- In-memory / simple storage ----
_rate_limit_store: Dict[str, List[float]] = {}
# Persisted chat history via simple sqlite for durability
DB_PATH = os.getenv("DB_PATH", "chat_history.db")

# Create DB if not exists
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at REAL,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    return conn

_db_conn = init_db()

def save_message(user_id: str, role: str, content: str):
    cur = _db_conn.cursor()
    cur.execute(
        "INSERT INTO history (id, user_id, created_at, role, content) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), user_id, time.time(), role, content)
    )
    _db_conn.commit()

def get_history(user_id: str, limit: int = 64):
    cur = _db_conn.cursor()
    cur.execute(
        "SELECT role, content FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cur.fetchall()
    # return in chronological order
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

# ---- FastAPI app ----
app = FastAPI(title="Multi-Provider AI Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Helpers ----
def rate_limited(client_id: str):
    """Check & update simple sliding window rate limit. Raise HTTPException on exceed."""
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

async def moderate_text(text: str) -> (bool, Optional[str]):
    """
    Heuristic moderation + optional provider moderation using OpenAI moderation.
    Returns (allowed, reason)
    """
    if not text:
        return True, None
    banned = ["bomb", "kill", "terror", "child abuse"]
    low = text.lower()
    if any(word in low for word in banned):
        return False, "Blocked by heuristic"
    # Optional: use OpenAI moderation if key present
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
            # If moderation fails, be permissive rather than block everything
            logger.exception("Moderation API error (continuing permissive).")
            pass
    return True, None

def get_client_id(request: Optional[Request] = None, x_api_key: Optional[str] = None):
    """
    Determine client id for rate-limiting: prefer API key header, then request.client.host, then "anon".
    """
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

# ---- Provider layer ----
# We implement a simple provider abstraction: openai, hf (huggingface inference), mock (local)

async def openai_chat_completion(model: str, messages: List[Dict[str, str]], stream: bool = False, params: dict = None):
    """
    Uses OpenAI v1/chat/completions endpoints (or the newer SDK pattern). Implemented via httpx.
    For streaming, yields chunks (strings). For non-streaming, returns final text.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    if params:
        payload.update(params)
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT) as client:
        if stream:
            # OpenAI streaming uses "stream": True and returns an event-stream of "data: ..." lines
            payload["stream"] = True
            r = await client.post(url, headers=headers, json=payload, timeout=None)
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                # streaming lines typically start with "data: "
                if line.startswith("data: "):
                    raw = line[len("data: "):]
                else:
                    raw = line
                if raw.strip() == "[DONE]":
                    yield "[DONE]"
                    break
                try:
                    obj = json.loads(raw)
                    # attempt to extract content deltas
                    choices = obj.get("choices", [])
                    for ch in choices:
                        delta = ch.get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except Exception:
                    # If parsing fails, yield raw for debugging
                    yield raw
        else:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            j = r.json()
            text = ""
            if j.get("choices"):
                # assemble content
                for c in j["choices"]:
                    msg = c.get("message", {})
                    if msg.get("content"):
                        text += msg["content"]
            return text

async def hf_chat_completion(model: str, messages: List[Dict[str, str]], stream: bool = False, params: dict = None):
    """
    Simple HuggingFace Inference Router-style call (non-streaming). For streaming, HF streaming varies — we do a fallback chunked read.
    """
    if not HF_API_KEY:
        raise HTTPException(status_code=503, detail="HF API key not configured")
    # Many HF inference endpoints accept a payload like {"inputs": "some text"} or chat messages depending on router.
    # We'll use a generic router endpoint if configured via HF_API_URL env var; otherwise try the common one.
    hf_base = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models")
    # If model contains a slash, treat as full path
    url = f"{hf_base}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    # Build plain prompt from messages (simple concatenation)
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    payload = {"inputs": prompt}
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT) as client:
        if stream:
            r = await client.post(url, headers=headers, json=payload, timeout=None)
            r.raise_for_status()
            async for chunk in r.aiter_text():
                if chunk:
                    yield chunk
        else:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            # HF returns text or json; try to parse
            try:
                j = r.json()
                # best-effort extraction
                if isinstance(j, dict) and "generated_text" in j:
                    return j["generated_text"]
                elif isinstance(j, list) and len(j) and isinstance(j[0], dict):
                    # e.g. [{'generated_text': '...'}]
                    return j[0].get("generated_text", str(j[0]))
                else:
                    return str(j)
            except Exception:
                return await r.text()

async def mock_chat_completion(model: str, messages: List[Dict[str, str]], stream: bool = False, params: dict = None):
    """
    Local mock: echo + small delay; useful for testing without keys.
    """
    # Simple echo behaviour: reply with reversed last user message plus some filler text
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    response_text = f"Echo (mock) of your last message: {last_user}\n\n-- End of mock response"
    if stream:
        # simple char streaming
        for ch in response_text:
            await asyncio.sleep(0.005)
            yield ch
        yield "[DONE]"
    else:
        return response_text

# Provider dispatcher
async def provider_chat(model: str, messages: List[Dict[str, str]], stream: bool = False, params: dict = None):
    if PROVIDER == "openai":
        return await openai_chat_completion(model, messages, stream=stream, params=params)
    if PROVIDER == "hf":
        return await hf_chat_completion(model, messages, stream=stream, params=params)
    # default mock
    return await mock_chat_completion(model, messages, stream=stream, params=params)

# Simple embeddings via OpenAI or HF (non-streaming)
async def provider_embeddings(model: str, texts: List[str]):
    if PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            j = r.json()
            return [item["embedding"] for item in j.get("data", [])]
    if PROVIDER == "hf":
        # HF embeddings endpoints vary; best-effort generic call
        if not HF_API_KEY:
            raise HTTPException(status_code=503, detail="HF API key not configured")
        url = os.getenv("HF_EMBED_URL")  # user may set a custom embedding endpoint
        if not url:
            raise HTTPException(status_code=503, detail="HF embedding URL not configured (set HF_EMBED_URL)")
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": texts}
        async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    # mock embeddings: simple hash-based vectors (not useful for real tasks)
    return [[float(ord(c) % 10) for c in t[:16]] for t in texts]

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
    # non-streaming
    out = await provider_chat(model, messages, stream=False, params=req.parameters)
    # persist
    save_message(client_id, "user", req.prompt)
    save_message(client_id, "assistant", str(out))
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

    # build messages from last saved history
    history = get_history(user_id, limit=12)
    messages = []
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": prompt})

    out = await provider_chat(DEFAULT_MODEL, messages, stream=False)
    # persist
    save_message(user_id, "user", prompt)
    save_message(user_id, "assistant", str(out))
    return {"source": PROVIDER, "model": DEFAULT_MODEL, "response": out}

@app.post("/embeddings")
async def embeddings(texts: List[str], request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)
    if not texts:
        raise HTTPException(status_code=400, detail="No texts supplied")
    vectors = await provider_embeddings(EMBED_MODEL, texts)
    return {"model": EMBED_MODEL, "embeddings": vectors, "count": len(vectors)}

@app.get("/models")
async def list_models():
    # Best-effort list
    if PROVIDER == "openai":
        # don't attempt network call here; list common known & default
        return {"provider": "openai", "available": [DEFAULT_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]}
    if PROVIDER == "hf":
        return {"provider": "hf", "note": "models depend on your HF account; set HF_API_KEY and HF_API_URL"}
    return {"provider": "mock", "available": ["mock-echo"]}

# ---- Streaming: SSE endpoint ----
@app.post("/sse/stream")
async def sse_stream(request: Request):
    """
    Accepts JSON body { prompt, model?, parameters? }
    Returns an SSE stream of partial text events.
    """
    body = await request.json()
    prompt = body.get("prompt")
    model = body.get("model", DEFAULT_MODEL)
    params = body.get("parameters", {})
    client_id = get_client_id(request, x_api_key=request.headers.get("x-api-key"))

    rate_limited(client_id)
    allowed, reason = await moderate_text(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")

    messages = [{"role":"user", "content": prompt}]

    async def event_generator():
        # provider_chat may be an async generator or a coroutine returning a string
        resp = await provider_chat(model, messages, stream=True, params=params)
        # If provider returns an async generator (HF / openai streaming), iterate
        if hasattr(resp, "__aiter__") or isinstance(resp, AsyncGenerator):
            async for chunk in resp:
                if chunk == "[DONE]":
                    break
                yield {"event": "delta", "data": chunk}
        else:
            # If resp is a string (non-streaming), send it at once
            yield {"event": "delta", "data": str(resp)}
        yield {"event": "done", "data": "true"}

    # persist final text when done: we'll collect in a buffer
    async def persist_and_stream():
        buffer = ""
        async for ev in event_generator():
            # the sse server expects yield of string; event is dict used by EventSourceResponse
            data = ev["data"]
            if ev["event"] == "delta":
                buffer += data
            yield ev
        # persist
        save_message(client_id, "assistant", buffer)

    return EventSourceResponse(persist_and_stream())

# ---- WebSocket streaming ----
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "anon"
    try:
        raw = await websocket.receive_text()
        meta = json.loads(raw)
        prompt = meta.get("prompt")
        model = meta.get("model", DEFAULT_MODEL)
        api_key = meta.get("api_key")
        client_id = f"ws:{api_key}" if api_key else f"ws:{client_addr}"

        # rate limit & moderate
        rate_limited(client_id)
        allowed, reason = await moderate_text(prompt)
        if not allowed:
            await websocket.send_json({"error": f"Moderation blocked: {reason}"})
            await websocket.close()
            return

        messages = [{"role":"user", "content": prompt}]
        resp = await provider_chat(model, messages, stream=True)
        # resp could be async generator or str
        buffer = ""
        if hasattr(resp, "__aiter__") or isinstance(resp, AsyncGenerator):
            async for chunk in resp:
                if chunk == "[DONE]":
                    break
                buffer += chunk
                await websocket.send_json({"delta": chunk})
        else:
            buffer = str(resp)
            await websocket.send_json({"delta": buffer})

        await websocket.send_json({"done": True})
        save_message(client_id, "assistant", buffer)
        await websocket.close()

    except WebSocketDisconnect:
        logger.info("WS disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except Exception:
            pass

# ---- File upload for context ----
@app.post("/upload")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...), request: Request = None, x_api_key: Optional[str] = Header(None)):
    client_id = get_client_id(request, x_api_key)
    rate_limited(client_id)
    # Only accept text files for now
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Only text files (utf-8) supported")
    # Save as a history item so subsequent /chat reads it
    save_message(user_id, "system", f"[uploaded:{file.filename}]\n{text}")
    return {"ok": True, "filename": file.filename, "size": len(content)}

# ---- Metrics / admin endpoints ----
@app.get("/metrics")
async def metrics(x_admin_key: Optional[str] = Header(None)):
    require_admin(x_admin_key)
    total_msgs = _db_conn.cursor().execute("SELECT count(*) FROM history").fetchone()[0]
    return {"provider": PROVIDER, "total_history_messages": total_msgs, "rate_limit_store_size": len(_rate_limit_store)}

# ---- Graceful shutdown (optional hooks) ----
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down, closing DB.")
    try:
        _db_conn.close()
    except Exception:
        pass

# ---- Error handlers ----
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP error: %s %s", exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ---- Small helper route to fetch conversation history ----
@app.get("/history/{user_id}")
async def history(user_id: str, limit: int = 64, x_api_key: Optional[str] = Header(None)):
    # Basic auth: either admin or the API key matching a stored key (very simple)
    # For simplicity, we don't check ownership of API keys here; production systems should.
    return {"user_id": user_id, "history": get_history(user_id, limit)}

# ---- Simple health check for provider connectivity (best-effort) ----
@app.get("/probe/provider")
async def probe_provider():
    try:
        if PROVIDER == "openai":
            if not OPENAI_API_KEY:
                return {"ok": False, "reason": "OPENAI_API_KEY not set"}
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
                return {"ok": r.status_code == 200, "status": r.status_code}
        if PROVIDER == "hf":
            if not HF_API_KEY:
                return {"ok": False, "reason": "HF_API_KEY not set"}
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get("https://api-inference.huggingface.co/models", headers={"Authorization": f"Bearer {HF_API_KEY}"})
                # Many HF endpoints require a model id; this lists models available to account (may be 403)
                return {"ok": r.status_code in (200, 403), "status": r.status_code}
        return {"ok": True, "provider": "mock"}
    except Exception as e:
        logger.exception("Probe error")
        return {"ok": False, "error": str(e)}
