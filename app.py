# main.py
"""
Groq-powered FastAPI "main.py" for streaming & batch generation.

USAGE:
  - Set environment variables:
      GROQ_API_URL   -> Full HTTP endpoint for model invocation (e.g. https://api.groq.example/v1/invoke)
      GROQ_API_KEY   -> API key for provider
      OPENAI_API_KEY -> optional; used for moderation if present
      DEFAULT_MODEL  -> default model id (optional)
  - Run: uvicorn main:app --host 0.0.0.0 --port $PORT

NOTES:
  - This file intentionally keeps provider call generic. If Groq's real API differs in payload/headers,
    update groq_invoke() accordingly.
  - WebSocket streaming attempts to forward streaming chunks when the provider streams newline-delimited
    JSON/lines or server-sent-events. If not supported by your provider, then generate endpoint will return
    the final text and websocket will chunk it for "simulated streaming".
"""

import os
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Form
from fastapi import UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import uuid

# -----------------------
# Logging & config
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("groq-server")

GROQ_API_URL = os.getenv("GROQ_API_URL")         # e.g. https://api.groq.example/v1/models/{model}/invoke
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     # optional moderation
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-like-model")  # change to a real model id for your provider

# Timeout for provider calls
PROVIDER_TIMEOUT = int(os.getenv("PROVIDER_TIMEOUT", "120"))

# Basic rate-limit placeholder (per-process, simple)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 60
_rate_limit_store: Dict[str, List[float]] = {}

# -----------------------
# App init
# -----------------------
app = FastAPI(title="Groq-Adapter AI", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory very small chat memory (per-user). Replace with DB for prod.
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# -----------------------
# Utilities
# -----------------------
def rate_limit_key(client_id: str) -> str:
    return f"rl:{client_id}"

def check_rate_limit(client_id: str):
    key = rate_limit_key(client_id)
    now = time.time()
    entries = _rate_limit_store.get(key, [])
    # drop old
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[key] = entries

async def moderate_text_if_configured(text: str) -> (bool, Optional[str]):
    """
    If OPENAI_API_KEY set, call OpenAI moderation endpoint to check text.
    If not set, do a small heuristic check.
    Returns (allowed, reason_if_blocked)
    """
    if not text:
        return True, None
    if OPENAI_API_KEY:
        # Minimal OpenAI moderation call using httpx to /moderations (v1)
        try:
            url = "https://api.openai.com/v1/moderations"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "omni-moderation-latest", "input": text}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                j = r.json()
                # Follow OpenAI structure: results[0].categories or flagged
                if isinstance(j, dict) and j.get("results"):
                    res = j["results"][0]
                    flagged = res.get("flagged", False)
                    if flagged:
                        return False, "Blocked by moderation"
                return True, None
        except Exception as e:
            logger.warning("OpenAI moderation failed: %s", e)
            # fallback to heuristic below
    # heuristic fallback
    banned = ["bomb", "kill", "terror", "explosive", "child abuse"]
    low = text.lower()
    if any(b in low for b in banned):
        return False, "Blocked by heuristic"
    return True, None

# -----------------------
# Provider invocation wrapper (Groq or generic)
# -----------------------
async def groq_invoke(model: str, prompt: str, stream: bool = False, parameters: Optional[Dict[str, Any]] = None, timeout: int = PROVIDER_TIMEOUT) -> Any:
    """
    Generic provider call. Expects provider to accept:
      POST {GROQ_API_URL} with headers Authorization: Bearer <GROQ_API_KEY>
      Body JSON: { "model": model, "input": prompt, "parameters": {...} }
    If your provider requires a different shape, change here.

    If stream=True and the provider streams, this generator yields streaming chunks (strings).
    Otherwise returns final JSON/text.
    """
    if not GROQ_API_URL or not GROQ_API_KEY:
        raise RuntimeError("Provider not configured: set GROQ_API_URL and GROQ_API_KEY")

    # Construct URL; if GROQ_API_URL contains {model} allow substitution
    if "{model}" in GROQ_API_URL:
        url = GROQ_API_URL.format(model=model)
    else:
        url = GROQ_API_URL  # you might include model via payload depending on API

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": model,
        "input": prompt,
    }
    if parameters:
        payload["parameters"] = parameters

    # If the provider supports streaming responses via chunked-transfer or server-sent events
    # you'll want to set stream=True and the provider must support it.
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if stream:
                # Attempt streaming call (Transfer-Encoding: chunked)
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=text.decode(errors="ignore"))
                    # yield text lines/chunks as they arrive
                    async for chunk in resp.aiter_text():
                        if not chunk:
                            continue
                        yield chunk
                    return
            else:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                try:
                    return r.json()
                except Exception:
                    return r.text
        except httpx.HTTPStatusError as e:
            logger.error("Provider returned error: %s %s", e.response.status_code, e.response.text[:400])
            raise HTTPException(status_code=500, detail=f"Provider error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            logger.exception("Provider call failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Routes
# -----------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "groq_configured": bool(GROQ_API_URL and GROQ_API_KEY),
        "default_model": DEFAULT_MODEL
    }

@app.post("/generate")
async def generate(request: Request):
    """
    JSON body: { "model": "...", "prompt": "..." , "max_tokens": 256, "temperature":0.7 }
    """
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    prompt = body.get("prompt")
    params = body.get("parameters", {})
    client_id = request.client.host if request.client else "anon"
    check_rate_limit(client_id)

    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    allowed, reason = await moderate_text_if_configured(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")

    logger.info("Generate request model=%s len(prompt)=%d", model, len(prompt))
    # Non-streaming call; provider returns final output
    out = await groq_invoke(model=model, prompt=prompt, stream=False, parameters=params)
    # Try to extract textual output safely
    if isinstance(out, dict):
        # common fields: text, generated_text, output, result, choices
        for k in ("text", "generated_text", "output", "result"):
            if k in out:
                return {"source": "provider", "model": model, "text": out[k]}
        # check choices
        if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
            choice = out["choices"][0]
            if isinstance(choice, dict) and "text" in choice:
                return {"source": "provider", "model": model, "text": choice["text"]}
        return {"source": "provider", "model": model, "raw": out}
    else:
        return {"source": "provider", "model": model, "text": str(out)}

@app.post("/chat")
async def chat(user_id: Optional[str] = Form("guest"), prompt: Optional[str] = Form(None)):
    """
    Simple chat that stores last few messages in memory and calls provider.
    Form fields: user_id, prompt
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    # rate limit & moderation
    check_rate_limit(user_id)
    allowed, reason = await moderate_text_if_configured(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")

    history = CHAT_MEMORY.setdefault(user_id, [])
    # Compose context (very simple)
    convo = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history[-6:]])
    composed = f"{convo}\nUser: {prompt}\nAssistant:"
    model = DEFAULT_MODEL
    logger.info("Chat invoke user=%s model=%s", user_id, model)
    out = await groq_invoke(model=model, prompt=composed, stream=False, parameters={"max_tokens": 400})
    # Extract text
    text = ""
    if isinstance(out, dict):
        text = out.get("text") or out.get("generated_text") or out.get("output") or json.dumps(out)[:4000]
    else:
        text = str(out)
    # store
    history.append({"user": prompt, "assistant": text})
    # cap history
    CHAT_MEMORY[user_id] = history[-64:]
    return {"source": "provider", "model": model, "response": text}

# -----------------------
# WebSocket streaming endpoint
# -----------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Expect client to send one JSON message after connect:
    { "model": "model-id", "prompt": "Hello", "parameters": {...} }

    The endpoint tries to stream provider output to the websocket as JSON messages:
      {"delta": "<chunk>"}  -- incremental content
      {"done": true, "final": "<final text>"} -- finalization
      {"error": "..."} -- error
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        try:
            meta = json.loads(raw)
        except Exception:
            await websocket.send_json({"error": "Invalid JSON in initial message"})
            await websocket.close()
            return

        model = meta.get("model", DEFAULT_MODEL)
        prompt = meta.get("prompt")
        parameters = meta.get("parameters", {})
        client_id = websocket.client.host if websocket.client else "anon-ws"

        if not prompt:
            await websocket.send_json({"error": "Missing prompt"})
            await websocket.close()
            return

        check_rate_limit(client_id)
        allowed, reason = await moderate_text_if_configured(prompt)
        if not allowed:
            await websocket.send_json({"error": f"Moderation blocked: {reason}"})
            await websocket.close()
            return

        # Try provider streaming first. If provider yields chunks, forward them.
        stream_gen = groq_invoke(model=model, prompt=prompt, stream=True, parameters=parameters)
        is_stream_generator = hasattr(stream_gen, "__aiter__")

        if is_stream_generator:
            # Forward streaming provider chunks directly (best-effort)
            try:
                partial = ""
                async for chunk in stream_gen:
                    # chunk may be large; try to parse newline-separated tokens or plain text
                    if not chunk:
                        continue
                    # forward in moderate-sized deltas
                    # try to split by newline and send each
                    for part in chunk.splitlines():
                        if not part:
                            continue
                        await websocket.send_json({"delta": part})
                # after stream ends, done
                await websocket.send_json({"done": True})
                await websocket.close()
                return
            except Exception as e:
                # fallback to non-stream flow below
                logger.exception("Streaming provider failed: %s", e)

        # If we reach here the provider doesn't stream: do non-stream call and chunk result to websocket
        out = await groq_invoke(model=model, prompt=prompt, stream=False, parameters=parameters)
        if isinstance(out, dict):
            # try to extract text
            for k in ("text", "generated_text", "output", "result"):
                if k in out:
                    final = out[k]
                    break
            else:
                if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
                    final = out["choices"][0].get("text") or out["choices"][0].get("message", {}).get("content", "")
                else:
                    final = json.dumps(out)
        else:
            final = str(out)

        # chunk final into words (or moderate-sized pieces) and send
        async def chunk_and_send(text_to_send: str):
            # Try to send sentence by sentence first
            sent_seps = ".!?;\n"
            idx = 0
            # split on whitespace into tokens but group tokens ~ up to 100 chars for speed
            tokens = text_to_send.split()
            cur = ""
            for tok in tokens:
                if len(cur) + 1 + len(tok) <= 140:
                    cur = (cur + " " + tok).strip()
                else:
                    await websocket.send_json({"delta": cur})
                    cur = tok
                    await asyncio.sleep(0.02)
            if cur:
                await websocket.send_json({"delta": cur})
        await chunk_and_send(final)
        await websocket.send_json({"done": True, "final": final})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except Exception:
            pass

# -----------------------
# Embedding endpoint (simple)
# -----------------------
@app.post("/embed")
async def embed(prompt: str = Form(...)):
    """
    Generate embeddings (provider-dependent). Body form: prompt
    """
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    if not (GROQ_API_URL and GROQ_API_KEY):
        raise HTTPException(status_code=503, detail="Provider not configured")
    # Use a naming convention model="embed-model" or allow user to override via env
    model = os.getenv("EMBED_MODEL", "embed-model")
    out = await groq_invoke(model=model, prompt=prompt, stream=False, parameters={"task": "embed"})
    return {"source": "provider", "model": model, "embeddings": out}

# -----------------------
# Admin / info endpoints
# -----------------------
@app.get("/admin/models")
async def admin_models():
    return {
        "configured": {
            "groq_api_url": bool(GROQ_API_URL),
            "groq_api_key": bool(GROQ_API_KEY),
            "default_model": DEFAULT_MODEL
        },
        "notes": "Set GROQ_API_URL to your provider's invoke endpoint. If it contains {model}, it will be substituted."
    }

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("Starting Groq-Adapter on port %d", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
