# main.py
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from email.message import EmailMessage
import smtplib
import re
from typing import Optional
import os
import asyncio
import logging
import tempfile
import uuid
import json
import uvicorn

# optional heavy imports
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

_transformers_available = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    _transformers_available = False

_diffusers_available = True
try:
    from diffusers import StableDiffusionPipeline
except Exception:
    _diffusers_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proai")

# ---------- Config (env driven) ----------
PORT = int(os.getenv("PORT", "8000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
# Default to LLaMA 3 8B Instruct HF name (change if you have a different path)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_API_KEY = os.getenv("HF_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
ENABLE_IMAGE_GEN = os.getenv("ENABLE_IMAGE_GEN", "false").lower() == "true"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "CompVis/stable-diffusion-v1-4")
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "shaynengunga15@gmail.com")

# SMTP settings for notify_admin
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", os.getenv("SMTP_PORT", "25")))
SMTP_USER = os.getenv("SMTP_USER", None)
SMTP_PASS = os.getenv("SMTP_PASS", None)

# About info (shows projects, refuses to disclose details)
ABOUT_INFO = {
    "creator": "GoldBoy",
    "age": 17,
    "bio": "I am a 17-year-old programmer working on multiple projects/sites.",
    "contact": "shaynengunga15@gmail.com",
    "projects": ["NGG", "ST", "MZ, BB, NL"]
}

# ---------- FastAPI app ----------
app = FastAPI(title="ProAI - GoldBoy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Redis client (lazy) ----------
redis_client = None
async def get_redis():
    global redis_client
    if redis_client is None:
        if aioredis is None:
            raise RuntimeError("redis.asyncio not installed - please install redis>=4.2 if you want Redis features")
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await redis_client.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.warning("⚠️ Redis connect failed: %s", e)
    return redis_client

# ---------- Model holders (lazy loaded) ----------
_text_tokenizer = None
_text_model = None
_image_pipe = None

async def load_text_model_if_needed():
    global _text_tokenizer, _text_model
    if _text_model is not None and _text_tokenizer is not None:
        return
    if not USE_LOCAL_MODEL:
        logger.info("USE_LOCAL_MODEL is false; skipping local model load")
        return
    if not _transformers_available:
        logger.error("Transformers not available. Install transformers and torch to load local models.")
        return

    logger.info("Loading local text model: %s", LOCAL_MODEL_NAME)
    try:
        _text_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, use_fast=True)
        # try to load with automatic device mapping; use float16 if CUDA available
        if torch.cuda.is_available():
            _text_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            _text_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, device_map="auto")
        logger.info("✅ Local text model loaded")
    except Exception as e:
        logger.exception("Failed to load local text model: %s", e)
        # ensure variables are cleared to avoid partially initialised state
        _text_tokenizer = None
        _text_model = None

async def load_image_model_if_needed():
    global _image_pipe
    if not ENABLE_IMAGE_GEN:
        return
    if _image_pipe is not None:
        return
    if not _diffusers_available:
        logger.error("Diffusers not installed; install diffusers and torch to enable image generation.")
        return
    try:
        device = "cuda" if ("torch" in globals() and torch.cuda.is_available()) else "cpu"
        _image_pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL)
        _image_pipe.to(device)
        logger.info("✅ Image pipeline ready on %s", device)
    except Exception as e:
        logger.exception("Failed loading image pipeline: %s", e)
        _image_pipe = None

# ---------- Pydantic models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    persona: Optional[str] = "default"
    stream: Optional[bool] = False
    code_mode: Optional[bool] = False

class ImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512

# ---------- Helpers ----------
def safe_filename(prefix="out", ext="png"):
    return f"{prefix}-{uuid.uuid4().hex}.{ext}"

async def generate_text_local(prompt: str, max_new_tokens: int = 256):
    """
    Generate text with local model. This may be slow and memory heavy.
    """
    if _text_model is None or _text_tokenizer is None:
        raise RuntimeError("Local model not loaded")
    inputs = _text_tokenizer(prompt, return_tensors="pt", truncation=True).to(_text_model.device)
    out = _text_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, top_k=50, temperature=0.8)
    text = _text_tokenizer.decode(out[0], skip_special_tokens=True)
    # remove echo of prompt, if present
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

async def generate_text_remote_hf(prompt: str):
    """
    Use Hugging Face Inference API (requires HF_API_KEY)
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set")
    import httpx
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "") or str(data)
        return str(data)

def extract_code_blocks(text: str):
    """
    finds triple-backtick code blocks and returns list of {language, code}
    """
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    blocks = []
    for lang, code in matches:
        blocks.append({"language": (lang or "text").lower(), "code": code.strip()})
    return blocks

# ---------- Public info endpoints ----------
@app.get("/about")
async def about():
    """
    Returns creator and project list but refuses to disclose project details.
    """
    return {
        "creator": ABOUT_INFO["creator"],
        "bio": ABOUT_INFO["bio"],
        "projects": ABOUT_INFO["projects"],
        "message": "Project details are private and will not be disclosed."
    }

# ---------- Admin notification endpoint ----------
@app.post("/notify_admin")
async def notify_admin(reason: str = Form(...)):
    """
    Sends a short email to ADMIN_EMAIL. Uses SMTP env settings if provided,
    otherwise attempts localhost SMTP.
    """
    msg = EmailMessage()
    msg.set_content(f"User reported AI issue: {reason}")
    msg["Subject"] = "AI Service Alert"
    msg["From"] = ADMIN_EMAIL
    msg["To"] = ADMIN_EMAIL

    try:
        if SMTP_HOST and SMTP_HOST != "localhost" and SMTP_USER:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            server.quit()
        else:
            # try localhost
            with smtplib.SMTP("localhost") as s:
                s.send_message(msg)
        return {"status": "notification sent"}
    except Exception as e:
        logger.exception("notify_admin failed: %s", e)
        return {"status": "failed", "error": str(e)}

# ---------- Startup tasks ----------
@app.on_event("startup")
async def startup_event():
    # Try connect redis (non-fatal)
    try:
        await get_redis()
    except Exception as e:
        logger.warning("Redis not available at startup: %s", e)

    # Preload local model if requested (background)
    if USE_LOCAL_MODEL:
        asyncio.create_task(load_text_model_if_needed())
    if ENABLE_IMAGE_GEN:
        asyncio.create_task(load_image_model_if_needed())

# ---------- Chat endpoints ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Lightweight chat endpoint using:
      1) Hugging Face Inference API
      2) OpenAI fallback
    Returns JSON { response: str, code_blocks?: [] }
    """
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }
    persona_prompt = persona_map.get(req.persona, persona_map["default"])

    # Build conversation history
    history = ""
    for m in req.messages:
        role = "User" if m.role == "user" else "Assistant"
        history += f"{role}: {m.content}\n"

    prompt = persona_prompt + "\n\n" + history + "Assistant:"
    out = None

    # 1) Hugging Face Inference API
    if HF_API_KEY:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}
            }
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    out = data[0].get("generated_text", "")
        except Exception:
            logger.exception("HF inference failed.")

    # 2) OpenAI fallback
    if out is None and OPENAI_API_KEY:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI fallback failed.")

    if not out:
        raise HTTPException(status_code=503, detail="No text generation backend available")

    response = {"response": out}
    if req.code_mode:
        response["code_blocks"] = extract_code_blocks(out)
    return JSONResponse(response)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE streaming endpoint using only cloud APIs.
    Streams text in chunks, then sends code blocks (if code_mode) and [DONE].
    """
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }
    persona_prompt = persona_map.get(req.persona, persona_map["default"])

    history = ""
    for m in req.messages:
        role = "User" if m.role == "user" else "Assistant"
        history += f"{role}: {m.content}\n"

    prompt = persona_prompt + "\n\n" + history + "Assistant:"
    text = None

    # 1) Hugging Face API
    if HF_API_KEY:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "")
        except Exception:
            logger.exception("HF streaming failed.")

    # 2) OpenAI fallback
    if text is None and OPENAI_API_KEY:
        import httpx
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages":[{"role":"user","content":prompt}], "max_tokens":512}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI streaming fallback failed.")

    if not text:
        raise HTTPException(status_code=500, detail="Generation failed")

    code_blocks = extract_code_blocks(text) if req.code_mode else []

    async def event_stream():
        chunk_size = 60
        for i in range(0, len(text), chunk_size):
            yield f"data: {text[i:i+chunk_size]}\n\n"
            await asyncio.sleep(0.04)
        if code_blocks:
            yield f"data: {json.dumps({'code_blocks': code_blocks})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Image generation ----------
@app.post("/image")
async def gen_image(req: ImageRequest):
    """
    Cloud-only image generation. Uses:
      1) Hugging Face Inference API (preferred)
      2) OpenAI DALL·E fallback
    Returns URL/base64 image string (no GPU needed)
    """
    import httpx
    import base64
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
            payload = {"inputs": req.prompt, "parameters": {"width": req.width, "height": req.height}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                # Hugging Face returns base64-encoded images in some models
                if isinstance(data, dict) and "image" in data:
                    return {"image_base64": data["image"]}
                # fallback: some APIs return list of dicts
                elif isinstance(data, list) and len(data) > 0 and "image_base64" in data[0]:
                    return {"image_base64": data[0]["image_base64"]}
        except Exception:
            logger.exception("HF image generation failed.")

    if OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            import json
            body = {"prompt": req.prompt, "size": f"{req.width}x{req.height}"}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post("https://api.openai.com/v1/images/generations", headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                img_b64 = data["data"][0]["b64_json"]
                return {"image_base64": img_b64}
        except Exception:
            logger.exception("OpenAI DALL·E generation failed.")

    raise HTTPException(status_code=503, detail="No cloud image generation backend available")

# ---------- Minimal matchmaking helpers (Redis) ----------
@app.post("/enqueue")
async def enqueue_user(payload: dict):
    r = await get_redis()
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    await r.hset("matchQueue", uid, json.dumps(payload))
    return {"status": "queued"}

@app.post("/dequeue")
async def dequeue_user(payload: dict):
    r = await get_redis()
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    await r.hdel("matchQueue", uid)
    return {"status": "removed"}

# ---------- Simple WebSocket example ----------
active_ws_connections = {}
@app.websocket("/ws/{uid}")
async def websocket_endpoint(websocket: WebSocket, uid: str):
    await websocket.accept()
    active_ws_connections[uid] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            target = data.get("to")
            message = data.get("message")
            if target and target in active_ws_connections:
                await active_ws_connections[target].send_json({"from": uid, "message": message})
            else:
                for k, conn in list(active_ws_connections.items()):
                    if k != uid:
                        try:
                            await conn.send_json({"from": uid, "message": message})
                        except Exception:
                            pass
    except WebSocketDisconnect:
        logger.info("WS disconnected %s", uid)
    finally:
        active_ws_connections.pop(uid, None)

# ---------- Shutdown ----------
@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client is not None:
        try:
            await redis_client.close()
        except Exception:
            pass
    logger.info("Server shutdown complete")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
