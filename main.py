from fastapi import Form
from email.message import EmailMessage
import smtplib
import re
import os
import asyncio
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import uuid

# Optional libs; import lazily to avoid heavy startup cost
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

# Optional HF / transformers imports (lazily used)
_transformers_available = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
except Exception:
    _transformers_available = False

# Optional diffusers
_diffusers_available = True
try:
    from diffusers import StableDiffusionPipeline
    import torch
except Exception:
    _diffusers_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proai")

# ---------- Config ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "gpt2")
HF_API_KEY = os.getenv("HF_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
ENABLE_IMAGE_GEN = os.getenv("ENABLE_IMAGE_GEN", "false").lower() == "true"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "CompVis/stable-diffusion-v1-4")
PORT = int(os.getenv("PORT", "8000"))
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")  # CSV of origins

# ---------- FastAPI app ----------
app = FastAPI(title="ProAI - main")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Redis client (async) ----------
redis_client = None

async def get_redis():
    global redis_client
    if redis_client is None:
        if aioredis is None:
            raise RuntimeError("redis.asyncio not installed - please install redis>=4.2")
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        # Attempt a ping
        try:
            await redis_client.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.warning("⚠️ Redis connect failed: %s", e)
    return redis_client

# ---------- Model holders (lazy-loaded) ----------
_text_tokenizer = None
_text_model = None
_image_pipe = None

async def load_text_model_if_needed():
    global _text_tokenizer, _text_model
    if _text_model is not None:
        return
    if USE_LOCAL_MODEL:
        if not _transformers_available:
            logger.error("Transformers not available to load local model.")
            return
        logger.info("Loading local text model: %s", LOCAL_MODEL_NAME)
        _text_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, use_fast=True)
        _text_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME, device_map="auto")
        logger.info("✅ Local text model loaded")
    else:
        logger.info("Local model disabled; will use remote inference APIs (HF/OpenAI) if configured.")

async def load_image_model_if_needed():
    global _image_pipe
    if not ENABLE_IMAGE_GEN:
        return
    if _image_pipe is not None:
        return
    if not _diffusers_available:
        logger.error("Diffusers not available; enable IMAGE via external API or install diffusers/torch.")
        return
    logger.info("Loading image pipeline (may require GPU)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _image_pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL).to(device)
    logger.info("✅ Image model ready on %s", device)

# ---------- Pydantic request/response models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    persona: Optional[str] = "default"
    stream: Optional[bool] = False
    code_mode: Optional[bool] = False  # new: focus on code output

class ImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512

# ---------- Helper utilities ----------
def safe_filename(prefix="out", ext="png"):
    return f"{prefix}-{uuid.uuid4().hex}.{ext}"

async def generate_text_local(prompt: str, max_new_tokens: int = 256):
    """
    Generate text using local model (transformers).
    """
    if _text_model is None or _text_tokenizer is None:
        raise RuntimeError("Local text model is not loaded")
    inputs = _text_tokenizer(prompt, return_tensors="pt", truncation=True).to(_text_model.device)
    out = _text_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95, top_k=50, temperature=0.8)
    text = _text_tokenizer.decode(out[0], skip_special_tokens=True)
    # remove prompt prefix if model echoes it
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

async def generate_text_remote_hf(prompt: str):
    """
    Call Hugging Face Inference API synchronously (simple).
    Requires HF_API_KEY to be set.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not configured")
    import httpx
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError("HF inference error: " + data["error"])
        # HF returns a list with generated_text
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        return str(data)

def extract_code_blocks(text):
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    blocks = []
    for lang, code in matches:
        blocks.append({"language": lang or "text", "code": code.strip()})
    return blocks

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "shaynengunga15@gmail.com")

@app.post("/notify_admin")
async def notify_admin(reason: str = Form(...)):
    msg = EmailMessage()
    msg.set_content(f"User reported an AI issue: {reason}")
    msg["Subject"] = "AI Service Alert"
    msg["From"] = ADMIN_EMAIL
    msg["To"] = ADMIN_EMAIL

    try:
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
        return {"status": "notification sent"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.on_event("startup")
async def startup_event():
    # Connect Redis
    try:
        await get_redis()
    except Exception as e:
        logger.warning("Redis not available at startup: %s", e)

    # Optionally pre-load small models in background:
    if USE_LOCAL_MODEL:
        # load in background
        asyncio.create_task(load_text_model_if_needed())
    if ENABLE_IMAGE_GEN:
        asyncio.create_task(load_image_model_if_needed())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Single-shot chat endpoint.
    Returns normal text and optionally code blocks if code_mode=True.
    """
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant.",
    }
    persona_prompt = persona_map.get(req.persona, persona_map["default"])

    # Build prompt with chat history
    history = ""
    for m in req.messages:
        if m.role == "user":
            history += f"User: {m.content}\n"
        elif m.role == "assistant":
            history += f"Assistant: {m.content}\n"
    prompt = persona_prompt + "\n\n" + history + "Assistant:"

    out = None

    # 1) Local model
    if USE_LOCAL_MODEL and _text_model is not None:
        try:
            out = await generate_text_local(prompt)
        except Exception:
            logger.exception("Local text generation failed.")

    # 2) Hugging Face API
    if out is None and HF_API_KEY:
        try:
            out = await generate_text_remote_hf(prompt)
        except Exception:
            logger.exception("HF inference failed.")

    # 3) OpenAI fallback
    if out is None and OPENAI_API_KEY:
        try:
            import httpx
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI call failed.")

    if not out:
        raise HTTPException(status_code=503, detail="No text generation backend available")

    # Extract code blocks if code_mode is enabled
    response = {"response": out}
    if req.code_mode:
        code_blocks = extract_code_blocks(out)
        response["code_blocks"] = code_blocks

    return response
    
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint via Server-Sent Events (SSE).
    Supports code_mode to extract code blocks from generated text.
    """
    persona_map = {"default": "You are a helpful assistant."}
    persona_prompt = persona_map.get(req.persona, persona_map["default"])

    # Build prompt with chat history
    history = ""
    for m in req.messages:
        if m.role == "user":
            history += f"User: {m.content}\n"
        elif m.role == "assistant":
            history += f"Assistant: {m.content}\n"
    prompt = persona_prompt + "\n\n" + history + "Assistant:"

    # Generate text
    text = None
    try:
        if USE_LOCAL_MODEL and _text_model is not None:
            text = await generate_text_local(prompt)
    except Exception:
        logger.exception("Local generation error (stream).")

    if text is None:
        if HF_API_KEY:
            text = await generate_text_remote_hf(prompt)
        elif OPENAI_API_KEY:
            import httpx
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages":[{"role":"user","content":prompt}], "max_tokens":256}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]

    if not text:
        raise HTTPException(status_code=500, detail="Generation failed")

    # Extract code blocks if code_mode enabled
    code_blocks = extract_code_blocks(text) if req.code_mode else []

    async def event_stream():
        chunk_size = 60
        # Stream the text in chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.05)
        # Send code blocks at the end if any
        if code_blocks:
            import json
            yield f"data: {json.dumps({'code_blocks': code_blocks})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    
@app.post("/image", response_class=JSONResponse)
async def gen_image(req: ImageRequest, background: BackgroundTasks):
    """
    Generate an image. If ENABLE_IMAGE_GEN and diffusers available, generate locally.
    Otherwise throw an error (or integrate remote HF API).
    Returns { "url": "/generated/..." } (file saved to tmp or static folder).
    """
    if not ENABLE_IMAGE_GEN:
        raise HTTPException(status_code=503, detail="Image generation disabled on this service")

    await load_image_model_if_needed()
    if _image_pipe is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    # generate image synchronously (this can block - consider backgrounding)
    logger.info("Generating image for prompt: %s", req.prompt)
    img = _image_pipe(req.prompt, height=req.height or 1024, width=req.width or 1024).images[0]

    tmp_dir = tempfile.gettempdir()
    fname = safe_filename("img", "png")
    path = os.path.join(tmp_dir, fname)
    img.save(path)

    # optionally move to static folder for serving
    return {"path": f"/static_temp/{fname}", "tmp_path": path}

# Serve generated images from /static_temp (don't rely on tmp to persist in production)
STATIC_TEMP_DIR = tempfile.gettempdir()
app.mount("/static_temp", StaticFiles(directory=STATIC_TEMP_DIR), name="static_temp")

# ---------- Matchmaking helpers (example using Redis hashes) ----------
@app.post("/enqueue")
async def enqueue_user(payload: dict):
    """
    Expected payload includes uid and prefs, e.g. {"uid": "user1", "gender":"any","location":"any","interests": [...]}
    """
    r = await get_redis()
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    await r.hset("matchQueue", uid, json_serialize(payload))
    return {"status": "queued"}

@app.post("/dequeue")
async def dequeue_user(payload: dict):
    r = await get_redis()
    uid = payload.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    await r.hdel("matchQueue", uid)
    return {"status": "removed"}

async def json_serialize(obj):
    import json
    return json.dumps(obj)

# ---------- Simple WebSocket chat example ----------
active_ws_connections = {}

@app.websocket("/ws/{uid}")
async def websocket_endpoint(websocket: WebSocket, uid: str):
    await websocket.accept()
    active_ws_connections[uid] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            # echo or route message - simplistic example
            target = data.get("to")
            message = data.get("message")
            if target and target in active_ws_connections:
                await active_ws_connections[target].send_json({"from": uid, "message": message})
            else:
                # broadcast to all (example)
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

# ---------- Shutdown gracefully ----------
@app.on_event("shutdown")
async def shutdown_event():
    global redis_client
    if redis_client is not None:
        try:
            await redis_client.close()
        except Exception:
            pass
    logger.info("Server shutdown complete")

# ---------- CLI (uvicorn) ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
