import os
import re
import json
import base64
import uuid
import asyncio
import logging
from io import BytesIO
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
from supabase import create_client

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HeloxAi")

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://heloxai.xyz","https://www.heloxai.xyz"],  # Tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global State for Stream Cancellation (In-memory for single-instance/Railway sticky sessions)
# For true multi-pod scaling, this should move to Redis
active_streams: Dict[str, asyncio.Task] = {}

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True

class RegenerateRequest(BaseModel):
    conversation_id: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"

# =========================
# HELPERS
# =========================
COOKIE_NAME = "session_token"

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def get_user(request: Request, response: Response) -> dict:
    """Get or create anonymous user"""
    session_token = request.cookies.get(COOKIE_NAME)
    
    if session_token:
        try:
            user_resp = await asyncio.to_thread(
                lambda: supabase.table("users")
                .select("*")
                .eq("session_token", session_token)
                .limit(1)
                .execute()
            )
            if user_resp.data:
                return user_resp.data[0]
        except Exception as e:
            logger.error(f"User lookup failed: {e}")

    # Create new
    new_token = uuid.uuid4().hex
    user_data = {
        "id": str(uuid.uuid4()),
        "email": f"anon+{uuid.uuid4().hex}@local",
        "anonymous": True,
        "session_token": new_token,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        await asyncio.to_thread(
            lambda: supabase.table("users").insert(user_data).execute()
        )
        response.set_cookie(
            key=COOKIE_NAME, value=new_token, max_age=86400*30, 
            httponly=True, secure=True, samesite="lax"
        )
        return user_data
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(500, "Auth failed")

def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# =========================
# CORE LOGIC
# =========================

async def save_message(user_id: str, conv_id: str, role: str, content: str):
    await asyncio.to_thread(
        lambda: supabase.table("messages").insert({
            "id": str(uuid.uuid4()),
            "conversation_id": conv_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    )

async def get_history(conv_id: str, limit: int = 10):
    res = await asyncio.to_thread(
        lambda: supabase.table("messages")
        .select("role, content")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return [{"role": m["role"], "content": m["content"]} for m in (res.data or [])]

async def stream_groq_chat(messages: list):
    """Streams response from Groq"""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json={"model": "llama-3.3-70b-versatile", "messages": messages, "stream": True, "max_tokens": 1024}
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]": break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content")
                        if delta: yield delta
                    except: pass

# LAZY LOADING FOR VISION (To save RAM/Startup time)
vision_model = None
def get_vision_model():
    global vision_model
    if vision_model is None:
        # Import heavy libraries ONLY when needed
        from ultralytics import YOLO
        import torch
        logger.info("Loading YOLO model...")
        vision_model = YOLO("yolov8n.pt")
        if torch.cuda.is_available():
            vision_model.to("cuda")
    return vision_model

# =========================
# ENDPOINTS
# =========================

@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    body = await req.json()
    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    
    if not prompt:
        raise HTTPException(400, "Prompt required")

    user = await get_user(req, res)
    user_id = user["id"]

    # 1. Handle Conversation
    if not conv_id:
        conv_id = str(uuid.uuid4())
        await asyncio.to_thread(
            lambda: supabase.table("conversations").insert({
                "id": conv_id, "user_id": user_id, 
                "title": prompt[:30], "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        )

    # 2. Save User Message
    await save_message(user_id, conv_id, "user", prompt)

    # 3. Intent Detection
    p_low = prompt.lower()
    
    # IMAGE GENERATION
    if any(x in p_low for x in ["image", "draw", "picture", "photo", "art"]):
        return await handle_image_generation(prompt, user_id, stream)

    # VIDEO GENERATION
    if any(x in p_low for x in ["video", "animation", "animate"]):
        return await handle_video_generation(prompt, user_id, stream)

    # CHAT (DEFAULT)
    if stream:
        async def event_gen():
            # Register stream for cancellation
            task = asyncio.current_task()
            active_streams[user_id] = task
            
            try:
                history = await get_history(conv_id)
                full_history = [{"role": "system", "content": "You are a helpful AI."}] + history
                
                full_text = ""
                async for token in stream_groq_chat(full_history):
                    # Check cancellation
                    if task and task.cancelled(): 
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                
                await save_message(user_id, conv_id, "assistant", full_text)
                yield sse({"type": "done"})
            except asyncio.CancelledError:
                yield sse({"type": "stopped"})
            finally:
                active_streams.pop(user_id, None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    else:
        # Non-streaming chat (simplified)
        history = await get_history(conv_id)
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": "llama-3.3-70b-versatile", "messages": history, "max_tokens": 1024}
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
            await save_message(user_id, conv_id, "assistant", reply)
            return {"reply": reply}

@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    """Creates a new conversation and returns the ID"""
    user = await get_user(req, res)
    cid = str(uuid.uuid4())
    await asyncio.to_thread(
        lambda: supabase.table("conversations").insert({
            "id": cid, "user_id": user["id"], 
            "title": "New Chat", "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    )
    return {"conversation_id": cid}

@app.post("/stop")
async def stop_generation(req: Request, res: Response):
    """Cancels the current stream for the user"""
    user = await get_user(req, res)
    user_id = user["id"]
    
    task = active_streams.get(user_id)
    if task and not task.done():
        task.cancel()
        active_streams.pop(user_id, None)
        return {"status": "stopped"}
    return {"status": "no_active_stream"}

@app.post("/regenerate")
async def regenerate(req: Request, res: Response):
    """Regenerates the last assistant response"""
    body = await req.json()
    conv_id = body.get("conversation_id")
    
    user = await get_user(req, res)
    user_id = user["id"]

    if not conv_id:
        raise HTTPException(400, "conversation_id required")

    # 1. Get last user message
    msgs = await asyncio.to_thread(
        lambda: supabase.table("messages")
        .select("*")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    
    if not msgs.data:
        raise HTTPException(404, "No messages found")

    # Find last user message
    last_user_msg = None
    for m in msgs.data:
        if m["role"] == "user":
            last_user_msg = m
            break
    
    if not last_user_msg:
        raise HTTPException(400, "No user message to regenerate from")

    # 2. Delete assistant messages after that user message
    await asyncio.to_thread(
        lambda: supabase.table("messages")
        .delete()
        .gt("created_at", last_user_msg["created_at"])
        .eq("role", "assistant")
        .eq("conversation_id", conv_id)
        .execute()
    )

    # 3. Re-run stream logic (Simplified copy of /ask/universal logic)
    async def event_gen():
        task = asyncio.current_task()
        active_streams[user_id] = task
        try:
            history = await get_history(conv_id)
            full_text = ""
            async for token in stream_groq_chat(history):
                if task and task.cancelled(): break
                full_text += token
                yield sse({"type": "token", "text": token})
            
            await save_message(user_id, conv_id, "assistant", full_text)
            yield sse({"type": "done"})
        except asyncio.CancelledError:
            yield sse({"type": "stopped"})
        finally:
            active_streams.pop(user_id, None)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/chats")
async def list_chats(req: Request, res: Response):
    """List all user chats"""
    user = await get_user(req, res)
    res = await asyncio.to_thread(
        lambda: supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True)
        .execute()
    )
    return {"chats": res.data or []}

# =========================
# MEDIA ENDPOINTS
# =========================

@app.post("/tts")
async def text_to_speech(req: Request):
    """Text to Speech"""
    data = await req.json()
    text = data.get("text", "")
    voice = data.get("voice", "alloy")
    
    if not text: raise HTTPException(400, "text required")
    if not OPENAI_API_KEY: raise HTTPException(500, "OpenAI Key missing")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers=get_openai_headers(),
            json={"model": "tts-1", "voice": voice, "input": text}
        )
        r.raise_for_status()
        return Response(content=r.content, media_type="audio/mpeg")

@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "echo", "name": "Echo"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"}
        ]
    }

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Speech to Text (Whisper)"""
    if not OPENAI_API_KEY: raise HTTPException(500, "OpenAI Key missing")
    
    content = await file.read()
    
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (file.filename, content, file.content_type)}
        data = {"model": "whisper-1"}
        r = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files, data=data
        )
        r.raise_for_status()
        return r.json()

# =========================
# INTERNAL HANDLERS
# =========================

async def handle_image_generation(
    prompt: str,
    user_id: str,
    stream: bool,
    style: str = None,
    size: str = "1024x1024",
    num_images: int = 1
):
    if not OPENAI_API_KEY:
        if stream:
            async def gen(): yield sse({"type": "error", "message": "No API Key"})
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"error": "No API Key"}

    if not prompt:
        raise ValueError("Prompt is required")

    STYLES = {
        "realistic": "ultra realistic, 4k, detailed",
        "cartoon": "cartoon style, colorful",
        "anime": "anime style, studio ghibli",
        "cinematic": "cinematic lighting, dramatic",
        "cyberpunk": "cyberpunk, neon futuristic",
    }

    if style in STYLES:
        prompt = f"{prompt}, {STYLES[style]}"

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers=get_openai_headers(),
            json={
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "n": num_images
            }
        )

        if r.status_code != 200:
            print("IMAGE ERROR:", r.status_code, r.text)

        r.raise_for_status()
        data = r.json()

    images = []

    for item in data["data"]:
        b64 = item["b64_json"]
        img_bytes = base64.b64decode(b64)

        fname = f"{uuid.uuid4().hex}.png"
        path = f"public/{fname}"

        try:
            await asyncio.to_thread(
                lambda: supabase.storage.from_("ai-images").upload(
                    path, img_bytes, {"content-type": "image/png"}
                )
            )
            url = f"{SUPABASE_URL}/storage/v1/object/public/ai-images/{path}"
        except:
            url = "data:image/png;base64," + b64

        images.append({"url": url})

    if stream:
        async def gen():
            yield sse({"type": "images", "images": images})
            yield sse({"type": "done"})
        return StreamingResponse(gen(), media_type="text/event-stream")

    return {"images": images}
    
async def handle_video_generation(prompt: str, user_id: str, stream: bool):
    """Generate Video (Replicate or Placeholder)"""
    if not REPLICATE_API_TOKEN:
        if stream:
            async def gen(): yield sse({"type": "error", "message": "Replicate Key missing"})
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"error": "Replicate Key missing"}

    # This is a placeholder for Replicate integration
    # Since Replicate requires a library that might be heavy, we do a basic HTTP call here or mock it
    # For a full implementation, you would poll the Replicate API.
    
    if stream:
        async def gen(): 
            yield sse({"type": "starting"})
            # Mock delay
            await asyncio.sleep(2)
            yield sse({"type": "error", "message": "Video generation requires specific Replicate polling logic implementation."})
        return StreamingResponse(gen(), media_type="text/event-stream")
    
    return {"error": "Video generation not implemented in this clean version"}

# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {"status": "running", "service": "HeloxAi Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
