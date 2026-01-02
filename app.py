# app.py — ZyNara1 AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import utils
import PIL
import json
from utils import safe_system_prompt  # adjust path as needed
import uuid
import sqlite3
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
import asyncio
import base64
import time
import logging
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List
from ultralytics import YOLO
import cv2

YOLO_OBJECTS = None
YOLO_FACES = None
YOLO_DEVICE = "cpu"

def get_yolo_objects():
    global YOLO_OBJECTS
    if YOLO_OBJECTS is None:
        YOLO_OBJECTS = YOLO("yolov8n.pt")
        YOLO_OBJECTS.to(YOLO_DEVICE)
    return YOLO_OBJECTS

def get_yolo_faces():
    global YOLO_FACES
    if YOLO_FACES is None:
        YOLO_FACES = YOLO("yolov8n-face.pt")
        YOLO_FACES.to(YOLO_DEVICE)
    return YOLO_FACES

import httpx
from fastapi import FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from supabase import create_client
await save_message(user_id, "user", prompt)

# ---------- ENV KEYS ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")
    
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

groq_client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("zynara-server")

app = FastAPI(
    title="ZyNaraAI1.0 Multimodal Server",
    redirect_slashes=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SSE HELPER ----------------
def sse(obj: dict) -> str:
    """
    Formats a dict as a Server-Sent Event (SSE) message.
    """
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ---------- ENV KEYS ----------
# strip GROQ API key in case it contains whitespace/newlines
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is not None:
    GROQ_API_KEY = GROQ_API_KEY.strip()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL")
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

# Quick log so you can confirm key presence without printing the key itself
logger.info(f"GROQ key present: {bool(GROQ_API_KEY)}")

# -------------------
# Models
# -------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")

# TTS/STT are handled via ElevenLabs now
TTS_MODEL = None
STT_MODEL = None

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldYLocks",
    "age": 17,
    "country": "England",
    "projects": ["MZ", "LS", "SX", "CB"],
    "socials": { "discord":"@nexisphere123_89431", "twitter":"@NexiSphere"},
    "bio": "Created by GoldBoy (17, England). Projects: MZ, LS, SX, CB. Socials: Discord @nexisphere123_89431 Twitter @NexiSphere."
}

JUDGE0_LANGUAGES = {
    # --- C / C++ ---
    "c": 50,
    "c_clang": 49,
    "cpp": 54,
    "cpp_clang": 53,

    # --- Java ---
    "java": 62,

    # --- Python ---
    "python": 71,
    "python2": 70,
    "micropython": 79,

    # --- JavaScript / TS ---
    "javascript": 63,
    "nodejs": 63,
    "typescript": 74,

    # --- Go ---
    "go": 60,

    # --- Rust ---
    "rust": 73,

    # --- C# / .NET ---
    "csharp": 51,
    "fsharp": 87,
    "dotnet": 51,

    # --- PHP ---
    "php": 68,

    # --- Ruby ---
    "ruby": 72,

    # --- Swift ---
    "swift": 83,

    # --- Kotlin ---
    "kotlin": 78,

    # --- Scala ---
    "scala": 81,

    # --- Objective-C ---
    "objective_c": 52,

    # --- Bash / Shell ---
    "bash": 46,
    "sh": 46,

    # --- PowerShell ---
    "powershell": 88,

    # --- Perl ---
    "perl": 85,

    # --- Lua ---
    "lua": 64,

    # --- R ---
    "r": 80,

    # --- Dart ---
    "dart": 75,

    # --- Julia ---
    "julia": 84,

    # --- Haskell ---
    "haskell": 61,

    # --- Elixir ---
    "elixir": 57,

    # --- Erlang ---
    "erlang": 58,

    # --- OCaml ---
    "ocaml": 65,

    # --- Crystal ---
    "crystal": 76,

    # --- Nim ---
    "nim": 77,

    # --- Zig ---
    "zig": 86,

    # --- Assembly ---
    "assembly": 45,

    # --- COBOL ---
    "cobol": 55,

    # --- Fortran ---
    "fortran": 59,

    # --- Prolog ---
    "prolog": 69,

    # --- Scheme ---
    "scheme": 82,

    # --- Common Lisp ---
    "lisp": 66,

    # --- Brainf*ck ---
    "brainfuck": 47,

    # --- V ---
    "vlang": 91,

    # --- Groovy ---
    "groovy": 56,

    # --- Hack ---
    "hack": 67,

    # --- Pascal ---
    "pascal": 67,

    # --- Scratch ---
    "scratch": 92,

    # --- Solidity ---
    "solidity": 94,

    # --- SQL ---
    "sql": 82,

    # --- Text / Plain ---
    "plain_text": 43,
    "text": 43,
}

JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
JUDGE0_KEY = os.getenv("JUDGE0_API_KEY")

if not JUDGE0_KEY:
    logger.warning("⚠️ Judge0 key not set — code execution disabled")

if not JUDGE0_KEY:
    logger.warning("Code execution disabled (missing Judge0 API key)")

async def run_code_judge0(code: str, language_id: int):
    payload = {
        "language_id": language_id,
        "source_code": code
    }

    headers = {
        "X-RapidAPI-Key": JUDGE0_KEY,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30) as client:
        submit = await client.post(
            "https://judge0-ce.p.rapidapi.com/submissions?wait=false",
            json=payload,
            headers=headers
        )

        if submit.status_code == 403:
            return {
                "error": "Judge0 execution blocked (403). Check RapidAPI key or plan."
            }

        submit.raise_for_status()
        return submit.json()

# ---------- Dynamic, user-focused system prompt ----------
def get_system_prompt(user_message: Optional[str] = None) -> str:
    base = "You are ZynaraAI1.0: helpful, concise, friendly, and focus entirely on what the user asks. Do not reference your creator or yourself unless explicitly asked."
    if user_message:
        base += f" The user said: \"{user_message}\". Tailor your response to this."
    return base

def build_contextual_prompt(user_id: str, message: str) -> str:
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM memory WHERE user_id=? ORDER BY updated_at DESC LIMIT 5", (user_id,))
    rows = cur.fetchall()
    conn.close()
    context = "\n".join(f"{k}: {v}" for k, v in rows)
    return f"You are ZyNaraAI1.0 : helpful, concise, friendly. Focus on exactly what the user wants.\nUser context:\n{context}\nUser message: {message}"


# ---------- Image helpers ----------
def unique_filename(ext="png"):
    return f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"

def upload_to_supabase(
    file_bytes: bytes,
    filename: str,
    bucket: str = "ai-images",
    content_type: str = "application/octet-stream"
) -> str:
    """
    Upload a file (image or video) to Supabase storage.
    """
    supabase.storage.from_(bucket).upload(
        path=filename,
        file=file_bytes,
        file_options={"content-type": content_type}
    )
    return filename

# Helper wrappers for missing functions
def upload_image_to_supabase(image_bytes: bytes, filename: str):
    upload = supabase.storage.from_("ai-images").upload(
        filename,
        image_bytes,
        {"content-type": "image/png"}
    )

    if upload.get("error"):
        raise Exception(upload["error"]["message"])

    return upload

def save_image_record(user_id, prompt, path, is_nsfw):
    try:
        supabase.table("images").insert({
            "user_id": user_id,
            "prompt": prompt,
            "image_path": path,
            "is_nsfw": is_nsfw
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save image record: {e}")

async def get_user_id_from_cookie(request: Request, response: Response) -> str:
    return await get_or_create_user(request, response)

async def nsfw_check(prompt: str) -> bool:
    if not OPENAI_API_KEY: return False
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.openai.com/v1/moderations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "omni-moderation-latest", "input": prompt}
        )
        r.raise_for_status()
        result = r.json()["results"][0]
        return result["flagged"]
        
# ---------- USER / COOKIE ----------
async def get_or_create_user(req: Request, res: Response) -> str:
    user_id = req.cookies.get("uid")

    if user_id:
        return user_id

    user_id = str(uuid.uuid4()) 
    try:
        supabase.table("users").insert({"id": user_id}).execute()
    except Exception:
        pass # Ignore if exists

    res.set_cookie(
        key="uid",
        value=user_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 365
    )
    return user_id

async def save_message(user_id: str, role: str, content: str):
    await supabase.table("conversations").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()


async def load_history(user_id: str, limit: int = 20):
    resp = await supabase.table("conversations") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    
    messages = [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(resp.data)
    ]
    return messages


# ----------------------------------
# MEMORY LOADER
# ----------------------------------
def load_memory(conversation_id: str, limit: int = 20):
    res = supabase.table("messages") \
        .select("role,content") \
        .eq("conversation_id", conversation_id) \
        .order("created_at") \
        .limit(limit) \
        .execute()

    return res.data or []

# ----------------------------------
# NEW CHAT
# ----------------------------------

@app.post("/chat/new")
async def new_chat(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)
    cid = str(uuid.uuid4())

    supabase.table("conversations").insert({
        "id": cid,
        "user_id": user_id,
        "title": "New Chat"
    }).execute()

    return {"conversation_id": cid}

@app.post("/chat/{conversation_id}")
async def send_message(conversation_id: str, req: Request, res: Response):
    user_id = await get_or_create_user(req, res)
    body = await req.json()
    text = body.get("message")

    if not text:
        raise HTTPException(400, "message required")

    supabase.table("messages").insert({
        "id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "role": "user",
        "content": text
    }).execute()

    messages = (
        supabase.table("messages")
        .select("role,content")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
        .data
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=groq_headers(),
            json=payload
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]

    supabase.table("messages").insert({
        "id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": reply
    }).execute()

    return {"reply": reply}

# ----------------------------------
# LIST CHATS
# ----------------------------------
@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    res = supabase.table("conversations") \
        .select("*") \
        .eq("user_id", user_id) \
        .eq("archived", False) \
        .order("pinned", desc=True) \
        .order("created_at", desc=True) \
        .execute()

    return res.data or []

# ----------------------------------
# SEARCH CHATS
# ----------------------------------
@app.get("/chats/search")
async def search_chats(q: str, req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    res = supabase.table("conversations") \
        .select("id,title") \
        .eq("user_id", user_id) \
        .ilike("title", f"%{q}%") \
        .execute()

    return res.data or []

# ----------------------------------
# PIN / ARCHIVE
# ----------------------------------
@app.post("/chat/{id}/pin")
async def pin_chat(id: str):
    supabase.table("conversations").update({"pinned": True}).eq("id", id).execute()
    return {"status": "pinned"}

@app.post("/chat/{id}/archive")
async def archive_chat(id: str):
    supabase.table("conversations").update({"archived": True}).eq("id", id).execute()
    return {"status": "archived"}

# ----------------------------------
# FOLDER
# ----------------------------------
@app.post("/chat/{id}/folder")
async def move_folder(id: str, folder: Optional[str] = None):
    supabase.table("conversations").update({"folder": folder}).eq("id", id).execute()
    return {"status": "moved"}

# ----------------------------------
# SHARE CHAT
# ----------------------------------
@app.post("/chat/{id}/share")
async def share_chat(id: str):
    token = uuid.uuid4().hex

    supabase.table("conversations").update({
        "share_token": token,
        "is_public": True
    }).eq("id", id).execute()

    return {"share_url": f"/share/{token}"}

# ----------------------------------
# VIEW SHARED CHAT (READ ONLY)
# ----------------------------------
@app.get("/share/{token}")
async def view_shared_chat(token: str):
    convo = supabase.table("conversations") \
        .select("id,title") \
        .eq("share_token", token) \
        .eq("is_public", True) \
        .single() \
        .execute()

    if not convo.data:
        raise HTTPException(404)

    messages = supabase.table("messages") \
        .select("role,content,created_at") \
        .eq("conversation_id", convo.data["id"]) \
        .order("created_at") \
        .execute()

    return {
        "title": convo.data["title"],
        "messages": messages.data
    }

#------duckduckgo

async def duckduckgo_search(q: str):
    """
    Use DuckDuckGo Instant Answer API (no API key required).
    Returns a simple structured result with abstract, answer and a list of related topics.
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html":1, "skip_disambig": 1}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        results = []
        # RelatedTopics can contain nested topics or single items; handle both.
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict):
                # Some items are like {"Text": "...", "FirstURL": "..."}
                if item.get("Text"):
                    results.append({"title": item.get("Text"), "url": item.get("FirstURL")})
                # Some are category blocks with "Topics" list
                elif item.get("Topics"):
                    for t in item.get("Topics", []):
                        if t.get("Text"):
                            results.append({"title": t.get("Text"), "url": t.get("FirstURL")})
        # Limit results to a reasonable number
        results = results[:10]

        return {
            "query": q,
            "abstract": data.get("AbstractText"),
            "answer": data.get("Answer"),
            "results": results
            }

# ---------- Helper: centralize Groq headers ----------
def get_groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

async def tts_stream_helper(text: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "tts-1", 
        "voice": "alloy",
        "input": text
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload
        )
        r.raise_for_status()

    b64 = base64.b64encode(r.content).decode()
    yield {"type": "tts_done", "audio": b64}
    
# ---------- Prompt enhancer ----------
async def enhance_prompt_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return prompt
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = get_groq_headers()
        system = "Rewrite the user's short prompt into a detailed, professional SDXL-style art prompt. Be concise but specific. Avoid explicit sexual or illegal content."
        body = {"model": CHAT_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}], "temperature":0.6,"max_tokens":300}
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code != 200:
                logger.warning("Groq enhance_prompt_with_groq failed: status=%s text=%s", r.status_code, (r.text[:100] + '...') if r.text else "")
                r.raise_for_status()
            jr = r.json()
            content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or prompt
    except Exception:
        logger.exception("Prompt enhancer failed")
    return prompt


async def generate_video_internal(prompt: str, samples: int = 1, user_id: str = "anonymous") -> dict:
    """
    Generate video (stub). Stores video files in Supabase storage like images.
    Returns a list of signed URLs.
    """
    urls = []

    for i in range(samples):
        # For now, we create a placeholder video file
        placeholder_content = b"This is a placeholder video for prompt: " + prompt.encode('utf-8')
        filename = f"{user_id}/video-{int(time.time())}-{uuid.uuid4().hex[:8]}.mp4"

        # Upload to Supabase
        try:
            supabase.storage.from_("ai-videos").upload(
                path=filename,
                file=placeholder_content,
                file_options={"content-type": "video/mp4"}
            )
            # Get signed URL
            signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
            urls.append(signed["signedURL"])
        except Exception as e:
            logger.error(f"Video upload failed: {e}")

    return {"provider": "stub", "videos": urls}

# ---------- Intent Detection ----------
def detect_intent(prompt: str) -> str:
    p = prompt.lower()

    # Image requests
    if any(w in p for w in ["image of", "draw", "picture of", "generate image", "make me an image", "photo of", "art of"]):
        return "image"

    # Video generation (placeholder – uses your future DreamWAN integration)
    if any(w in p for w in ["video of", "make a video", "animation of", "clip of"]):
        return "video"

    # TTS
    if any(w in p for w in ["say this", "speak", "tts", "read this", "read aloud"]):
        return "tts"

    # Code generation
    if any(w in p for w in ["write code", "generate code", "python code", "javascript code", "fix this code"]):
        return "code"

    # Search / info lookup
    if any(w in p for w in ["search", "look up", "find info", "who is", "what is"]):
        return "search"

    # Default → Chat model
    return "chat"

# =========================
# STREAM HELPERS FOR UNIVERSAL ENDPOINT
# =========================
async def universal_chat_stream(user_id: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = get_groq_headers()

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
            {"role": "user", "content": prompt}
           ],
            "max_tokens": 1024
        
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield {"type": "chat_token", "text": delta}
                except Exception:
                    continue
            
# ---------- Prompt analysis ----------
def analyze_prompt(prompt: str):
    p = prompt.lower()
    settings = {"model": "stable-diffusion-xl-v1","width":1024,"height":1024,"steps":30,"cfg_scale":7,"samples":1,"negative_prompt":"nsfw, nudity, watermark, lowres, text, logo"}
    if any(w in p for w in ("wallpaper","background","poster")):
        settings["width"], settings["height"] = 1920, 1080
    if any(w in p for w in ("landscape","city","wide","panorama")):
        settings["width"], settings["height"] = 1280, 720
    for token in p.split():
        if token.isnumeric():
            n = int(token)
            if 1 <= n <= 6:
                settings["samples"] = n
    return settings

async def image_stream_helper(prompt: str, samples: int):
    try:
        result = await _generate_image_core(prompt, samples, "anonymous", return_base64=False)
        yield {
            "type": "images",
            "provider": result["provider"],
            "images": result["images"]
        }
    except HTTPException as e:
        yield {
            "type": "image_error",
            "error": e.detail
        }
    except Exception as e:
        logger.exception("Unhandled image stream error")
        yield {
            "type": "image_error",
            "error": "Unexpected image error"
        }

# ---------- Streaming chat ----------
async def chat_stream_helper(user_id: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
    "model": CHAT_MODEL,
    "stream": True,
    "messages": [
        {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1024
}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            url,
            headers=get_groq_headers(),
            json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield {"type": "token", "text": delta}
                except Exception:
                    continue
                    
# Tracks currently active SSE/streaming tasks per user
active_streams: Dict[str, asyncio.Task] = {}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }
    
@app.get("/")
async def root():
    return {"message": "Billy AI Backend is Running ✔"}
    
@app.post("/chat/stream")
async def chat_stream(req: Request, res: Response, tts: bool = False, samples: int = 1):
    """
    Unified streaming endpoint:
    - Image streaming (if prompt implies image)
    - Chat streaming (Groq)
    - Optional TTS
    Cookie-based identity + safe cancellation
    """
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)

    async def event_generator():
        # ✅ REGISTER STREAM IN SUPABASE
        stream_id = str(uuid.uuid4())
        supabase.table("active_streams").upsert({
            "user_id": user_id,
            "stream_id": stream_id,
            "started_at": datetime.utcnow().isoformat()
        }).execute()

        try:
            # ---------- 1️⃣ IMAGE STREAM ----------
            if any(w in prompt.lower() for w in ("image", "draw", "illustrate", "painting", "art", "picture")):
                try:
                    yield sse({"status": "image_start", "message": "Starting image generation"})

                    img_payload = {"prompt": prompt, "samples": samples, "base64": False}

                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "http://127.0.0.1:8000/image/stream",
                            json=img_payload
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if line.strip():
                                    yield line + "\n\n"

                    yield sse({"status": "image_done", "message": "Image generation complete"})

                except Exception:
                    logger.exception("Image streaming failed")
                    yield sse({"status": "image_error", "message": "Image generation failed"})

            # ---------- 2️⃣ CHAT STREAM ----------
            try:
                messages = [
                    {"role": "system", "content": safe_system_prompt(build_contextual_prompt(user_id, prompt))},
                    {"role": "user", "content": prompt}
                ]

                payload = {
                    "model": CHAT_MODEL,
                    "stream": True,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }

                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=get_groq_headers(),
                        json=payload
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue

                            # Check if stream was cancelled
                            active = supabase.table("active_streams").select("stream_id").eq("user_id", user_id).execute().data
                            if not active:
                                logger.info(f"Stream cancelled by user {user_id}")
                                break

                            data = line[len("data:"):].strip()
                            if data == "[DONE]":
                                break

                            delta = json.loads(data)["choices"][0]["delta"].get("content")
                            if delta:
                                yield sse({"token": delta})

            except Exception:
                logger.exception("Chat streaming failed")
                yield sse({"status": "chat_error", "message": "Chat stream failed"})

            # ---------- 3️⃣ TTS ----------
            if tts:
                try:
                    tts_payload = {
                        "model": "tts-1",
                        "voice": "alloy",
                        "input": prompt,
                        "format": "mp3"
                    }

                    headers = {
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    }

                    audio_buffer = bytearray()
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "https://api.openai.com/v1/audio/speech",
                            headers=headers,
                            json=tts_payload
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                if chunk:
                                    audio_buffer.extend(chunk)

                    b64_audio = base64.b64encode(audio_buffer).decode("utf-8")
                    yield sse({"status": "tts_done", "audio": b64_audio})

                except Exception:
                    logger.exception("TTS failed")
                    yield sse({"status": "tts_error", "message": "TTS failed"})

            yield sse({"status": "done"})

        except asyncio.CancelledError:
            logger.info(f"Chat stream cancelled for user {user_id}")
            yield sse({"status": "stopped"})
            raise

        except Exception as e:
            logger.exception("Universal stream crashed")
            yield sse({"status": "fatal_error", "error": str(e)})

        finally:
            # ✅ CLEANUP ACTIVE STREAM IN SUPABASE
            supabase.table("active_streams").delete().eq("user_id", user_id).execute()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
# ---------- Chat endpoint ----------
@app.post("/chat")
async def chat_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt","")
    user_id = body.get("user_id", "anonymous")
    if not prompt:
        raise HTTPException(400,"prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400,"no groq key")
    payload = {"model":CHAT_MODEL,"messages":[{"role":"system","content":build_contextual_prompt(user_id, prompt)},{"role":"user","content":prompt}]}

    headers = get_groq_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if r.status_code != 200:
                logger.warning("Groq /chat returned status=%s text=%s", r.status_code, (r.text[:500] + '...') if r.text else "")
                r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            logger.exception("Groq HTTP error on /chat: %s", getattr(exc.response, "text", "no-response-text"))
            raise HTTPException(status_code=exc.response.status_code if exc.response is not None else 500, detail=f"Groq error: {exc.response.text[:400] if exc.response is not None else str(exc)}")
        except Exception:
            logger.exception("Groq /chat request failed")
            raise HTTPException(500, "groq_request_failed")

# =========================================================
# 🚀 UNIVERSAL MULTIMODAL ENDPOINT — /ask/universal
# =========================================================

# ---------- Core Image Logic (Refactored) ----------
async def _generate_image_core(
    prompt: str,
    samples: int,
    user_id: str,
    return_base64: bool = False
):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    provider_used = "openai"
    urls = []

    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,  # DALL·E 3 supports only 1 image
        "size": "1024x1024",
        "response_format": "b64_json"
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # ---------- CALL OPENAI ----------
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/images/generations",
                json=payload,
                headers=headers
            )
            r.raise_for_status()
            result = r.json()

    except Exception:
        logger.exception("OpenAI image API call failed")
        raise HTTPException(500, "Image generation provider error")

    # ---------- VALIDATE ----------
    if not result or not result.get("data"):
        logger.error("OpenAI returned empty image response: %s", result)
        raise HTTPException(500, "Image generation failed")

    # ---------- PROCESS IMAGES ----------
    for img in result["data"]:
        try:
            b64 = img.get("b64_json")
            if not b64:
                continue

            image_bytes = base64.b64decode(b64)
            filename = f"{user_id}/{uuid.uuid4().hex}.png"

            upload = supabase.storage.from_("ai-images").upload(
                path=filename,
                file=image_bytes,
                file_options={
                    "content-type": "image/png",
                    "upsert": True
                }
            )

            if isinstance(upload, dict) and upload.get("error"):
                raise RuntimeError(upload["error"])

            signed = supabase.storage.from_("ai-images").create_signed_url(
                filename, 60 * 60
            )

            if signed and signed.get("signedURL"):
                urls.append(signed["signedURL"])

        except Exception:
            logger.exception("Failed processing or uploading image")
            continue

    if not urls:
        raise HTTPException(500, "No images generated")

    cache_result(prompt, provider_used, {"images": urls})

    return {
        "provider": provider_used,
        "images": urls
    }


async def image_gen_internal(prompt: str, samples: int = 1):
    """Helper for streaming /ask/universal."""
    result = await _generate_image_core(prompt, samples, "anonymous", return_base64=False)

async def stream_images(prompt: str, samples: int):
    try:
        async for chunk in image_stream_helper(prompt, samples):
            yield sse(chunk)
    except HTTPException as e:
        yield sse({"type": "image_error", "error": e.detail})

async def run_code_safely(prompt: str):
    """Helper for streaming /ask/universal."""
    # Default to python if not specified for this helper
    language = "python" 
    
    # 1. Generate code
    code_prompt = f"Write a complete {language} program to: {prompt}"
    payload = {
    "model": CHAT_MODEL,
    "messages": [{"role": "user", "content": code_prompt}],
    "max_tokens": 2048
}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json=payload
        )
        r.raise_for_status()
        code = r.json()["choices"][0]["message"]["content"]

    # 2. Run code
    lang_id = JUDGE0_LANGUAGES.get(language, 71)
    execution = await run_code_judge0(code, lang_id)
    
    return {"code": code, "execution": execution}


@app.post("/ask/universal")
async def ask_universal(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    prompt = body.get("prompt", "").strip()
    user_id = body.get("user_id", "anonymous")
    stream = bool(body.get("stream", False))

    if not prompt:
        raise HTTPException(400, "prompt required")

    # ✅ Load previous conversation
    history = await load_history(user_id)
    messages = history + [{"role": "user", "content": prompt}]

    # =========================
    # NON-STREAM MODE
    # =========================
    if not stream:
        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                )
                r.raise_for_status()
                result = r.json()

            # ✅ Save assistant reply
            assistant_content = result["choices"][0]["message"]["content"]
            await save_message(user_id, "assistant", assistant_content)

            return result

        except httpx.HTTPStatusError as e:
            raise HTTPException(500, f"Groq error: {e.response.text}")

        except Exception as e:
            raise HTTPException(500, str(e))

    # =========================
    # STREAM MODE
    # =========================
    async def event_generator():
        yield sse({"type": "starting"})

        payload = {
            "model": CHAT_MODEL,
            "stream": True,
            "messages": messages,
            "max_tokens": 1024
        }

        assistant_reply = ""

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                assistant_reply += delta
                                yield sse({"type": "token", "text": delta})
                        except Exception:
                            continue

            # ✅ Save assistant reply after streaming
            await save_message(user_id, "assistant", assistant_reply)

        except Exception as e:
            yield sse({"type": "error", "error": str(e)})

        yield sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/message/{message_id}/edit")
async def edit_message(
    message_id: str,
    req: Request,
    res: Response
):
    user_id = await get_or_create_user(req, res)
    body = await req.json()
    new_text = body.get("content")

    if not new_text:
        raise HTTPException(400, "content required")

    # Get message
    msg = supabase.table("messages") \
        .select("id,role,conversation_id,created_at") \
        .eq("id", message_id) \
        .single() \
        .execute()

    if not msg.data:
        raise HTTPException(404, "message not found")

    if msg.data["role"] != "user":
        raise HTTPException(403, "only user messages can be edited")

    conversation_id = msg.data["conversation_id"]
    edited_at = msg.data["created_at"]

    # Update message content
    supabase.table("messages").update({
        "content": new_text
    }).eq("id", message_id).execute()

    # 🔥 DELETE ALL ASSISTANT MESSAGES AFTER THIS MESSAGE
    supabase.table("messages") \
        .delete() \
        .eq("conversation_id", conversation_id) \
        .gt("created_at", edited_at) \
        .eq("role", "assistant") \
        .execute()

    return {
        "status": "edited",
        "conversation_id": conversation_id
    }
    
# -----------------------------
# Stop endpoint
# -----------------------------
@app.post("/stop")
async def stop_stream(req: Request, res: Response):
    # ✅ resolve user from cookie
    user_id = await get_or_create_user(req, res)

    task = active_streams.get(user_id)
    if task and not task.done():
        task.cancel()
        return {"status": "stopped"}

    return {"status": "no_active_stream"}
    
# -----------------------------
# Regenerate endpoint
# -----------------------------
@app.post("/regenerate")
async def regenerate(req: Request, res: Response, tts: bool = False, samples: int = 1):
    """
    Cancel current stream (if any) and re-run the prompt as a fresh stream.
    Cookie-based, streaming-safe.
    """
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(400, "prompt required")

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)

    # ✅ CANCEL EXISTING STREAM (IF ANY)
    old_task = active_streams.get(user_id)
    if old_task and not old_task.done():
        old_task.cancel()

    async def event_generator():
        # ✅ REGISTER NEW STREAM
        task = asyncio.current_task()
        active_streams[user_id] = task

        try:
            # --- IMAGE (OPTIONAL) ---
            if any(w in prompt.lower() for w in ("image", "draw", "illustrate", "painting", "art", "picture")):
                try:
                    yield sse({"status": "image_start", "message": "Regenerating image"})

                    img_payload = {
                        "prompt": prompt,
                        "samples": samples,
                        "base64": False
                    }

                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "http://127.0.0.1:8000/image/stream",
                            json=img_payload
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if line.strip():
                                    yield line + "\n\n"

                    yield sse({"status": "image_done"})

                except Exception:
                    logger.exception("Image regenerate failed")
                    yield sse({"status": "image_error"})

            # --- CHAT ---
            payload = {
                "model": CHAT_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
                    {"role": "user", "content": prompt}
                ]
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break

                        yield sse({
                            "status": "chat_progress",
                            "message": data
                        })

            # --- TTS (OPTIONAL) ---
            if tts:
                try:
                    tts_payload = {
                        "model": "tts-1",
                        "voice": "alloy",
                        "input": prompt
                    }

                    headers = {
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    }

                    audio_buffer = bytearray()

                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "https://api.openai.com/v1/audio/speech",
                            headers=headers,
                            json=tts_payload
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                if chunk:
                                    audio_buffer.extend(chunk)

                    yield sse({
                        "status": "tts_done",
                        "audio": base64.b64encode(audio_buffer).decode()
                    })

                except Exception:
                    logger.exception("TTS regenerate failed")
                    yield sse({"status": "tts_error"})

            yield sse({"status": "done"})

        except asyncio.CancelledError:
            logger.info(f"Regenerate cancelled for user {user_id}")
            yield sse({"status": "stopped"})
            raise

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

   
@app.post("/video")
async def generate_video(request: Request):
    """
    Generate a video from a prompt using Hugging Face and upload to Supabase.
    Returns signed URL(s) to the video(s).
    """
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    user_id = body.get("user_id", "anonymous")
    samples = max(1, int(body.get("samples", 1)))

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not HF_API_KEY:
        raise HTTPException(500, "HF_API_KEY missing")

    video_urls = []

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            for _ in range(samples):
                payload = {"inputs": prompt}

                r = await client.post(
                    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-videos",
                    headers=headers,
                    json=payload
                )
                r.raise_for_status()
                # HF returns bytes, not base64
                video_bytes = r.content

                # Generate a unique filename
                filename = f"{user_id}/{int(time.time())}-video-{uuid.uuid4().hex[:10]}.mp4"

                # Upload to Supabase
                supabase.storage.from_("ai-videos").upload(
                    path=filename,
                    file=video_bytes,
                    file_options={"content-type": "video/mp4"}
                )

                # Create signed URL (1 hour)
                signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
                video_urls.append(signed["signedURL"])

    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {str(e)}")

    if not video_urls:
        raise HTTPException(500, "No video generated")

    return {"provider": "huggingface", "videos": video_urls}
    
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    user_id = body.get("user_id", "anonymous")

    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1

    return_base64 = bool(body.get("base64", False))

    if not prompt:
        raise HTTPException(400, "prompt required")

    return await _generate_image_core(prompt, samples, user_id, return_base64)

@app.get("/test-stream")
async def test_stream(request: Request):
    async def event_generator():
        for i in range(1, 6):
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield sse({"message": f"This is chunk {i}"})
            await asyncio.sleep(1)
        yield sse({"message": "Done"})
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
@app.post("/image/stream")
async def image_stream(req: Request, res: Response):
    """
    Stream progress to the client while generating images.
    Uses cookies for user identity and supports safe cancellation.
    """
    body = await req.json()
    prompt = body.get("prompt", "")

    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1

    return_base64 = bool(body.get("base64", False))

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not OPENAI_API_KEY:
        raise HTTPException(400, "no OpenAI KEY")

    # ✅ COOKIE-BASED USER ID
    user_id = await get_or_create_user(req, res)

    async def event_generator():
        # ✅ REGISTER STREAM TASK (MUST BE HERE)
        task = asyncio.current_task()
        active_streams[user_id] = task

        try:
            # --- initial message ---
            yield sse({"status": "starting", "message": "Preparing request"})
            await asyncio.sleep(0)

            payload = {
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,  # DALL·E 3 supports only 1
                "size": "1024x1024",
                "response_format": "b64_json"
            }

            yield sse({"status": "request", "message": "Sending to OpenAI"})

            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )

            if r.status_code != 200:
                text_snip = (await r.aread()).decode(errors="ignore")[:1000]
                yield sse({
                    "status": "error",
                    "message": "OpenAI error",
                    "detail": text_snip
                })
                return

            jr = r.json()
            urls = []

            data_list = jr.get("data", [])
            if not data_list:
                yield sse({"status": "warning", "message": "No data returned from provider"})

            for i, d in enumerate(data_list, start=1):
                yield sse({
                    "status": "progress",
                    "message": f"Processing {i}/{samples}"
                })
                await asyncio.sleep(0)

                b64 = d.get("b64_json")
                if not b64:
                    continue

                try:
                    image_bytes = base64.b64decode(b64)
                    filename = f"streaming/{unique_filename('png')}"
                    upload_image_to_supabase(image_bytes, filename)

                    signed = supabase.storage.from_("ai-images").create_signed_url(
                        filename, 60 * 60
                    )
                    urls.append(signed["signedURL"])

                except Exception as e:
                    logger.exception("Supabase upload failed in stream")
                    yield sse({
                        "status": "error",
                        "message": f"Storage failed: {str(e)}"
                    })

            yield sse({"status": "done", "images": urls})

        except asyncio.CancelledError:
            logger.info(f"Image stream cancelled for user {user_id}")
            yield sse({"status": "stopped"})
            raise

        except Exception as e:
            logger.exception("image_stream exception")
            yield sse({"status": "exception", "message": str(e)})

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
# ---------- Img2Img (DALL·E edits) ----------
@app.post("/img2img")
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = "", user_id: str = "anonymous"):
    if not prompt:
        raise HTTPException(400, "prompt required")
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    if not OPENAI_API_KEY:
        raise HTTPException(400, "no OpenAI API key configured")

    urls = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"image": (file.filename, content)}
            data = {"prompt": prompt, "n": 1, "size": "1024x1024"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            r = await client.post("https://api.openai.com/v1/images/edits", headers=headers, files=files, data=data)
            r.raise_for_status()
            jr = r.json()
            for d in jr.get("data", []):
                b64 = d.get("b64_json")
                if b64:
                    fname = unique_filename("png")
                    # Note: Edit API still returns B64 or URL. If B64, we MUST upload to Supabase.
                    # If we use local_image_url for edits, we will hit the same 404 error on Railway.
                    # Let's upload to Supabase here too for consistency.
                    
                    
                    supabase_fname = f"{user_id}/edits/{fname}"
                    upload_image_to_supabase(image_bytes, supabase_fname)
                    signed = supabase.storage.from_("ai-images").create_signed_url(supabase_fname, 60*60)
                    urls.append(signed["signedURL"])
    except Exception:
        logger.exception("img2img DALL-E edit failed")
        raise HTTPException(400, "img2img failed")

    return {"provider": "dalle3-edit", "images": urls}
    
# ---------- TTS ----------
@app.post("/tts")
async def text_to_speech(request: Request):
    """
    Convert text to speech using OpenAI TTS (tts-1).
    Accepts either JSON: {"text": "..."} or raw text/plain in the body.
    Returns audio/mpeg directly.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    # Try JSON first
    try:
        data = await request.json()
        text = data.get("text", None)
    except Exception:
        # Fallback: read raw text from body
        text = (await request.body()).decode("utf-8")

    if not text or not text.strip():
        raise HTTPException(400, "Missing 'text' in request")

    payload = {
        "model": "tts-1", 
        "voice": "alloy",  # default voice
        "input": text.strip(),
        "format": "mp3"
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload
            )
            r.raise_for_status()

            return Response(
                content=r.content,
                media_type="audio/mpeg"
            )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {"error": f"OpenAI HTTP error: {e.response.status_code}", "detail": e.response.text},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"error": "TTS request failed", "detail": str(e)},
            status_code=500
        )
        
@app.post("/tts/stream")
async def tts_stream(req: Request, res: Response):
    data = await req.json()
    text = data.get("text", "")

    if not text:
        raise HTTPException(400, "text required")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "tts-1",
        "voice": "alloy",
        "input": text
    }

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)

    async def audio_streamer():
        # ✅ REGISTER STREAM TASK
        task = asyncio.current_task()
        active_streams[user_id] = task

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/audio/speech",
                    json=payload,
                    headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk

        except asyncio.CancelledError:
            logger.info(f"TTS stream cancelled for user {user_id}")
            raise

        except Exception as e:
            logger.exception("TTS streaming failed")
            # Audio streams cannot emit JSON errors safely mid-stream

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)

    return StreamingResponse(
        audio_streamer(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ---------- Vision analyze ----------
@app.post("/vision/analyze")
async def vision_analyze(
    req: Request,
    res: Response,
    file: UploadFile = File(...)
):
    user_id = await get_or_create_user(req, res)
    content = await file.read()

    if not content:
        raise HTTPException(400, "empty file")

    # Load image
    img = Image.open(BytesIO(content)).convert("RGB")
    np_img = np.array(img)
    annotated = np_img.copy()

    # =========================
    # 1️⃣ YOLO OBJECT DETECTION
    # =========================
    obj_results = get_yolo_objects()(np_img, conf=0.25)
    detections = []

    for r in obj_results:
        for box in r.boxes:
            label = YOLO_OBJECTS.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    # =========================
    # 2️⃣ FACE DETECTION
    # =========================
    face_results = get_yolo_faces()(np_img)
    face_count = 0

    for r in face_results:
        for box in r.boxes:
            face_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(
                annotated,
                "face",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,0),
                2
            )

    # =========================
    # 3️⃣ DOMINANT COLORS
    # =========================
    hex_colors = []
    try:
        from sklearn.cluster import KMeans
        pixels = np_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
        hex_colors = [
            '#%02x%02x%02x' % tuple(map(int, c))
            for c in kmeans.cluster_centers_
        ]
    except Exception:
        pass

    # =========================
    # 4️⃣ UPLOAD TO SUPABASE
    # =========================
    raw_path = f"{user_id}/raw/{uuid.uuid4().hex}.png"
    ann_path = f"{user_id}/annotated/{uuid.uuid4().hex}.png"

    _, ann_buf = cv2.imencode(".png", annotated)

    supabase.storage.from_("ai-images").upload(
        raw_path,
        content,
        {"content-type": "image/png", "upsert": True}
    )

    supabase.storage.from_("ai-images").upload(
        ann_path,
        ann_buf.tobytes(),
        {"content-type": "image/png", "upsert": True}
    )

    raw_url = supabase.storage.from_("ai-images").create_signed_url(raw_path, 3600)["signedURL"]
    ann_url = supabase.storage.from_("ai-images").create_signed_url(ann_path, 3600)["signedURL"]

    # =========================
    # 5️⃣ SAVE HISTORY
    # =========================
    supabase.table("vision_history").insert({
        "user_id": user_id,
        "image_path": raw_path,
        "annotated_path": ann_path,
        "detections": detections,
        "faces": face_count
    }).execute()

    return {
        "objects": detections,
        "faces_detected": face_count,
        "dominant_colors": hex_colors,
        "image_url": raw_url,
        "annotated_image_url": ann_url
    }

@app.get("/vision/history")
async def vision_history(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    data = supabase.table("vision_history") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(50) \
        .execute()

    return data.data or []


    
# ---------- Code generation ----------
@app.post("/code")
async def code_gen(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    language = body.get("language", "python").lower()
    run_flag = bool(body.get("run", False))
    user_id = body.get("user_id", "anonymous")

    if not prompt:
        raise HTTPException(400, "prompt required")

    # Generate code using Groq (unchanged)
    contextual_prompt = build_contextual_prompt(
        user_id,
        f"Write a complete {language} program:\n{prompt}"
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": contextual_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json=payload
        )
        r.raise_for_status()
        code = r.json()["choices"][0]["message"]["content"]

    response = {
        "language": language,
        "generated_code": code
    }

    # ✅ Run via Judge0
    if run_flag:
        lang_id = JUDGE0_LANGUAGES.get(language, 71)
        execution = await run_code_judge0(code, lang_id)
        response["execution"] = execution

    return response

@app.get("/search")
async def duck_search(q: str = Query(..., min_length=1)):
    """
    Lightweight search endpoint backed by DuckDuckGo Instant Answer API.
    Example: /search?q=python+asyncio
    """
    try:
        return await duckduckgo_search(q)
    except httpx.HTTPStatusError as e:
        logger.exception("DuckDuckGo returned HTTP error")
        raise HTTPException(502, "duckduckgo_error")
    except Exception:
        logger.exception("DuckDuckGo search failed")
        raise HTTPException(500, "search_failed")

# ---------- STT ----------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/audio/transcriptions"

    # Whisper API requires multipart/form-data, NOT JSON
    files = {
        "file": (file.filename, content, file.content_type or "audio/mpeg"),
        "model": (None, "whisper-1"),
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, files=files)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"OpenAI STT error: {r.text}")

    data = r.json()
    return {"transcription": data.get("text", "")}
