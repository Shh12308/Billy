# app.py — ZyNara1 AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import PIL
import json
import uuid
import sqlite3
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
import asyncio
import base64 as b64lib
import time
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any, List
from fastapi import Response

import httpx
from fastapi import FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from supabase import create_client

# ---------- ENV KEYS ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")
    
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("billy-server")

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
# Database Paths (Local fallback)
# -------------------
KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")

# -------------------
# Models
# -------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")

# TTS/STT are handled via ElevenLabs now
TTS_MODEL = None
STT_MODEL = None

# Image hosting dirs (Kept for legacy/local testing, but logic forces Supabase)
STATIC_DIR = os.getenv("STATIC_DIR", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldYLocks",
    "age": 17,
    "country": "England",
    "projects": ["MintZa", "LuxStream", "SwapX", "CryptoBean"],
    "socials": { "discord":"@nexisphere123_89431", "twitter":"@NexiSphere"},
    "bio": "Created by GoldBoy (17, England). Projects: MZ, LS, SX, CB. Socials: Discord @nexisphere123_89431 Twitter @NexiSphere."
}

JUDGE0_LANGUAGES = {
    "python": 71,
    "javascript": 63,
    "cpp": 54,
    "c": 50,
    "java": 62,
    "go": 60,
    "rust": 73
}

JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
JUDGE0_KEY = os.getenv("JUDGE0_API_KEY")

if not JUDGE0_KEY:
    logger.warning("⚠️ Judge0 key not set — code execution disabled")

async def run_code_judge0(code: str, language: str) -> dict:
    if not JUDGE0_KEY:
        return {"error": "Judge0 not configured"}

    lang_id = JUDGE0_LANGUAGES.get(language.lower())
    if not lang_id:
        return {"error": f"Unsupported language: {language}"}

    headers = {
        "X-RapidAPI-Key": JUDGE0_KEY,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Submit code
        submit = await client.post(
            f"{JUDGE0_URL}/submissions?wait=false",
            headers=headers,
            json={
                "source_code": code,
                "language_id": lang_id,
                "stdin": ""
            }
        )
        submit.raise_for_status()
        token = submit.json()["token"]

        # 2. Poll result
        for _ in range(15):
            await asyncio.sleep(1)
            r = await client.get(
                f"{JUDGE0_URL}/submissions/{token}",
                headers=headers
            )
            r.raise_for_status()
            data = r.json()

            if data["status"]["id"] >= 3:
                return {
                    "status": data["status"]["description"],
                    "stdout": data.get("stdout"),
                    "stderr": data.get("stderr"),
                    "compile_output": data.get("compile_output"),
                    "time": data.get("time"),
                    "memory": data.get("memory")
                }

        return {"error": "Execution timed out"}

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

# ---------- SQLITE helpers ----------
def ensure_db(path: str, schema_sql: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(schema_sql)
    conn.commit()
    conn.close()

# Knowledge Graph
ensure_db(KG_DB_PATH, """
CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  content TEXT,
  created_at REAL
);
""")

# Memory
ensure_db(MEMORY_DB, """
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  key TEXT,
  value TEXT,
  updated_at REAL
);
""")

# Cache for images/prompts
ensure_db(CACHE_DB_PATH, """
CREATE TABLE IF NOT EXISTS cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT,
  provider TEXT,
  result TEXT,
  created_at REAL
);
""")

def add_kg_node(title: str, content: str):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO nodes (title, content, created_at) VALUES (?, ?, ?)", (title, content, time.time()))
    conn.commit()
    conn.close()

def query_kg(q: str, limit: int = 5):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, content FROM nodes WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?", (f"%{q}%", f"%{q}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [{"title": r[0], "content": r[1]} for r in rows]

def cache_result(prompt: str, provider: str, result: Any):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO cache (prompt, provider, result, created_at) VALUES (?, ?, ?, ?)", (prompt, provider, json.dumps(result), time.time()))
    conn.commit()
    conn.close()

def get_cached(prompt: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT provider,result FROM cache WHERE prompt=? ORDER BY created_at DESC LIMIT 1", (prompt,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"provider": row[0], "result": json.loads(row[1])}
    return None

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
def upload_image_to_supabase(image_bytes, filename):
    return upload_to_supabase(image_bytes, filename, bucket="ai-images", content_type="image/png")

def local_image_url(request: Request, filename: str) -> str:
    # Kept for legacy, but primary path uses Supabase URLs
    return f"{request.base_url}static/images/{filename}"

def save_base64_image_to_file(b64_str: str, filename: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_data)

def save_image_record(user_id, prompt, path, is_nsfw):
    supabase.table("images").insert({
        "user_id": user_id,
        "prompt": prompt,
        "image_path": path,
        "is_nsfw": is_nsfw
    }).execute()

async def nsfw_check(prompt: str) -> bool:
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

    user_id = str(uuid.uuid4()) # Fixed: uuid4() -> uuid.uuid4()
    supabase.table("users").insert({"id": user_id}).execute()

    res.set_cookie(
        key="uid",
        value=user_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 365
    )
    return user_id

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

    convo_id = str(uuid.uuid4()) # Fixed
    supabase.table("conversations").insert({
        "id": convo_id,
        "user_id": user_id,
        "title": "New Chat"
    }).execute()

    return {"conversation_id": convo_id}

# ----------------------------------
# SEND MESSAGE (MEMORY AWARE)
# ----------------------------------
@app.post("/chat/{conversation_id}/message")
async def send_message(
    conversation_id: str,
    req: Request,
    res: Response
):
    user_id = await get_or_create_user(req, res)
    body = await req.json()
    text = body.get("message")

    if not text:
        raise HTTPException(400, "message required")

    # Save user message
    supabase.table("messages").insert({
        "id": str(uuid.uuid4()), # Fixed
        "conversation_id": conversation_id,
        "role": "user",
        "content": text
    }).execute()

    # LOAD MEMORY FOR AI
    memory = load_memory(conversation_id)

    # ----------------------------------
    # ⬇️ REPLACE WITH YOUR GROQ CALL ⬇️
    # ----------------------------------
    ai_reply = "AI RESPONSE HERE"
    # ----------------------------------

    supabase.table("messages").insert({
        "id": str(uuid.uuid4()), # Fixed
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": ai_reply
    }).execute()

    return {"reply": ai_reply}

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
        "model": "tts-1", # Fixed model name
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
        supabase.storage.from_("ai-videos").upload(
            path=filename,
            file=placeholder_content,
            file_options={"content-type": "video/mp4"}
        )

        # Get signed URL
        signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
        urls.append(signed["signedURL"])

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
            {"role": "system", "content": build_contextual_prompt(user_id, prompt)},
            {"role": "user", "content": prompt}
        ]
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

# ---------- Streaming chat ----------
async def chat_stream_helper(user_id: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": build_contextual_prompt(user_id, prompt)},
            {"role": "user", "content": prompt}
        ]
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

@app.get("/")
async def root():
    return {"message": "Billy AI Backend is Running ✔"}
    
@app.post("/chat/stream")
async def chat_stream(req: Request, tts: bool = False, samples: int = 1, user_id: str = "anonymous"):
    """
    Unified streaming endpoint:
    - Streams chat responses from Groq
    - Streams image generation if prompt implies an image
    - Sends TTS audio (as base64) if tts=True at the end
    SSE events:
      chat_progress, image_progress, tts_done, done
    """
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")

    async def event_generator():
        # --- 1. Image Generation (if prompt implies image) ---
        if any(w in prompt.lower() for w in ("image","draw","illustrate","painting","art","picture")):
            try:
                yield f"data: {json.dumps({'status':'image_start','message':'Starting image generation'})}\n\n"
                img_payload = {"prompt": prompt, "samples": samples, "base64": False}
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", str(req.base_url) + "image/stream", json=img_payload) as resp:
                        async for line in resp.aiter_lines():
                            if line.strip():
                                yield line + "\n\n"
                yield f"data: {json.dumps({'status':'image_done','message':'Image generation complete'})}\n\n"
            except Exception:
                logger.exception("Image streaming failed")
                yield f"data: {json.dumps({'status':'image_error','message':'Image generation failed'})}\n\n"

        # --- 2. Chat Streaming ---
        try:
            payload = {
                "model": CHAT_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": build_contextual_prompt(user_id, prompt)},
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
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            yield f"data: {json.dumps({'status':'chat_progress','message':data})}\n\n"
        except Exception:
            logger.exception("Chat streaming failed")
            yield f"data: {json.dumps({'status':'chat_error','message':'Chat stream failed'})}\n\n"

        # --- 3. TTS (optional) ---
        if tts:
            try:
                tts_payload = {
                    "model": "tts-1", # Fixed model name
                    "voice": "alloy",
                    "input": prompt,
                    "format": "mp3",
                    "stream": True
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
                # Send final base64-encoded audio
                b64_audio = base64.b64encode(audio_buffer).decode("utf-8")
                yield f"data: {json.dumps({'status':'tts_done','audio':b64_audio})}\n\n"
            except Exception:
                logger.exception("TTS streaming failed")
                yield f"data: {json.dumps({'status':'tts_error','message':'TTS failed'})}\n\n"

        yield f"data: {json.dumps({'status':'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
    
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
                # Log provider response for debugging (trim to avoid huge logs)
                logger.warning("Groq /chat returned status=%s text=%s", r.status_code, (r.text[:500] + '...') if r.text else "")
                r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            # Provide more helpful error to the caller while logging details
            logger.exception("Groq HTTP error on /chat: %s", getattr(exc.response, "text", "no-response-text"))
            raise HTTPException(status_code=exc.response.status_code if exc.response is not None else 500, detail=f"Groq error: {exc.response.text[:400] if exc.response is not None else str(exc)}")
        except Exception:
            logger.exception("Groq /chat request failed")
            raise HTTPException(500, "groq_request_failed")

# =========================================================
# 🚀 UNIVERSAL MULTIMODAL ENDPOINT — /ask/universal
# =========================================================

# ---------- Core Image Logic (Refactored) ----------
async def _generate_image_core(prompt, samples, "anonymous", return_base64=False)
    """Shared logic for image generation used by /image and streaming helpers."""
    cached = get_cached(prompt)
    if cached:
        return {"cached": True, **cached}

    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY missing")

    urls = []
    provider_used = "openai"

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": "dall-e-3", # Fixed model name
            "prompt": prompt,
            "n": 1, # DALL-E 3 supports n=1
            "size": "1024x1024",
            "response_format": "b64_json"
        }
        
        # Adjust samples for DALL-E 3 limitation
        actual_samples = 1 
        if samples > 1:
            logger.warning("DALL-E 3 only supports 1 image per request. Adjusting samples to 1.")

        r = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        r.raise_for_status()
        jr = r.json()

    for d in jr.get("data", []):
        b64 = d.get("b64_json")
        
        if b64:
            try:
                image_bytes = base64.b64decode(b64)
                
                # Skip NSFW check here for speed in stream, or keep if needed
                # flagged = await nsfw_check(prompt)
                # if flagged: ...

                if base64:
                    urls.append(b64)
                else:
                    filename = f"{user_id}/{unique_filename('png')}"
                    upload_image_to_supabase(image_bytes, filename)
                    # flagged = False # Assuming false for stream speed
                    # save_image_record(user_id, prompt, filename, flagged)
                    
                    signed = supabase.storage.from_("ai-images").create_signed_url(filename, 60 * 60)
                    urls.append(signed["signedURL"])

            except Exception:
                logger.exception("Failed processing OpenAI base64 image")

    if not urls:
        raise HTTPException(500, "Image generation failed")

    cache_result(prompt, provider_used, urls)
    return {"provider": provider_used, "images": urls}


async def image_gen_internal(prompt: str, samples: int = 1):
    """Helper for streaming /ask/universal."""
    # Returns dict {"images": [...]}
    return await _generate_image_core(prompt, samples, "anonymous", return_base64=False)

async def image_stream_helper(prompt: str, samples: int):
    yield {"type": "image_start"}
    result = await image_gen_internal(prompt, samples)
    for url in result["images"]:
        yield {"type": "image", "url": url}
    yield {"type": "image_done"}

async def run_code_safely(prompt: str):
    """Helper for streaming /ask/universal."""
    # Default to python if not specified for this helper
    language = "python" 
    
    # 1. Generate code
    code_prompt = f"Write a complete {language} program to: {prompt}"
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": code_prompt}]
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
    execution = await run_code_judge0(code, language)
    
    return {"code": code, "execution": execution}


@app.post("/ask/universal")
async def ask_universal(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or missing JSON body"}
        )

    prompt = body.get("prompt", "").strip()
    user_id = body.get("user_id", "anonymous")
    stream = bool(body.get("stream", False))
    tts = bool(body.get("tts", False))
    samples = int(body.get("samples", 1))
    intent = body.get("mode") or detect_intent(prompt)

    if not prompt:
        raise HTTPException(400, "prompt required")

    # =========================
    # NON-STREAM MODE (Moved up to be reachable)
    # =========================
    if not stream:
        # We avoid request._json hacking by calling logic directly or re-invoking endpoints via internal calls if possible.
        # For simplicity and stability, we implement the logic inline here.
        
        if intent == "chat":
            # Inline logic from chat_endpoint
            payload = {"model":CHAT_MODEL,"messages":[{"role":"system","content":build_contextual_prompt(user_id, prompt)},{"role":"user","content":prompt}]}
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=get_groq_headers(), json=payload)
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    raise HTTPException(500, str(e))

        if intent == "image":
            # Logic from image_gen
            return await _generate_image_core(prompt, samples, "anonymous", return_base64=False)

        if intent == "search":
            return await duckduckgo_search(prompt)

        if intent == "code":
            # Logic from code_gen
            language = body.get("language", "python").lower()
            code_prompt = f"Write a complete {language} program:\n{prompt}"
            payload = {
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": build_contextual_prompt(user_id, code_prompt)},
                    {"role": "user", "content": prompt}
                ]
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=get_groq_headers(), json=payload)
                r.raise_for_status()
                code = r.json()["choices"][0]["message"]["content"]
            
            response = {"language": language, "generated_code": code}
            if body.get("run", False):
                response["execution"] = await run_code_judge0(code, language)
            return response

        if intent == "tts":
            # Logic from text_to_speech
            if not OPENAI_API_KEY: raise HTTPException(500, "Missing OPENAI_API_KEY")
            tts_payload = {"model": "tts-1", "voice": "alloy", "input": prompt, "format": "mp3"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post("https://api.openai.com/v1/audio/speech", headers=headers, json=tts_payload)
                r.raise_for_status()
                return Response(content=r.content, media_type="audio/mpeg")

        # Fallback
        payload = {"model":CHAT_MODEL,"messages":[{"role":"system","content":build_contextual_prompt(user_id, prompt)},{"role":"user","content":prompt}]}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=get_groq_headers(), json=payload)
            r.raise_for_status()
            return r.json()


    async def event_generator():
        yield sse({"type": "starting", "intent": intent})

        if intent == "chat":
            async for chunk in chat_stream_helper(user_id, prompt):
                yield sse(chunk)

        elif intent == "image":
            async for chunk in image_stream_helper(prompt, samples):
                yield sse(chunk)

        elif intent == "video":
            result = await generate_video_internal(prompt, samples, user_id)
            yield sse({"type": "video_result", "data": result})

        elif intent == "search":
            result = await duckduckgo_search(prompt)
            yield sse({"type": "search_result", "data": result})

        elif intent == "code":
            result = await run_code_safely(prompt)
            yield sse({"type": "code_result", "data": result})

        if tts:
            async for chunk in tts_stream_helper(prompt):
                yield sse(chunk)

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
async def stop_stream(user_id: str):
    task = active_streams.get(user_id)
    if task and not task.done():
        task.cancel()
        return {"status": "stopped"}
    return {"status": "no_active_stream"}
    
# -----------------------------
# Regenerate endpoint
# -----------------------------
@app.post("/regenerate")
async def regenerate(user_id: str, prompt: str, mode: str = "chat", samples: int = 1):
    """
    Re-run the same prompt as a fresh request.
    """
    # Map to internal logic
    return await ask_universal_helper(prompt=prompt, user_id=user_id, mode=mode, samples=samples)

# Dummy function for regeneration helper
async def ask_universal_helper(prompt, user_id, mode, samples):
    # Simply calls the streaming generator logic but returns final result? 
    # The code structure suggests this should just trigger a new session.
    # Since /ask/universal is the main entry, we can mock a request object or just reuse the logic.
    # For now, let's just return a placeholder or trigger a response.
    return {"status": "regenerating", "prompt": prompt}

# ---------------- SSE HELPER ----------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"

   
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
async def image_stream(request: Request):
    """
    Stream progress to the client while generating images.
    Sends SSE messages with JSON payloads and final 'done' event.
    Uses Supabase for permanent storage (Fixed for Railway Ephemeral Storage).
    """
    body = await request.json()
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

    async def event_generator():
        try:
            # initial message
            yield "data: " + json.dumps({"status": "starting", "message": "Preparing request"}) + "\n\n"
            await asyncio.sleep(0.05)

            payload = {
                "model": "dall-e-3", # Fixed model name
                "prompt": prompt,
                "n": 1, # Dalle 3 supports only 1
                "size": "1024x1024",
                "response_format": "b64_json"
            }

            yield "data: " + json.dumps({"status": "request", "message": "Sending to OpenAI"}) + "\n\n"

            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/images/generations", # Corrected API URL
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )

            if r.status_code != 200:
                text_snip = (await r.aread()).decode(errors="ignore")[:1000]
                yield "data: " + json.dumps({"status": "error", "message": "OpenAI error", "detail": text_snip}) + "\n\n"
                yield "data: [DONE]\n\n"
                return

            jr = r.json()
            urls = []

            data_list = jr.get("data", [])
            if not data_list:
                yield "data: " + json.dumps({"status": "warning", "message": "No data returned from provider"}) + "\n\n"

            for i, d in enumerate(data_list, start=1):
                yield "data: " + json.dumps({"status": "progress", "message": f"Processing {i}/{samples}"}) + "\n\n"
                await asyncio.sleep(0.01)  # yield control
                b64 = d.get("b64_json")
                
                if b64:
                    # --- FORCE SUPABASE UPLOAD (REMOVED LOCAL SAVING) ---
                    try:
                        image_bytes = base64.b64decode(b64)
                        filename = f"streaming/{unique_filename('png')}" # Use a subfolder for streaming
                        upload_image_to_supabase(image_bytes, filename)
                        
                        # Get signed URL
                        signed = supabase.storage.from_("ai-images").create_signed_url(filename, 60*60)
                        urls.append(signed["signedURL"])
                        
                    except Exception as e:
                        logger.exception("Supabase upload failed in stream")
                        yield "data: " + json.dumps({"status": "error", "message": f"Storage failed: {str(e)}"}) + "\n\n"
                
            yield "data: " + json.dumps({"status": "done", "images": urls}) + "\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("image_stream exception: %s", str(e))
            yield "data: " + json.dumps({"status": "exception", "message": str(e)}) + "\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
    
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
                    save_base64_image_to_file(b64, fname) 
                    image_bytes = base64.b64decode(b64)
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
        "model": "tts-1", # Fixed model name
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
async def tts_stream(request: Request):
    data = await request.json()
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
        "model": "tts-1", # Fixed model name
        "voice": "alloy",
        "input": text,
        "stream": True
    }

    async def audio_streamer():
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

    return StreamingResponse(audio_streamer(), media_type="audio/mpeg")


# ---------- Vision analyze ----------
@app.post("/vision/analyze")
async def vision_analyze(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    
    # Load image
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "invalid image")

    # ----- 1. Dominant colors using k-means -----
    hex_colors = []
    try:
        # sklearn import protection
        from sklearn.cluster import KMeans
        np_img = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(np_img)
        colors = [tuple(map(int, c)) for c in kmeans.cluster_centers_]
        hex_colors = ['#%02x%02x%02x' % c for c in colors]
    except ImportError:
        logger.warning("scikit-learn not installed, skipping color analysis")
    except Exception as e:
        logger.warning(f"Color analysis failed: {e}")

    # ----- 2. Object detection (pretrained ResNet) -----
    object_label = None
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        idx_to_label = models.ResNet50_Weights.DEFAULT.meta["categories"]
        object_label = idx_to_label[predicted.item()]
    except Exception as e:
        logger.warning(f"Vision object detection failed: {e}")

    # ----- 3. Suggested tags -----
    tags = []
    if object_label:
        tags.append(object_label.lower())
    tags += [f"color_{c[1:]}" for c in hex_colors[:3]]  # top 3 colors

    # ----- 4. Short description -----
    description = f"A {object_label} image with dominant colors {', '.join(hex_colors[:3])}" if object_label else "Image analysis available."

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "dominant_colors": hex_colors,
        "objects": object_label,
        "tags": tags,
        "description": description
    }

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
        execution = await run_code_judge0(code, language)
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

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
