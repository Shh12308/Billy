# app.py — ZyNara1 AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import json
import uuid
import sqlite3
import asyncio
import base64
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

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("billy-server")

app = FastAPI(title="Billy AI Multimodal Server")

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

IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "").strip()
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

# -------------------
# Supabase Config
# -------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Image hosting dirs
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

# ---------- Dynamic, user-focused system prompt ----------
def get_system_prompt(user_message: Optional[str] = None) -> str:
    base = "You are Billy AI: helpful, concise, friendly, and focus entirely on what the user asks. Do not reference your creator or yourself unless explicitly asked."
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
    return f"You are ZyNara1 AI: helpful, concise, friendly. Focus on exactly what the user wants.\nUser context:\n{context}\nUser message: {message}"

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

def local_image_url(request: Request, filename: str):
    return str(request.base_url).rstrip("/") + f"/static/images/{filename}"

def save_base64_image_to_file(b64: str, filename: str) -> str:
    path = os.path.join(IMAGES_DIR, filename)
    img_bytes = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

#------duckduckgo

async def duckduckgo_search(q: str):
    """
    Use DuckDuckGo Instant Answer API (no API key required).
    Returns a simple structured result with abstract, answer and a list of related topics.
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
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

def run_code_safely(code: str, language: str = "python") -> Dict[str, str]:
    """
    Run code in a temporary file safely.
    Returns {'output': ..., 'error': ...}
    """
    if language.lower() != "python":
        return {"output": "", "error": f"Execution for {language} not supported yet."}

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=True) as tmpfile:
        tmpfile.write(code)
        tmpfile.flush()
        try:
            result = subprocess.run(
                ["python3", tmpfile.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            return {
                "output": result.stdout.decode().strip(),
                "error": result.stderr.decode().strip()
            }
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Execution timed out."}
        except Exception as e:
            return {"output": "", "error": str(e)}



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
async def groq_stream(prompt: str):
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'error':'no_groq_key'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = get_groq_headers()
    payload = {"model": CHAT_MODEL, "stream": True,"messages":[{"role":"system","content":get_system_prompt(prompt)},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    # Log more context for debugging
                    logger.warning("Groq stream provider error: status=%s text=%s", resp.status_code, (text.decode(errors='ignore')[:300] + '...') if text else "")
                    yield f"data: {json.dumps({'error':'provider_error','status':resp.status_code,'text': text.decode(errors='ignore')[:300]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for raw_line in resp.aiter_lines():
                    if raw_line.startswith("data: "):
                        data = raw_line[len("data: "):]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        yield f"data: {data}\n\n"
        except Exception as e:
            logger.exception("groq_stream error")
            yield f"data: {json.dumps({'error':'stream_exception','msg':str(e)})}\n\n"
            yield "data: [DONE]\n\n"


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
                    "model": "gpt-4o-mini-tts",
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

# ---------------------------------------------------------
# 🚀 UNIVERSAL MULTIMODAL ENDPOINT — /ask
# ---------------------------------------------------------
@app.post("/ask")
async def ask(
    request: Request,
    prompt: str = Form(...),
    user_id: str = Form("anonymous"),
):
    if not prompt:
        raise HTTPException(400, "prompt is required")

    intent = detect_intent(prompt)

    # 🔑 FIX: normalize base URL (prevents 301 redirects)
    base = str(request.base_url).rstrip("/") + "/"

    async with httpx.AsyncClient(timeout=60.0) as client:

        # ---------- IMAGE ----------
        if intent == "image":
            r = await client.post(base + "image", json={"prompt": prompt})
            return r.json() if r.headers.get("content-type","").startswith("application/json") else {
                "error": "image_non_json",
                "status": r.status_code,
                "body": r.text[:500]
            }

        # ---------- VIDEO ----------
        if intent == "video":
            return {
                "status": "video_requested",
                "message": "Video model integration ready",
                "prompt": prompt
            }

        # ---------- TTS ----------
        if intent == "tts":
            r = await client.post(base + "tts", json={"text": prompt})
            return Response(content=r.content, media_type="audio/mpeg")

        # ---------- CODE ----------
        if intent == "code":
            r = await client.post(
                base + "code",
                json={"prompt": prompt, "user_id": user_id}
            )
            return r.json() if r.headers.get("content-type","").startswith("application/json") else {
                "error": "code_non_json",
                "status": r.status_code,
                "body": r.text[:500]
            }

        # ---------- SEARCH ----------
        if intent == "search":
            r = await client.get(base + "search", params={"q": prompt})
            return r.json() if r.headers.get("content-type","").startswith("application/json") else {
                "error": "search_non_json",
                "status": r.status_code,
                "body": r.text[:500]
            }

        # ---------- DEFAULT → CHAT ----------
        r = await client.post(
            base + "chat",
            json={"prompt": prompt, "user_id": user_id}
        )
        return r.json() if r.headers.get("content-type","").startswith("application/json") else {
            "error": "chat_non_json",
            "status": r.status_code,
            "body": r.text[:500]
        }
        
@app.post("/image")
async def image_gen(request: Request):
    """
    Generate images via OpenAI (DALL·E 3). Requests b64_json but will
    fall back to URL-based results if present. Saves files locally and caches results.
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

    # Check cache first
    cached = get_cached(prompt)
    if cached:
        logger.info("Image cache hit for prompt: %s", prompt[:80])
        return {"cached": True, **cached}

    urls: List[str] = []
    provider_used = None

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not configured; skipping OpenAI provider")
    else:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": samples,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                }
                logger.info("Sending image generation request to OpenAI for prompt: %s", prompt[:120])
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                # If OpenAI returns an error (e.g. unauthorized), this will raise
                r.raise_for_status()
                jr = r.json()
                logger.debug("OpenAI image response keys: %s", list(jr.keys()))
                data_list = jr.get("data", [])
                if not data_list:
                    # Try to detect URL-style response (some accounts/configs)
                    logger.warning("OpenAI returned empty 'data' for prompt: %s", prompt[:120])
                for d in data_list:
                    # Accept either b64_json or url (fallback)
                    b64 = d.get("b64_json")
                    url_field = d.get("url") or d.get("image_url")
                    if b64:
                        if return_base64:
                            urls.append(b64)
                        else:
                            fname = unique_filename("png")
                            try:
                                save_base64_image_to_file(b64, fname)
                                urls.append(local_image_url(request, fname))
                            except Exception as e:
                                logger.exception("Failed to save base64 image to file: %s", str(e))
                                continue
                    elif url_field:
                        # Download the image and save locally so clients get a stable /static URL
                        try:
                            async with httpx.AsyncClient(timeout=30.0) as dl_client:
                                dl = await dl_client.get(url_field)
                                dl.raise_for_status()
                                fname = unique_filename("png")
                                path = os.path.join(IMAGES_DIR, fname)
                                with open(path, "wb") as f:
                                    f.write(dl.content)
                                urls.append(local_image_url(request, fname))
                        except Exception:
                            logger.exception("Failed to download image from provider URL: %s", url_field)
                            continue
                    else:
                        logger.warning("No b64_json or url for one result item: %s", d.keys())

            provider_used = "dalle3"
        except httpx.HTTPStatusError as exc:
            logger.exception("OpenAI returned HTTP error for image generation: %s", getattr(exc.response, "text", ""))
        except Exception as e:
            logger.exception("OpenAI DALL-E 3 generation failed: %s", str(e))

    if not urls:
        logger.error("Image generation returned no images for prompt: %s", prompt[:120])
        raise HTTPException(500, "Image generation failed or returned no data")

    # Cache results (cache the final URLs or base64 list)
    try:
        cache_result(prompt, provider_used or "openai", urls)
    except Exception:
        logger.exception("Failed to cache image result")

    return {"provider": provider_used or "openai", "images": urls}

@app.post("/image/stream")
async def image_stream(request: Request):
    """
    Stream progress to the client while generating images.
    Sends SSE messages with JSON payloads and final 'done' event.
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
                "model": "gpt-image-3",
                "prompt": prompt,
                "n": samples,
                "size": "1024x1024",
                "response_format": "b64_json"
            }

            yield "data: " + json.dumps({"status": "request", "message": "Sending to OpenAI"}) + "\n\n"

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
                url_field = d.get("url") or d.get("image_url")
                if b64:
                    if return_base64:
                        urls.append(b64)
                    else:
                        fname = unique_filename("png")
                        try:
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))
                        except Exception:
                            logger.exception("Failed saving streamed b64 image")
                elif url_field:
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as dl_client:
                            dl = await dl_client.get(url_field)
                            dl.raise_for_status()
                            fname = unique_filename("png")
                            with open(os.path.join(IMAGES_DIR, fname), "wb") as f:
                                f.write(dl.content)
                            urls.append(local_image_url(request, fname))
                    except Exception:
                        logger.exception("Failed to download streamed image URL")
                else:
                    logger.warning("Stream result missing both b64_json and url: %s", d.keys())

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
                    save_base64_image_to_file(b64, fname)
                    urls.append(local_image_url(request, fname))
    except Exception:
        logger.exception("img2img DALL-E edit failed")
        raise HTTPException(400, "img2img failed")

    return {"provider": "dalle3-edit", "images": urls}
    
# ---------- TTS ----------
@app.post("/tts")
async def text_to_speech(request: Request):
    """
    Convert text to speech using OpenAI TTS (gpt-4o-mini-tts).
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
        "model": "gpt-4o-mini-tts",
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
        "model": "gpt-4o-mini-tts",
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
    try:
        np_img = np.array(img).reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=0).fit(np_img)
        colors = [tuple(map(int, c)) for c in kmeans.cluster_centers_]
        hex_colors = ['#%02x%02x%02x' % c for c in colors]
    except Exception:
        hex_colors = []

    # ----- 2. Object detection (pretrained ResNet) -----
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
    except Exception:
        object_label = None

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
    language = body.get("language", "python")
    user_id = body.get("user_id", "anonymous")

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no groq key")

    # Build user-focused prompt with context
    contextual_prompt = build_contextual_prompt(user_id, f"Write a complete, well-documented {language} program for the following request:\n\n{prompt}")

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": contextual_prompt}, {"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = get_groq_headers()
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers=headers,
                              json=payload)
        if r.status_code != 200:
            logger.warning("Groq /code returned status=%s text=%s", r.status_code, (r.text[:400] + '...') if r.text else "")
            r.raise_for_status()
        return r.json()

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
        "model": (None, "gpt-4o-mini-transcribe"),
        # You can also use: "whisper-1"
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
