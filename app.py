# app.py — Billy AI server with SDXL, image hosting, remove-bg, upscale, img2img, prompt enhancer
import os
import io
import json
import uuid
import sqlite3
import base64
import time
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-server")

app = FastAPI(title="Billy AI Full Server (SDXL-ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ENV / KEYS ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "").strip()  # optional free provider base url
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

# Where we host images on the server filesystem
STATIC_DIR = os.getenv("STATIC_DIR", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- small system/creator metadata ----------
CREATOR_INFO = {
    "name": "GoldBoy",
    "age": 17,
    "country": "England",
    "projects": ["MintZa", "LuxStream", "SwapX", "CryptoBean"],
    "socials": {"instagram":"GoldBoyy", "twitter":"GoldBoy"},
    "bio": "Created by GoldBoy (17, England). Projects: MintZa, LuxStream, SwapX, CryptoBean. Socials: Instagram @GoldBoyy, Twitter @GoldBoy."
}

def get_system_prompt() -> str:
    return f"You are Billy AI: helpful, concise, friendly. Creator: {CREATOR_INFO['bio']}"

# ---------- sqlite helpers (kg & memory) ----------
def ensure_db(path: str, schema_sql: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(schema_sql)
    conn.commit()
    conn.close()

ensure_db(KG_DB_PATH, """
CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  content TEXT,
  created_at REAL
);
""")

ensure_db(MEMORY_DB, """
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  key TEXT,
  value TEXT,
  updated_at REAL
);
""")

# ---------- Utility helpers ----------
def unique_filename(ext="png"):
    return f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"

def local_image_url(request: Request, filename: str):
    # build fully qualified URL using request.base_url
    return str(request.base_url).rstrip("/") + f"/static/images/{filename}"

def save_base64_image_to_file(b64: str, filename: str) -> str:
    path = os.path.join(IMAGES_DIR, filename)
    img_bytes = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

async def enhance_prompt_with_groq(prompt: str) -> str:
    """
    If GROQ_API_KEY available, ask an LLM to rewrite the prompt into an SDXL-optimized prompt.
    If not available or fails, return original prompt.
    """
    if not GROQ_API_KEY:
        return prompt
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        system = "Rewrite the user's short prompt into a detailed, professional SDXL-style art prompt. Be concise but specific. Avoid explicit sexual or illegal content."
        body = {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.6,
            "max_tokens": 300
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            jr = r.json()
            content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content.strip()
    except Exception as e:
        logger.exception("Prompt enhancer failed")
    return prompt

# ---------- Prompt analyzer for heuristics ----------
def analyze_prompt(prompt: str):
    p = prompt.lower()
    settings = {
        "model": "stable-diffusion-xl-v1",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg_scale": 7,
        "samples": 1,
        "negative_prompt": "nsfw, nudity, watermark, lowres, text, logo"
    }
    # size heuristics
    if any(w in p for w in ("wallpaper","background","poster")):
        settings["width"], settings["height"] = 1920, 1080
    if any(w in p for w in ("landscape","city","wide","panorama")):
        settings["width"], settings["height"] = 1280, 720
    # samples numeric mention
    for token in p.split():
        if token.isnumeric():
            n = int(token)
            if 1 <= n <= 6:
                settings["samples"] = n
    return settings

# ---------- SSE / groq streaming chat (kept minimal) ----------
async def groq_stream(prompt: str):
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'error':'no_groq_key'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": CHAT_MODEL, "stream": True, "messages":[{"role":"system","content":get_system_prompt()},{"role":"user","content":prompt}]}

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    yield f"data: {json.dumps({'error':'provider_error','status':resp.status_code,'text': text.decode(errors='ignore')[:300]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
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

@app.get("/stream")
async def stream_chat(prompt: str):
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")

# ---------- CHAT (non-stream) ----------
@app.post("/chat")
async def chat_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no groq key configured")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type":"application/json"}
    payload = {"model": CHAT_MODEL, "messages":[{"role":"system","content":get_system_prompt()},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- IMAGE generation (SDXL primary -> OpenAI fallback -> free provider) ----------
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    samples = int(body.get("samples", 1))
    model_hint = body.get("model", "").lower()

    if not prompt:
        raise HTTPException(400, "prompt required")

    # Enhance prompt with Groq if possible
    enhanced = await enhance_prompt_with_groq(prompt)

    # analyze for sizes, samples, negative prompt
    settings = analyze_prompt(enhanced)
    settings["samples"] = samples or settings["samples"]

    # ------------------ 1) Stability SDXL primary ------------------
    if STABILITY_API_KEY:
        try:
            # Stability SDXL expects multipart with a 'json' part containing options
            s_payload = {
                "prompt": enhanced,
                "cfg_scale": settings["cfg_scale"],
                "steps": settings["steps"],
                "samples": settings["samples"],
                "height": settings["height"],
                "width": settings["width"],
                # model name may vary by account — we use stable-diffusion-xl-v1 as default
                "model": "stable-diffusion-xl-v1",
                "negative_prompt": settings.get("negative_prompt")
            }
            files = {"json": (None, json.dumps(s_payload), "application/json")}
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/core",
                    headers={"Authorization": f"Bearer {STABILITY_API_KEY}"},
                    files=files
                )
                # If 200, expect JSON with 'artifacts' list and base64 images
                if r.status_code == 200:
                    jr = r.json()
                    artifacts = jr.get("artifacts", [])
                    urls = []
                    for art in artifacts:
                        b64 = art.get("base64") or art.get("b64_json") or art.get("b64")
                        if not b64:
                            continue
                        fname = unique_filename("png")
                        save_base64_image_to_file(b64, fname)
                        urls.append(local_image_url(request, fname))
                    if urls:
                        return {"provider": "stability", "images": urls, "model": s_payload["model"]}
                else:
                    # log and fallthrough
                    logger.warning("Stability returned %s: %s", r.status_code, r.text[:400])
        except Exception:
            logger.exception("Stability SDXL generation failed (falling back)")

    # ------------------ 2) OpenAI Images fallback ------------------
    if OPENAI_API_KEY:
        try:
            o_payload = {"model": "gpt-image-1", "prompt": enhanced, "n": settings["samples"], "size": f"{settings['width']}x{settings['height']}"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post("https://api.openai.com/v1/images/generations", headers=headers, json=o_payload)
                if r.status_code == 200:
                    jr = r.json()
                    data = jr.get("data", [])
                    urls = []
                    for d in data:
                        b64 = d.get("b64_json")
                        if b64:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))
                    if urls:
                        return {"provider": "openai", "images": urls, "model": "gpt-image-1"}
                else:
                    logger.warning("OpenAI images returned %s: %s", r.status_code, r.text[:400])
        except Exception:
            logger.exception("OpenAI image fallback failed")

    # ------------------ 3) free provider ------------------
    if USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(IMAGE_MODEL_FREE_URL, json={"prompt": enhanced, "samples": settings["samples"]})
                r.raise_for_status()
                jr = r.json()
                images = jr.get("images") or ([jr.get("image")] if jr.get("image") else [])
                saved_urls = []
                for im in images:
                    # accept either base64 or url
                    if im.startswith("http://") or im.startswith("https://"):
                        # fetch and save
                        resp = await client.get(im)
                        if resp.status_code == 200:
                            fname = unique_filename("png")
                            with open(os.path.join(IMAGES_DIR, fname), "wb") as f:
                                f.write(resp.content)
                            saved_urls.append(local_image_url(request, fname))
                    else:
                        # assume base64
                        fname = unique_filename("png")
                        save_base64_image_to_file(im, fname)
                        saved_urls.append(local_image_url(request, fname))
                if saved_urls:
                    return {"provider": "free", "images": saved_urls}
        except Exception:
            logger.exception("Free image provider failed")

    return JSONResponse({"error": "all_providers_failed", "prompt": prompt}, status_code=400)

# ---------- img2img endpoint (multipart) ----------
@app.post("/img2img")
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = ""):
    """
    Upload a source image and a prompt. The server will call Stability img2img (if available).
    Returns hosted image URLs.
    """
    if not prompt:
        raise HTTPException(400, "prompt required")
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    enhanced = await enhance_prompt_with_groq(prompt)

    # prefer Stability
    if STABILITY_API_KEY:
        try:
            payload = {
                "prompt": enhanced,
                "init_image": None,  # we will send multipart with image file separately
                "cfg_scale": 7,
                "steps": 30,
                "samples": 1,
                "model": "stable-diffusion-xl-v1",
                "mode": "image-to-image"
            }
            files = {
                "json": (None, json.dumps(payload), "application/json"),
                "image[]": (file.filename, content, file.content_type or "application/octet-stream")
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post("https://api.stability.ai/v2beta/stable-image/generate/core", headers={"Authorization": f"Bearer {STABILITY_API_KEY}"}, files=files)
                if r.status_code == 200:
                    jr = r.json()
                    arts = jr.get("artifacts", [])
                    urls = []
                    for art in arts:
                        b64 = art.get("base64")
                        if b64:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))
                    if urls:
                        return {"provider":"stability","images":urls}
                else:
                    logger.warning("Stability img2img returned %s: %s", r.status_code, r.text[:400])
        except Exception:
            logger.exception("Stability img2img failed")

    raise HTTPException(400, "img2img failed or no provider configured")

# ---------- remove-bg ----------
@app.post("/remove-bg")
async def remove_bg(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    # Use free provider if configured to do remove-bg
    if USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(f"{IMAGE_MODEL_FREE_URL.rstrip('/')}/remove-bg", json={"image_base64": base64.b64encode(content).decode()})
                r.raise_for_status()
                jr = r.json()
                b64 = jr.get("image") or jr.get("result")
                if b64:
                    fname = unique_filename("png")
                    save_base64_image_to_file(b64, fname)
                    return {"image": local_image_url(request, fname)}
        except Exception:
            logger.exception("Free provider remove-bg failed")

    # no provider configured
    raise HTTPException(501, "Background removal not configured. Set IMAGE_MODEL_FREE_URL and enable USE_FREE_IMAGE_PROVIDER")

# ---------- upscale (super-resolution) ----------
@app.post("/upscale")
async def upscale(request: Request, file: UploadFile = File(...), scale: int = 4):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    # Use free provider if available
    if USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{IMAGE_MODEL_FREE_URL.rstrip('/')}/upscale", json={"image_base64": base64.b64encode(content).decode(), "scale": scale})
                r.raise_for_status()
                jr = r.json()
                b64 = jr.get("image") or jr.get("result")
                if b64:
                    fname = unique_filename("png")
                    save_base64_image_to_file(b64, fname)
                    return {"image": local_image_url(request, fname), "scale": scale}
        except Exception:
            logger.exception("Free provider upscale failed")

    raise HTTPException(501, "Upscale provider not configured. Set IMAGE_MODEL_FREE_URL and enable USE_FREE_IMAGE_PROVIDER")

# ---------- vision analyze (scaffold) ----------
@app.post("/vision/analyze")
async def vision_analyze(file: UploadFile = File(...), prompt: Optional[str] = None):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no vision provider configured (GROQ)")
    text = prompt or "Analyze this image: list objects, colors, suggested tags, and short description."
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role":"system","content":get_system_prompt()},
            {"role":"user","content": text + "\n\nImage (base64):\n" + base64.b64encode(content).decode()}
        ]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type":"application/json"}, json=body)
        r.raise_for_status()
        return r.json()

# ---------- code endpoint (scaffold) ----------
@app.post("/code")
async def code_gen(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    language = body.get("language", "python")
    if not prompt:
        raise HTTPException(400, "prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no groq key")
    code_prompt = f"Write a complete, well-documented {language} program for the following request:\n\n{prompt}"
    payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":code_prompt}], "temperature":0.1}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type":"application/json"}, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- health ----------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "providers": {
            "groq": bool(GROQ_API_KEY),
            "stability": bool(STABILITY_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "free_image_provider": bool(USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL)
        },
        "creator": CREATOR_INFO["bio"]
    }

# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
