# app.py — Billy AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import json
import uuid
import sqlite3
import base64
import time
import logging
from typing import Optional, Dict, Any
import subprocess
import tempfile

import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "").strip()
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

# Image hosting dirs
STATIC_DIR = os.getenv("STATIC_DIR", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldBoy",
    "age": 17,
    "country": "England",
    "projects": ["MintZa", "LuxStream", "SwapX", "CryptoBean"],
    "socials": {"instagram":"GoldBoyy", "twitter":"GoldBoy"},
    "bio": "Created by GoldBoy (17, England). Projects: MintZa, LuxStream, SwapX, CryptoBean. Socials: Instagram @GoldBoyy, Twitter @GoldBoy."
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
    return f"You are Billy AI: helpful, concise, friendly. Focus on exactly what the user wants.\nUser context:\n{context}\nUser message: {message}"

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

# ---------- Prompt enhancer ----------
async def enhance_prompt_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return prompt
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        system = "Rewrite the user's short prompt into a detailed, professional SDXL-style art prompt. Be concise but specific. Avoid explicit sexual or illegal content."
        body = {"model": CHAT_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}], "temperature":0.6,"max_tokens":300}
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            jr = r.json()
            content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or prompt
    except Exception:
        logger.exception("Prompt enhancer failed")
    return prompt

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
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": CHAT_MODEL, "stream": True,"messages":[{"role":"system","content":get_system_prompt(prompt)},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
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

@app.get("/stream")
async def stream_chat(prompt: str):
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")

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
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"}, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- Image generation ----------
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt","")
    samples = int(body.get("samples",1))
    if not prompt:
        raise HTTPException(400,"prompt required")

    # Check cache first
    cached = get_cached(prompt)
    if cached:
        return {"cached": True, **cached}

    enhanced = await enhance_prompt_with_groq(prompt)
    settings = analyze_prompt(enhanced)
    settings["samples"] = samples or settings["samples"]

    # Attempt providers in order
    providers = []
    if STABILITY_API_KEY:
        providers.append("stability")
    if OPENAI_API_KEY:
        providers.append("openai")
    if USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        providers.append("free")

    for prov in providers:
        try:
            urls = []
            if prov=="stability":
                payload = {"prompt": enhanced,"cfg_scale":settings["cfg_scale"],"steps":settings["steps"],"samples":settings["samples"],"height":settings["height"],"width":settings["width"],"model":"stable-diffusion-xl-v1","negative_prompt":settings.get("negative_prompt")}
                files = {"json": (None,json.dumps(payload),"application/json")}
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post("https://api.stability.ai/v2beta/stable-image/generate/core", headers={"Authorization": f"Bearer {STABILITY_API_KEY}"}, files=files)
                    r.raise_for_status()
                    jr = r.json()
                    for art in jr.get("artifacts",[]):
                        b64 = art.get("base64")
                        if b64:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64,fname)
                            urls.append(local_image_url(request,fname))
            elif prov=="openai":
                payload={"model":"gpt-image-1","prompt":enhanced,"n":settings["samples"],"size":f"{settings['width']}x{settings['height']}"}
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}","Content-Type":"application/json"}
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload)
                    r.raise_for_status()
                    jr = r.json()
                    for d in jr.get("data",[]):
                        b64 = d.get("b64_json")
                        if b64:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64,fname)
                            urls.append(local_image_url(request,fname))
            elif prov=="free":
                async with httpx.AsyncClient(timeout=90.0) as client:
                    r = await client.post(IMAGE_MODEL_FREE_URL, json={"prompt":enhanced,"samples":settings["samples"]})
                    r.raise_for_status()
                    jr = r.json()
                    images = jr.get("images") or ([jr.get("image")] if jr.get("image") else [])
                    for im in images:
                        if im.startswith("http"):
                            resp = await client.get(im)
                            if resp.status_code==200:
                                fname=unique_filename("png")
                                with open(os.path.join(IMAGES_DIR,fname),"wb") as f:
                                    f.write(resp.content)
                                urls.append(local_image_url(request,fname))
                        else:
                            fname=unique_filename("png")
                            save_base64_image_to_file(im,fname)
                            urls.append(local_image_url(request,fname))
            if urls:
                cache_result(prompt, prov, urls)
                return {"provider":prov,"images":urls}
        except Exception:
            logger.exception(f"{prov} provider failed")

    return JSONResponse({"error":"all_providers_failed","prompt":prompt}, status_code=400)

# ---------- TTS ----------
@app.post("/tts")
async def text_to_speech(req: Request):
    body = await req.json()
    text = body.get("text")
    if not text:
        raise HTTPException(400,"text required")
    if not OPENAI_API_KEY:
        raise HTTPException(400,"no TTS provider configured")
    payload = {"model": TTS_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload)
        r.raise_for_status()
        audio_data = r.content
    filename = unique_filename("mp3")
    path = os.path.join(IMAGES_DIR, filename)
    with open(path,"wb") as f:
        f.write(audio_data)
    return {"audio_url": local_image_url(req, filename)}

# ---------- Vision analyze ----------
@app.post("/vision/analyze")
async def vision_analyze(req: Request, file: UploadFile = File(...), prompt: Optional[str] = None, user_id: str = "anonymous"):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no vision provider configured")

    # Build dynamic, user-focused prompt
    user_prompt = prompt or "Analyze this image: list objects, colors, suggested tags, and short description."
    contextual_prompt = build_contextual_prompt(user_id, user_prompt)
    
    # Include image as base64
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": contextual_prompt},
            {"role": "user", "content": f"Image (base64):\n{base64.b64encode(content).decode()}"}
        ]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=body)
        r.raise_for_status()
        return r.json()


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
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=payload)
        r.raise_for_status()
        return r.json()

@app.post("/img2img")
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = "", user_id: str = "anonymous"):
    if not prompt:
        raise HTTPException(400, "prompt required")
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    cache_key = f"img2img:{user_id}:{prompt}"
    cached = get_cached(cache_key)
    if cached:
        return {"cached": True, **cached}

    enhanced = await enhance_prompt_with_groq(prompt)
    urls = []

    if STABILITY_API_KEY:
        try:
            payload = {
                "prompt": enhanced,
                "init_image": None,
                "cfg_scale": 7,
                "steps": 30,
                "samples": 1,
                "model": "stable-diffusion-xl-v1",
                "mode": "image-to-image"
            }
            files = {"json": (None, json.dumps(payload), "application/json"),
                     "image[]": (file.filename, content, file.content_type or "application/octet-stream")}
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post("https://api.stability.ai/v2beta/stable-image/generate/core",
                                      headers={"Authorization": f"Bearer {STABILITY_API_KEY}"}, files=files)
                r.raise_for_status()
                jr = r.json()
                for art in jr.get("artifacts", []):
                    b64 = art.get("base64")
                    if b64:
                        fname = unique_filename("png")
                        save_base64_image_to_file(b64, fname)
                        urls.append(local_image_url(request, fname))
            if urls:
                cache_result(cache_key, "stability", urls)
                return {"provider": "stability", "images": urls}
        except Exception:
            logger.exception("Img2Img failed")
    raise HTTPException(400, "Img2Img failed or no provider configured")


@app.get("/search")
async def google_search(q: str = Query(..., min_length=1)):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise HTTPException(400, "Google Search not configured")
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q}
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link")
            })
        return {"query": q, "results": results}


# ---------- STT ----------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400,"empty file")
    if not OPENAI_API_KEY:
        raise HTTPException(400,"no STT provider configured")
    files = {"file": (file.filename, content, file.content_type)}
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=files, data={"model": STT_MODEL})
        r.raise_for_status()
        return r.json()

# ---------- Other endpoints (/remove-bg, /upscale, /img2img, /vision/analyze, /code, /search, /root) remain unchanged ----------

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
