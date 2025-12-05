# app.py — Billy AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import json
import uuid
import sqlite3
import base64
import time
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

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
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
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
    
@app.get("/stream")
async def stream(prompt: str, user_id: str = "anonymous"):
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY missing")

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": build_contextual_prompt(user_id, prompt)},
            {"role": "user", "content": prompt}
        ]
    }

    async def event_generator():
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload,
            ) as response:

                if response.status_code != 200:
                    text = await response.aread()
                    yield f"data: {json.dumps({'error': 'provider_error', 'text': text.decode()[:300]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        # Proper SSE formatting
                        yield f"data: {data}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error':'exception','msg':str(e)})}\n\n"
            yield "data: [DONE]\n\n"

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

@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    samples = int(body.get("samples", 1))
    return_base64 = body.get("base64", False)

    if not prompt:
        raise HTTPException(400, "prompt required")

    cached = get_cached(prompt)
    if cached:
        return {"cached": True, **cached}

    urls = []
    provider_used = None

    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {
                    "model": "gpt-image-3",
                    "prompt": prompt,
                    "n": samples,
                    "size": "1024x1024"
                }
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json=payload
                )
                r.raise_for_status()
                jr = r.json()
                for d in jr.get("data", []):
                    b64 = d.get("b64_json")
                    if b64:
                        if return_base64:
                            urls.append(b64)
                        else:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))
            provider_used = "dalle3"
        except Exception:
            logger.exception("OpenAI DALL-E 3 generation failed")

    if not urls:
        raise HTTPException(500, "All image providers failed")

    cache_result(prompt, provider_used, urls)
    return {"provider": provider_used, "images": urls}

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
ELEVENLABS_VOICE = "Bella"

@app.post("/tts")
async def text_to_speech(req: Request):
    body = await req.json()
    text = body.get("text")
    if not text:
        raise HTTPException(400, "text required")
    if not ELEVENLABS_API_KEY:
        raise HTTPException(400, "no ElevenLabs API key configured")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"ElevenLabs TTS error: {r.text}")

        audio_data = r.content

    # Save audio locally
    filename = f"{int(time.time())}-{uuid.uuid4().hex[:10]}.mp3"
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(audio_data)

    return {"audio_url": local_image_url(req, filename)}
    

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
    if not ELEVENLABS_API_KEY:
        raise HTTPException(400, "no ElevenLabs API key configured")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": (file.filename, content, file.content_type or "audio/mpeg")}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, files=files)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"ElevenLabs STT error: {r.text}")
        data = r.json()

    # ElevenLabs returns the transcription in `text` field
    return {"transcription": data.get("text", "")}

# ---------- Other endpoints (/remove-bg, /upscale, /img2img, /vision/analyze, /code, /search, /root) remain unchanged ----------

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
