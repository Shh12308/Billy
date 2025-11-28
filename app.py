import os
import json
import sqlite3
import base64
import time
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy-server")

app = FastAPI(title="Billy AI Full Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Load keys from environment -----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()          # Groq (chat/tts/stt)
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()  # Stability.ai (images)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()        # OpenAI images fallback (gpt-image-1)
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "").strip()  # optional self-hosted/free provider
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

ENABLE_EMPATHY = os.getenv("ENABLE_EMPATHY", "false").lower() in ("1", "true", "yes")
ENABLE_DEBATE = os.getenv("ENABLE_DEBATE", "false").lower() in ("1", "true", "yes")

# ---------- SYSTEM / CREATOR PROMPTS ----------
SYSTEM_PROMPT_BASE = """You are Billy AI: helpful, concise, friendly.
If a user asks about your creator, respond with the official creator profile provided in CREATOR_INFO.
Do NOT claim you made yourself."""
SYSTEM_PROMPT_EMPATHY = "You are empathetic, patient, and supportive."
SYSTEM_PROMPT_DEBATE = "You are analytic, structured, evidence-based. Argue both sides fairly."

CREATOR_INFO = {
    "name": "GoldBoy",
    "age": 17,
    "country": "England",
    "projects": ["MintZa", "LuxStream", "SwapX", "CryptoBean"],
    "socials": {"instagram":"GoldBoyy", "twitter":"GoldBoy"},
    "bio": "Created by GoldBoy (17, England). Projects: MintZa, LuxStream, SwapX, CryptoBean. Socials: Instagram @GoldBoyy, Twitter @GoldBoy."
}

def get_system_prompt() -> str:
    parts = [SYSTEM_PROMPT_BASE]
    if ENABLE_EMPATHY:
        parts.append(SYSTEM_PROMPT_EMPATHY)
    if ENABLE_DEBATE:
        parts.append(SYSTEM_PROMPT_DEBATE)
    # include creator brief for consistency
    parts.append("Creator profile: " + CREATOR_INFO["bio"])
    return " ".join(parts)

def detect_creator_question(prompt: str):
    keywords = ["who made you", "who created you", "your creator", "who built you", "owner", "who made this", "who coded you"]
    p = prompt.lower()
    return any(k in p for k in keywords)

# ---------- SQLITE helpers (KG & Memory) ----------
def ensure_kg_db():
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        created_at REAL
    );
    """)
    conn.commit()
    conn.close()

def ensure_memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        key TEXT,
        value TEXT,
        updated_at REAL
      )
    """)
    conn.commit()
    conn.close()

ensure_kg_db()
ensure_memory_db()

def add_kg_node(title: str, content: str):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO nodes (title, content, created_at) VALUES (?, ?, ?)", (title, content, time.time()))
    conn.commit()
    conn.close()

def query_kg(query: str, limit: int = 5):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, content FROM nodes WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?", (f"%{query}%", f"%{query}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [{"title": r[0], "content": r[1]} for r in rows]

# ---------- GROQ STREAMING (chat) ----------
async def groq_stream(prompt: str):
    """
    Stream Groq chat completions as SSE. Each yielded line is "data: <json>\n\n".
    If GROQ_API_KEY is missing, return an error chunk.
    """
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'error': 'no_groq_key'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if detect_creator_question(prompt):
        # small convenience: directly return CREATOR_INFO as a simple message chunk
        payload = {"choices":[{"delta":{"content":CREATOR_INFO["bio"]}}]}
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    logger.error("Groq stream error status %s: %s", resp.status_code, text[:300])
                    yield f"data: {json.dumps({'error':'provider_error','status':resp.status_code,'text': text.decode(errors='ignore')[:300]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    # common pattern: "data: <json>"
                    if raw_line.startswith("data: "):
                        data = raw_line[len("data: "):]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        # forward JSON chunk as SSE data
                        yield f"data: {data}\n\n"
                    else:
                        # sometimes provider sends plain lines; try to wrap them
                        try:
                            # attempt to parse then forward
                            _ = json.loads(raw_line)
                            yield f"data: {raw_line}\n\n"
                        except Exception:
                            # ignore unparseable lines
                            continue
        except Exception as e:
            logger.exception("Error during groq_stream")
            yield f"data: {json.dumps({'error':'stream_exception','msg':str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

@app.get("/stream")
async def stream_chat(prompt: str):
    """
    SSE endpoint. Frontend connects with EventSource(`${API_BASE}/stream?prompt=...`)
    """
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")

# ---------- NON-STREAM CHAT (fallback) ----------
@app.post("/chat")
async def chat_endpoint(req: Request):
    payload = await req.json()
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not configured")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    body = {"model": CHAT_MODEL, "messages":[{"role":"system","content":get_system_prompt()},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=body, timeout=30.0)
        r.raise_for_status()
        return r.json()

# ---------- IMAGE GENERATION (stability -> openai -> free provider) ----------
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")

    # 1) Try Stability.ai if key provided
    if STABILITY_API_KEY:
        try:
            url = "https://api.stability.ai/v2beta/stable-image/generate/core"

            headers = {
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json",
            }

            form = {
                "prompt": (None, prompt),
                "output_format": (None, "png"),
                "mode": (None, "text-to-image"),
                "cfg_scale": (None, "7"),
                "width": (None, "512"),
                "height": (None, "512"),
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(url, headers=headers, files=form)

            if r.status_code == 200:
                jr = r.json()
                artifacts = jr.get("artifacts", [])
                if artifacts:
                    b64 = artifacts[0].get("base64")
                    if b64:
                        return {"image": b64}
            else:
                logger.warning("Stability returned %s: %s", r.status_code, r.text[:300])

        except Exception as e:
            logger.exception("Stability image error — falling back: %s", str(e))

    # 2) Try OpenAI images (gpt-image-1) if OPENAI_API_KEY present
    if OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"model": "gpt-image-1", "prompt": prompt, "size": "512x512"}

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=payload)

            if r.status_code == 200:
                jr = r.json()
                data = jr.get("data", [])
                if data and data[0].get("b64_json"):
                    return {"image": data[0]["b64_json"]}
            else:
                logger.warning("OpenAI image generation returned %s: %s", r.status_code, r.text[:300])

        except Exception as e:
            logger.exception("OpenAI image error — falling back: %s", str(e))

    # 3) Try free provider if configured
    if USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(IMAGE_MODEL_FREE_URL, json={"prompt": prompt})

            if r.status_code == 200:
                jr = r.json()
                if jr.get("image"):
                    return {"image": jr["image"]}

                if jr.get("url"):
                    resp = await client.get(jr["url"])
                    if resp.status_code == 200:
                        b64 = base64.b64encode(resp.content).decode()
                        return {"image": b64}

            else:
                logger.warning("Free image provider returned %s", r.status_code)

        except Exception as e:
            logger.exception("Free image provider error: %s", str(e))

    # nothing worked
    return JSONResponse({"error": "no_image_provider_available_or_all_failed"}, status_code=400)

# ---------- TTS (text-to-speech) ----------
@app.post("/tts")
async def tts_endpoint(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not text:
        raise HTTPException(400, "text required")

    # Prefer Groq if configured
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/audio/speech"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {"model": TTS_MODEL, "voice": "alloy", "input": text}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=body)
                r.raise_for_status()
                jr = r.json()
                # expected jr['audio'] base64
                audio_b64 = jr.get("audio") or jr.get("data", {}).get("audio")
                if audio_b64:
                    return {"audio": audio_b64}
        except Exception:
            logger.exception("Groq TTS error (fallthrough)")

    # Fallback: not implemented — return clear error
    raise HTTPException(400, "No TTS provider configured or providers failed")

# ---------- STT (speech-to-text) ----------
@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    """
    Upload file using multipart/form-data.
    We convert to base64 and send JSON to provider if provider expects base64 payloads.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty file")

    # Prefer Groq if configured
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": STT_MODEL, "file": base64.b64encode(audio_bytes).decode(), "format": "base64"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception:
            logger.exception("Groq STT error")

    raise HTTPException(400, "No STT provider configured or provider failed")

# ---------- NLU / Intent classification ----------
@app.post("/nlp/classify")
async def nlp_classify(req: Request):
    body = await req.json()
    text = body.get("text", "")
    use_llm = body.get("use_llm", False)
    if not text:
        raise HTTPException(400, "text required")

    # simple heuristic
    lower = text.lower()
    if any(w in lower for w in ("draw", "image", "picture", "photo", "generate", "make an image", "create image")):
        intent = "image"
    elif any(w in lower for w in ("say", "speak", "read aloud", "tts", "voice")):
        intent = "tts"
    elif any(w in lower for w in ("summarize", "shorten", "tl;dr")):
        intent = "summarize"
    else:
        intent = "chat"

    if use_llm and GROQ_API_KEY:
        try:
            prompt = f"Classify the user's intent into one of: image, tts, summarize, chat. Respond only JSON: {{\"intent\": <intent>, \"confidence\": 0.0}}. Text: '''{text}'''"
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0}
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                jr = r.json()
                content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
                try:
                    parsed = json.loads(content)
                    return {"intent": parsed.get("intent", intent), "confidence": parsed.get("confidence", 0.0), "source":"llm"}
                except Exception:
                    return {"intent": intent, "confidence": 0.5, "source":"heuristic"}
        except Exception:
            logger.exception("NLU LLM fallback failed")

    return {"intent": intent, "confidence": 0.6, "source":"heuristic"}

# ---------- Knowledge graph endpoints ----------
@app.post("/kg/add")
async def kg_add(req: Request):
    body = await req.json()
    title = body.get("title", "")
    content = body.get("content", "")
    if not title or not content:
        raise HTTPException(400, "title and content required")
    add_kg_node(title, content)
    return {"status": "ok"}

@app.get("/kg/query")
async def kg_query(q: str, limit: int = 5):
    rows = query_kg(q, limit)
    return {"results": rows}

# ---------- Memory endpoints ----------
@app.post("/memory/save")
async def save_memory(req: Request):
    body = await req.json()
    user = body.get("user_id", "guest")
    key = body.get("key")
    value = body.get("value")
    if not key or value is None:
        raise HTTPException(400, "key and value required")
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO memory (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)", (user, key, json.dumps(value), time.time()))
    conn.commit()
    conn.close()
    return {"status":"ok"}

@app.get("/memory/get")
async def get_memory(user_id: str, key: str):
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT value FROM memory WHERE user_id=? AND key=? ORDER BY updated_at DESC LIMIT 1", (user_id, key))
    row = cur.fetchone()
    conn.close()
    return {"value": json.loads(row[0]) if row else None}

# ---------- Translate scaffold ----------
@app.post("/translate")
async def translate(req: Request):
    body = await req.json()
    text = body.get("text", "")
    target = body.get("target", "en")
    if not text:
        raise HTTPException(400, "text required")
    if GROQ_API_KEY:
        try:
            prompt = f"Translate the following text to {target}: '''{text}'''"
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0}
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                jr = r.json()
                translated = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"text": translated}
        except Exception:
            logger.exception("Translate via Groq failed")
    # fallback: return original
    return {"text": text}

# ---------- Summarize ----------
@app.post("/summarize")
async def summarize(req: Request):
    body = await req.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "text required")
    if GROQ_API_KEY:
        try:
            prompt = f"Summarize the following text briefly:\n\n'''{text}'''"
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0.3}
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                jr = r.json()
                return {"summary": jr.get("choices", [{}])[0].get("message", {}).get("content", "")}
        except Exception:
            logger.exception("Summarize via Groq failed")
    raise HTTPException(400, "No LLM provider configured")

# ---------- Simple health check ----------
@app.get("/")
async def root():
    return {
        "status":"ok",
        "providers":{
            "groq": bool(GROQ_API_KEY),
            "stability": bool(STABILITY_API_KEY),
            "openai_images": bool(OPENAI_API_KEY),
            "free_image_provider": bool(USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL)
        },
        "creator": CREATOR_INFO["bio"]
    }

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="info")
