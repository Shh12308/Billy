# app.py — Billy AI server (full-featured, extensible)
# - Streaming chat (Groq streaming JSON -> SSE)
# - Image generation via OpenAI (gpt-image-1) OR free provider placeholder
# - TTS / STT (Groq endpoints if available)
# - NLU endpoint + LLM-assisted fallback
# - Lightweight Knowledge Graph (SQLite) & retrieval
# - Personalization memory APIs
# - Multilingual / translate helper (scaffold)
# - Debate / empathy modes
# - Clear env toggles, logging, and error handling

import os
import json
import sqlite3
import base64
import time
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
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

# Models / endpoints
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")         # Use for chat/tts/stt if available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")     # Use for image generation (gpt-image-1)
KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
IMAGE_MODEL_OPENAI = os.getenv("IMAGE_MODEL_OPENAI", "gpt-image-1")
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "")  # optional free provider endpoint
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

ENABLE_EMPATHY = os.getenv("ENABLE_EMPATHY", "false").lower() in ("1", "true", "yes")
ENABLE_DEBATE = os.getenv("ENABLE_DEBATE", "false").lower() in ("1", "true", "yes")
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

# Small helper system prompts
SYSTEM_PROMPT_BASE = """You are Billy AI: helpful, concise, friendly.
If a user asks about your creator, respond with the official creator profile provided in CREATOR_INFO.
Do NOT claim you made yourself."""
SYSTEM_PROMPT_EMPATHY = "You are empathetic, patient, and supportive."
SYSTEM_PROMPT_DEBATE = "You are analytic, structured, evidence-based. Argue both sides fairly."
# ------------------ CREATOR META (used when users ask) ------------------

CREATOR_INFO = """
Billy AI was created by **GoldBoy**, a 17-year-old developer from England.

He is currently working on futuristic projects such as:
• MintZa  
• LuxStream  
• SwapX  
• CryptoBean  

Socials:
• Instagram: @GoldBoyy  
• Twitter: @GoldBoy
"""

# ---------- UTILS ----------
def get_system_prompt() -> str:
    parts = [SYSTEM_PROMPT_BASE]
    if ENABLE_EMPATHY:
        parts.append(SYSTEM_PROMPT_EMPATHY)
    if ENABLE_DEBATE:
        parts.append(SYSTEM_PROMPT_DEBATE)
    return " ".join(parts)

def detect_creator_question(prompt: str):
    keywords = ["who made you", "who created you", "your creator", "who built you", "owner"]
    p = prompt.lower()
    return any(k in p for k in keywords)

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

ensure_kg_db()

def add_kg_node(title: str, content: str):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO nodes (title, content, created_at) VALUES (?, ?, ?)", (title, content, time.time()))
    conn.commit()
    conn.close()

def query_kg(query: str, limit: int = 5):
    # Very simple full-text LIKE search (scaffold; replace with embedding+semantic search later)
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, content FROM nodes WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?", (f"%{query}%", f"%{query}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [{"title": r[0], "content": r[1]} for r in rows]

# ---------- GROQ STREAMING (chat) ----------
# This function posts to Groq chat completions with stream=True and forwards chunks to SSE.
async def groq_stream(prompt: str):
    """
    Stream Groq chat completions as SSE. The frontend expects JSON chunks (one JSON per data: line).
    This function yields SSE events "data: <json>\n\n" where <json> is the Groq chunk JSON.
    """
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'error': 'no_groq_key'})}\n\n"
        return

    if detect_creator_question(prompt):
        yield f"data: {CREATOR_INFO}\n\n"
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
                    logger.error("Groq stream error status %s: %s", resp.status_code, text[:400])
                    yield f"data: {json.dumps({'error': 'provider_error', 'status': resp.status_code, 'text': text.decode(errors='ignore')[:400]})}\n\n"
                    return

                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    # Groq often prefixes chunks with "data: " like OpenAI
                    if raw_line.startswith("data: "):
                        data = raw_line[len("data: "):]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        # forward the JSON chunk as-is in SSE data field
                        yield f"data: {data}\n\n"
                    else:
                        # ignore headers/events we don't need
                        continue
        except Exception as e:
            logger.exception("Error during groq_stream")
            yield f"data: {json.dumps({'error': 'stream_exception', 'msg': str(e)})}\n\n"
            return

@app.get("/stream")
async def stream_chat(prompt: str):
    """
    SSE endpoint. Frontend should connect with EventSource(`${API_BASE}/stream?prompt=...`)
    Frontend expects each data: line to contain a JSON chunk from Groq.
    """
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")

# ---------- NON-STREAM CHAT (fallback) ----------
@app.post("/chat")
async def chat_endpoint(req: Request):
    payload = await req.json()
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")
    # Use Groq sync completion
    if not GROQ_API_KEY:
        raise HTTPException(400, "GROQ_API_KEY not configured")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    body = {"model": CHAT_MODEL, "messages":[{"role":"system","content":get_system_prompt()},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=body, timeout=30.0)
        r.raise_for_status()
        return r.json()

# ---------- IMAGE GENERATION ----------

@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json"
    }

    data = {
        "prompt": prompt,
        "output_format": "png"
    }

    # Stability requires multipart/form-data even if you send no file
    files = {
        "none": (None, None)
    }

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            url,
            headers=headers,
            data=data,
            files=files
        )

    if resp.status_code != 200:
        return JSONResponse({"error": "Stability generation failed", "details": resp.text}, status_code=400)

    try:
        json_data = resp.json()
        image_b64 = json_data["image"]  # Stability returns a base64 string
        return {"image": image_b64}
    except Exception as e:
        return JSONResponse({"error": "Parse failed", "details": str(e)}, status_code=400)
        
# ---------- TTS (text-to-speech) ----------
@app.post("/tts")
async def tts_endpoint(req: Request):
    data = await req.json()
    text = data.get("text", "")
    if not text:
        raise HTTPException(400, "text required")

    # Prefer Groq if key present (as in your earlier setup)
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/audio/speech"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            body = {"model": TTS_MODEL, "voice": "alloy", "input": text}
            async with httpx.AsyncClient() as client:
                r = await client.post(url, headers=headers, json=body, timeout=60.0)
                r.raise_for_status()
                jr = r.json()
                # jr should contain base64 audio in jr['audio']
                return {"audio": jr.get("audio")}
        except Exception as e:
            logger.exception("Groq TTS error")

    # Optional: If you want OpenAI TTS you can wire it here (requires different endpoints)
    raise HTTPException(400, "No TTS provider configured or provider failed (check logs)")

# ---------- STT (speech-to-text) ----------
@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if GROQ_API_KEY:
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": STT_MODEL, "file": base64.b64encode(audio_bytes).decode(), "format": "base64"}
            async with httpx.AsyncClient() as client:
                r = await client.post(url, headers=headers, json=payload, timeout=60.0)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.exception("Groq STT error")
            raise HTTPException(500, "STT failed")
    raise HTTPException(400, "No STT provider configured")

# ---------- NLU endpoint (intent classification & advanced) ----------
@app.post("/nlp/classify")
async def nlp_classify(req: Request):
    """
    Lightweight intent classifier with LLM fallback:
    - If 'use_llm' true in payload and GROQ_API_KEY exists, call a small Groq completion to classify intent.
    - Otherwise use keyword heuristics (fast, local).
    """
    body = await req.json()
    text = body.get("text", "")
    use_llm = body.get("use_llm", False)

    if not text:
        raise HTTPException(400, "text required")

    # quick heuristic classifier
    lower = text.lower()
    if any(w in lower for w in ("image", "draw", "picture", "photo", "generate")):
        intent = "image"
    elif any(w in lower for w in ("say", "speak", "read", "tts")):
        intent = "tts"
    elif any(w in lower for w in ("summarize", "shorten", "tl;dr")):
        intent = "summarize"
    else:
        intent = "chat"

    # optionally call small LLM classifier for better NLU
    if use_llm and GROQ_API_KEY:
        try:
            prompt = f"Classify the user's intent into one of: image, tts, summarize, chat. Give only JSON: {{\"intent\": <intent>, \"confidence\": 0.0}}. Text: '''{text}'''"
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": CHAT_MODEL, "messages":[{"role":"system","content":"You are an intent classifier."},{"role":"user","content":prompt}], "temperature":0}
            async with httpx.AsyncClient() as client:
                r = await client.post(url, headers=headers, json=payload, timeout=20.0)
                r.raise_for_status()
                jr = r.json()
                # Attempt to parse assistant content as JSON
                content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
                try:
                    parsed = json.loads(content)
                    return {"intent": parsed.get("intent", intent), "confidence": parsed.get("confidence", 0.0), "source":"llm"}
                except Exception:
                    # fallback to heuristic
                    return {"intent": intent, "confidence": 0.5, "source":"heuristic"}
        except Exception as e:
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

# ---------- Personalization / Memory endpoints ----------
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")
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

ensure_memory_db()

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

# ---------- Multilingual / translate scaffold ----------
@app.post("/translate")
async def translate(req: Request):
    body = await req.json()
    text = body.get("text", "")
    target = body.get("target", "en")
    if not text:
        raise HTTPException(400, "text required")
    # Scaffold: you can route to a translation service or call an LLM to translate
    # Here we fallback to a Groq completion prompt if available
    if GROQ_API_KEY:
        prompt = f"Translate the following text to {target}: '''{text}'''"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0}
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            jr = r.json()
            translated = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"text": translated}
    # else return original
    return {"text": text}

# ---------- Real-time data scaffolding (news, crypto, weather) ----------
@app.get("/realtime/news")
async def realtime_news(q: str = "latest", source: str = "newsapi", limit: int = 5):
    """
    Scaffold: this proxies to external news APIs you configure.
    Add NEWSAPI_KEY env var and implement logic here.
    """
    news_api_key = os.getenv("NEWSAPI_KEY", "")
    if not news_api_key:
        return {"error": "no_news_api_key"}
    # Example: call newsapi.org - left as scaffold because key may be private
    url = f"https://newsapi.org/v2/everything?q={httpx.utils.quote(q)}&pageSize={limit}&apiKey={news_api_key}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=20.0)
        r.raise_for_status()
        return r.json()

# ---------- Human-AI collaboration helpers ----------
@app.post("/summarize")
async def summarize(req: Request):
    body = await req.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "text required")
    # Use LLM to summarize via Groq
    if GROQ_API_KEY:
        prompt = f"Summarize the following text briefly:\n\n'''{text}'''"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": CHAT_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0.3}
        async with httpx.AsyncClient() as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            jr = r.json()
            return {"summary": jr.get("choices", [{}])[0].get("message", {}).get("content", "")}
    raise HTTPException(400, "No LLM provider configured")

# ---------- Health check ----------
@app.get("/")
async def root():
    return {
        "status":"ok",
        "providers":{
            "groq": bool(GROQ_API_KEY),
            "openai_images": bool(OPENAI_API_KEY),
            "free_image_provider": bool(USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL)
        }
    }

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="info")
