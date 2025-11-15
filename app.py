# main.py - Full-featured, small footprint AI server for Render Free
import os
import json
import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("render-ai-server")

# -----------------------
# Config
# -----------------------
GROQ_API_URL = os.getenv("GROQ_API_URL")         # Provider API endpoint
GROQ_API_KEY = os.getenv("GROQ_API_KEY")         # Provider API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     # Optional moderation
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-like-model")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embed-model")
PROVIDER_TIMEOUT = int(os.getenv("PROVIDER_TIMEOUT", "60"))

# Rate limiting
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30
_rate_limit_store: Dict[str, List[float]] = {}

# In-memory chat memory
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}

# -----------------------
# App
# -----------------------
app = FastAPI(title="Render-Free AI Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -----------------------
# Rate limiting
# -----------------------
def check_rate_limit(client_id: str):
    now = time.time()
    entries = _rate_limit_store.get(client_id, [])
    entries = [t for t in entries if t > now - RATE_LIMIT_WINDOW]
    if len(entries) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    entries.append(now)
    _rate_limit_store[client_id] = entries

# -----------------------
# Simple moderation
# -----------------------
async def moderate_text(text: str) -> (bool, Optional[str]):
    if not text:
        return True, None
    # heuristic
    banned = ["bomb", "kill", "terror", "child abuse"]
    if any(word in text.lower() for word in banned):
        return False, "Blocked by heuristic"
    # optional OpenAI moderation
    if OPENAI_API_KEY:
        try:
            url = "https://api.openai.com/v1/moderations"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "omni-moderation-latest", "input": text}
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(url, headers=headers, json=payload)
                j = r.json()
                if j.get("results", [{}])[0].get("flagged", False):
                    return False, "Blocked by OpenAI moderation"
        except Exception:
            pass
    return True, None

# -----------------------
# Provider call wrapper
# -----------------------
async def groq_invoke(model: str, prompt: str, stream=False, parameters=None):
    if not GROQ_API_URL or not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Provider not configured")
    url = GROQ_API_URL.format(model=model) if "{model}" in GROQ_API_URL else GROQ_API_URL
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt}
    if parameters: payload["parameters"] = parameters
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT) as client:
        if stream:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                async for chunk in resp.aiter_text():
                    if chunk:
                        yield chunk
        else:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            try: return r.json()
            except: return r.text

# -----------------------
# Health
# -----------------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "configured": bool(GROQ_API_URL and GROQ_API_KEY),
        "default_model": DEFAULT_MODEL
    }

# -----------------------
# Text endpoints
# -----------------------
@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    model = body.get("model", DEFAULT_MODEL)
    parameters = body.get("parameters", {})
    client_id = request.client.host if request.client else "anon"
    check_rate_limit(client_id)
    allowed, reason = await moderate_text(prompt)
    if not allowed: raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")
    out = await groq_invoke(model, prompt, stream=False, parameters=parameters)
    text = out.get("text") if isinstance(out, dict) else str(out)
    return {"source": "provider", "model": model, "text": text}

@app.post("/chat")
async def chat(user_id: Optional[str] = Form("guest"), prompt: Optional[str] = Form(None)):
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    check_rate_limit(user_id)
    allowed, reason = await moderate_text(prompt)
    if not allowed: raise HTTPException(status_code=400, detail=f"Moderation blocked: {reason}")
    history = CHAT_MEMORY.setdefault(user_id, [])
    convo = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history[-6:]])
    composed = f"{convo}\nUser: {prompt}\nAssistant:"
    out = await groq_invoke(DEFAULT_MODEL, composed)
    text = out.get("text") if isinstance(out, dict) else str(out)
    history.append({"user": prompt, "assistant": text})
    CHAT_MEMORY[user_id] = history[-64:]
    return {"source": "provider", "model": DEFAULT_MODEL, "response": text}

@app.post("/translate")
async def translate(text: str = Form(...), to: str = Form("fr")):
    model_map = {"fr":"Helsinki-NLP/opus-mt-en-fr","es":"Helsinki-NLP/opus-mt-en-es"}
    model = model_map.get(to,"Helsinki-NLP/opus-mt-en-fr")
    out = await groq_invoke(model, text)
    return {"source": "provider", "translation": out.get("translation_text", str(out)) if isinstance(out, dict) else str(out)}

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    out = await groq_invoke("sshleifer/distilbart-cnn-12-6", text)
    return {"source": "provider", "summary": str(out)}

@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    out = await groq_invoke("distilbert-base-uncased-finetuned-sst-2-english", text)
    return {"source": "provider", "sentiment": out}

@app.post("/qa")
async def qa(question: str = Form(...), context: str = Form(...)):
    out = await groq_invoke("deepset/roberta-base-squad2", {"question": question, "context": context})
    return {"source": "provider", "answer": out}

@app.post("/codegen")
async def codegen(prompt: str = Form(...)):
    out = await groq_invoke("Salesforce/codegen-350M-multi", prompt)
    return {"source": "provider", "code": str(out)}

# -----------------------
# Audio endpoints
# -----------------------
@app.post("/tts")
async def tts(text: str = Form(...)):
    out = await groq_invoke("espnet/kan-bayashi_ljspeech_vits", text)
    return {"source": "provider", "audio": out}

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    data = await file.read()
    out = await groq_invoke("openai/whisper-small", data)
    return {"source": "provider", "text": out}

# -----------------------
# Image endpoints
# -----------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    data = await file.read()
    out = await groq_invoke("Salesforce/blip-image-captioning-large", data)
    return {"source": "provider", "caption": out}

@app.post("/vqa")
async def vqa(file: UploadFile = File(...), question: str = Form(...)):
    data = await file.read()
    out = await groq_invoke("Salesforce/blip-vqa-base", {"image": data, "question": question})
    return {"source": "provider", "answer": out}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    out = await groq_invoke("facebook/detr-resnet-50", data)
    return {"source": "provider", "result": out}

# -----------------------
# Embeddings
# -----------------------
@app.post("/embed")
async def embed(prompt: str = Form(...)):
    out = await groq_invoke(EMBED_MODEL, prompt, parameters={"task":"embed"})
    return {"source": "provider", "embeddings": out}

# -----------------------
# WebSocket streaming
# -----------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        meta = json.loads(raw)
        prompt = meta.get("prompt")
        model = meta.get("model", DEFAULT_MODEL)
        client_id = websocket.client.host if websocket.client else "anon"
        check_rate_limit(client_id)
        allowed, reason = await moderate_text(prompt)
        if not allowed:
            await websocket.send_json({"error": f"Moderation blocked: {reason}"})
            await websocket.close(); return
        async for chunk in groq_invoke(model, prompt, stream=True):
            await websocket.send_json({"delta": chunk})
        await websocket.send_json({"done": True})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WS error: %s", e)
        await websocket.send_json({"error": str(e)}); await websocket.close()

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("Starting Render-Free AI Server on port %d", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
