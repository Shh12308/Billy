# main.py
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
import os
import asyncio
import logging
import json
import httpx
import random
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zynara")

# ---------- Config ----------
PORT = int(os.getenv("PORT", "8000"))
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")
HF_API_KEY = os.getenv("HF_API_KEY")  # e.g. hf_xxx
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")  # path to a .gguf LLaMA model if you upload one
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")  # used for HF inference
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion")  # optional HF image model

# ---------- App ----------
app = FastAPI(title="ZyNara (Billy) API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    persona: str = "default"
    stream: bool = False
    code_mode: bool = False

class ImageRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

# ---------- Optional local Llama (llama_cpp) ----------
local_llm = None
try:
    if LOCAL_MODEL_PATH:
        # llama_cpp import is optional; only used if LOCAL_MODEL_PATH is set and package installed
        from llama_cpp import Llama
        logger.info("Attempting to load local Llama model at %s", LOCAL_MODEL_PATH)
        local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048)
        logger.info("Local Llama loaded (llama_cpp).")
    else:
        logger.info("No LOCAL_MODEL_PATH provided; skipping local model load.")
except Exception as e:
    logger.warning("Local Llama not available or failed to load: %s", e)
    local_llm = None

# ---------- Helpers ----------
def extract_code_blocks(text: str):
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [{"language": (lang or "text").lower(), "code": code.strip()} for lang, code in matches]

def simple_fallback(prompt: str) -> str:
    choices = [
        f"Hey — ZyNara here. My main model is unavailable; quick thought about \"{prompt}\": try breaking it down into smaller steps.",
        f"ZyNara (fallback): I can't reach the main brain right now, but regarding \"{prompt}\", ask me for more details and I'll help as much as I can.",
        f"Sorry — main model offline. For \"{prompt}\", I'd suggest starting with the basics and building up."
    ]
    return random.choice(choices)

def build_persona_prompt(persona: str) -> str:
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }
    return persona_map.get(persona, persona_map["default"])

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
      <head><title>ZyNara</title></head>
      <body style="font-family: sans-serif; text-align:center; padding:40px;">
        <h1>🤖 ZyNara (Billy) API</h1>
        <p>Backend is running. Use <code>/chat</code> POST endpoint to interact.</p>
        <p>Env: LOCAL_MODEL_PATH={}, HF_API_KEY set: {}</p>
      </body>
    </html>
    """.format(bool(LOCAL_MODEL_PATH), bool(HF_API_KEY))

# ---------- Chat endpoint ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    # Build prompt
    persona_prompt = build_persona_prompt(req.persona)
    history_lines = []
    for m in req.messages:
        who = "User" if m.role == "user" else "Assistant"
        history_lines.append(f"{who}: {m.content}")
    history = "\n".join(history_lines)
    prompt = f"{persona_prompt}\n\n{history}\nAssistant:"

    # 1) Try local Llama (preferred if present)
    if local_llm:
        try:
            logger.info("Using local Llama model for generation.")
            out = local_llm.create(prompt=prompt, max_tokens=256, temperature=0.7)
            # llama_cpp returns different structures; handle safely
            text = ""
            if isinstance(out, dict):
                # new llama_cpp returns 'choices' with 'text'
                choices = out.get("choices")
                if choices and isinstance(choices, list):
                    text = "".join([c.get("text","") for c in choices])
                else:
                    text = out.get("text", "")
            else:
                text = str(out)
            if not text:
                raise RuntimeError("Local model produced empty text")
            response = {"response": text}
            if req.code_mode:
                response["code_blocks"] = extract_code_blocks(text)
            return JSONResponse(response)
        except Exception as e:
            logger.exception("Local model failed, falling back: %s", e)

    # 2) Try Hugging Face Inference API
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        text = data[0].get("generated_text") or data[0].get("generated_text", "") or str(data[0])
                    else:
                        text = str(data)
                    response = {"response": text}
                    if req.code_mode:
                        response["code_blocks"] = extract_code_blocks(text)
                    return JSONResponse(response)
                else:
                    logger.warning("HF inference returned status %s: %s", r.status_code, r.text[:400])
        except Exception as e:
            logger.exception("Hugging Face call failed: %s", e)

    # 3) Final fallback (friendly)
    logger.info("Using final fallback response.")
    fallback_text = simple_fallback(req.messages[-1].content if req.messages else "your request")
    return JSONResponse({"response": fallback_text})

# ---------- Stream Chat (SSE) ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    persona_prompt = build_persona_prompt(req.persona)
    history = "\n".join([f"{'User' if m.role=='user' else 'Assistant'}: {m.content}" for m in req.messages])
    prompt = f"{persona_prompt}\n\n{history}\nAssistant:"

    # Attempt local model streaming if available
    text = None
    if local_llm:
        try:
            out = local_llm.create(prompt=prompt, max_tokens=256, temperature=0.7)
            if isinstance(out, dict):
                choices = out.get("choices")
                if choices and isinstance(choices, list):
                    text = "".join([c.get("text","") for c in choices])
                else:
                    text = out.get("text","")
            else:
                text = str(out)
        except Exception as e:
            logger.exception("Local streaming failed: %s", e)
            text = None

    # Otherwise try HF
    if text is None and HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        text = data[0].get("generated_text", "")
                    else:
                        text = str(data)
                else:
                    logger.warning("HF stream returned %s: %s", r.status_code, r.text[:400])
        except Exception as e:
            logger.exception("HF streaming failed: %s", e)
            text = None

    if not text:
        text = simple_fallback(req.messages[-1].content if req.messages else "request")

    async def event_stream():
        chunk_size = 60
        for i in range(0, len(text), chunk_size):
            yield f"data: {text[i:i+chunk_size]}\n\n"
            await asyncio.sleep(0.03)
        if req.code_mode:
            blocks = extract_code_blocks(text)
            if blocks:
                yield f"data: {json.dumps({'code_blocks': blocks})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Image generation (HF only) ----------
@app.post("/image")
async def gen_image(req: ImageRequest):
    if HF_API_KEY and IMAGE_MODEL:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
            payload = {"inputs": req.prompt, "parameters": {"width": req.width, "height": req.height}}
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                # HF image responses vary — try common fields
                if isinstance(data, dict) and "image" in data:
                    return {"image_base64": data["image"]}
                if isinstance(data, list) and len(data) > 0 and "image_base64" in data[0]:
                    return {"image_base64": data[0]["image_base64"]}
                return {"result": data}
        except Exception as e:
            logger.exception("Image generation failed: %s", e)
            raise HTTPException(status_code=503, detail="Image generation failed")
    raise HTTPException(status_code=503, detail="No image backend available")

# ---------- WebSocket chat ----------
active_ws_connections = {}
@app.websocket("/ws/{uid}")
async def websocket_endpoint(websocket: WebSocket, uid: str):
    await websocket.accept()
    active_ws_connections[uid] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            target = data.get("to")
            message = data.get("message")
            if target and target in active_ws_connections:
                await active_ws_connections[target].send_json({"from": uid, "message": message})
            else:
                for k, conn in list(active_ws_connections.items()):
                    if k != uid:
                        try:
                            await conn.send_json({"from": uid, "message": message})
                        except Exception:
                            pass
    except WebSocketDisconnect:
        logger.info("WS disconnected %s", uid)
    finally:
        active_ws_connections.pop(uid, None)

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
