# main.py
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import asyncio
import logging
import json
import uuid
from email.message import EmailMessage
import smtplib

import httpx
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proai")

# ---------- Config (env driven) ----------
PORT = int(os.getenv("PORT", "8000"))
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "shaynengunga15@gmail.com")

HF_API_KEY = os.getenv("HF_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# SMTP settings for notify_admin
SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("SMTP_PORT", "25"))
SMTP_USER = os.getenv("SMTP_USER", None)
SMTP_PASS = os.getenv("SMTP_PASS", None)

# Chat model name for HF
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "CompVis/stable-diffusion-v1-4")

# ---------- FastAPI app ----------
app = FastAPI(title="ProAI - GoldBoy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
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

# ---------- Helpers ----------
def extract_code_blocks(text: str):
    import re
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    blocks = []
    for lang, code in matches:
        blocks.append({"language": (lang or "text").lower(), "code": code.strip()})
    return blocks

# ---------- Public info endpoints ----------
@app.get("/about")
async def about():
    return {
        "creator": "GoldBoy",
        "bio": "I am a 17-year-old programmer working on multiple projects/sites.",
        "projects": ["NGG", "ST", "MZ, BB, NL"],
        "message": "Project details are private and will not be disclosed."
    }

# ---------- Admin notification endpoint ----------
@app.post("/notify_admin")
async def notify_admin(reason: str = Form(...)):
    msg = EmailMessage()
    msg.set_content(f"User reported AI issue: {reason}")
    msg["Subject"] = "AI Service Alert"
    msg["From"] = ADMIN_EMAIL
    msg["To"] = ADMIN_EMAIL

    try:
        if SMTP_HOST and SMTP_HOST != "localhost" and SMTP_USER:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            server.quit()
        else:
            with smtplib.SMTP("localhost") as s:
                s.send_message(msg)
        return {"status": "notification sent"}
    except Exception as e:
        logger.exception("notify_admin failed: %s", e)
        return {"status": "failed", "error": str(e)}

# ---------- Chat endpoints ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }
    persona_prompt = persona_map.get(req.persona, persona_map["default"])

    history = ""
    for m in req.messages:
        role = "User" if m.role == "user" else "Assistant"
        history += f"{role}: {m.content}\n"

    prompt = persona_prompt + "\n\n" + history + "Assistant:"
    out = None

    # Hugging Face Inference API
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 256, "top_p": 0.95, "temperature": 0.8}
            }
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    out = data[0].get("generated_text", "")
        except Exception:
            logger.exception("HF inference failed.")

    # OpenAI fallback
    if out is None and OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI fallback failed.")

    if not out:
        raise HTTPException(status_code=503, detail="No text generation backend available")

    response = {"response": out}
    if req.code_mode:
        response["code_blocks"] = extract_code_blocks(out)
    return JSONResponse(response)
    
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # same logic as /chat, streaming in chunks
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }
    persona_prompt = persona_map.get(req.persona, persona_map["default"])
    history = ""
    for m in req.messages:
        role = "User" if m.role == "user" else "Assistant"
        history += f"{role}: {m.content}\n"
    prompt = persona_prompt + "\n\n" + history + "Assistant:"
    text = None

    # HF API
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "top_p":0.95,"temperature":0.8}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text","")
        except Exception:
            logger.exception("HF streaming failed.")

    # OpenAI fallback
    if text is None and OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}],"max_tokens":512}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenAI streaming fallback failed.")

    if not text:
        raise HTTPException(status_code=500, detail="Generation failed")

    code_blocks = extract_code_blocks(text) if req.code_mode else []

    async def event_stream():
        chunk_size = 60
        for i in range(0,len(text),chunk_size):
            yield f"data: {text[i:i+chunk_size]}\n\n"
            await asyncio.sleep(0.04)
        if code_blocks:
            yield f"data: {json.dumps({'code_blocks': code_blocks})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Image generation (cloud-only) ----------
@app.post("/image")
async def gen_image(req: ImageRequest):
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
            payload = {"inputs": req.prompt, "parameters":{"width":req.width,"height":req.height}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "image" in data:
                    return {"image_base64": data["image"]}
                elif isinstance(data, list) and len(data) > 0 and "image_base64" in data[0]:
                    return {"image_base64": data[0]["image_base64"]}
        except Exception:
            logger.exception("HF image generation failed.")

    if OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"prompt": req.prompt, "size": f"{req.width}x{req.height}"}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post("https://api.openai.com/v1/images/generations", headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                img_b64 = data["data"][0]["b64_json"]
                return {"image_base64": img_b64}
        except Exception:
            logger.exception("OpenAI DALL·E generation failed.")

    raise HTTPException(status_code=503, detail="No cloud image generation backend available")

# ---------- WebSocket chat (optional) ----------
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
