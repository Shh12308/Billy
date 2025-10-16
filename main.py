from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import asyncio
import logging
import json
import httpx
import re

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billy_ai")

# ---------- Config ----------
PORT = int(os.getenv("PORT", "8000"))
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "CompVis/stable-diffusion-v1-4")

# ---------- App ----------
app = FastAPI(title="Billy AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_ORIGINS,
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

# ---------- Helpers ----------
def extract_code_blocks(text: str):
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [{"language": (lang or "text").lower(), "code": code.strip()} for lang, code in matches]

# ---------- Routes ----------
@app.get("/")
async def home():
    return {"message": "✅ Billy AI backend is running successfully!"}

@app.get("/about")
async def about():
    return {
        "creator": "GoldBoy",
        "bio": "17-year-old programmer working on multiple AI and web projects.",
        "ai_name": "ZyNara (Billy AI)",
        "status": "Online",
    }

# ---------- Chat ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    persona_map = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }

    persona_prompt = persona_map.get(req.persona, persona_map["default"])
    history = "\n".join(
        [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in req.messages]
    )
    prompt = f"{persona_prompt}\n\n{history}\nAssistant:"
    out = None

    # Hugging Face
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.8}}
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    out = data[0].get("generated_text", "")
        except Exception as e:
            logger.error(f"Hugging Face inference failed: {e}")

    # OpenAI fallback
    if out is None and OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}")

    if not out:
        raise HTTPException(status_code=503, detail="No text generation backend available")

    response = {"response": out}
    if req.code_mode:
        response["code_blocks"] = extract_code_blocks(out)

    return JSONResponse(response)

# ---------- Stream Chat ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    persona_prompt = {
        "default": "You are a helpful assistant.",
        "teacher": "You are a calm teacher.",
        "funny": "You are a witty assistant."
    }.get(req.persona, "You are a helpful assistant.")

    history = "\n".join(
        [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in req.messages]
    )
    prompt = f"{persona_prompt}\n\n{history}\nAssistant:"
    text = None

    # Hugging Face
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "")
        except Exception as e:
            logger.error(f"Hugging Face streaming failed: {e}")

    # OpenAI fallback
    if text is None and OPENAI_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI streaming fallback failed: {e}")

    if not text:
        raise HTTPException(status_code=500, detail="Generation failed")

    code_blocks = extract_code_blocks(text) if req.code_mode else []

    async def event_stream():
        chunk_size = 60
        for i in range(0, len(text), chunk_size):
            yield f"data: {text[i:i+chunk_size]}\n\n"
            await asyncio.sleep(0.04)
        if code_blocks:
            yield f"data: {json.dumps({'code_blocks': code_blocks})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Image ----------
@app.post("/image")
async def gen_image(req: ImageRequest):
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
            payload = {"inputs": req.prompt}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) > 0 and "image_base64" in data[0]:
                    return {"image_base64": data[0]["image_base64"]}
        except Exception as e:
            logger.error(f"Image generation failed: {e}")

    raise HTTPException(status_code=503, detail="No image backend available")

# ---------- WebSocket ----------
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
                for k, conn in active_ws_connections.items():
                    if k != uid:
                        await conn.send_json({"from": uid, "message": message})
    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {uid}")
        active_ws_connections.pop(uid, None)

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
