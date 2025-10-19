# main.py
import os
import asyncio
import uuid
import logging
import random
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chloe")

# ---------- Config ----------
PORT = int(os.getenv("PORT", "8000"))
CLIENT_ORIGINS = os.getenv("CLIENT_ORIGINS", "*").split(",")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face API key
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion")

# ---------- App ----------
app = FastAPI(title="Chloe AI", version="1.0.0")
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

# ---------- Local LLaMA (optional) ----------
local_llm = None
try:
    if LOCAL_MODEL_PATH:
        from llama_cpp import Llama
        logger.info(f"Loading local LLaMA at {LOCAL_MODEL_PATH}")
        local_llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=2048)
        logger.info("✅ Local LLaMA loaded.")
except Exception as e:
    logger.warning(f"Local LLaMA not available: {e}")
    local_llm = None

# ---------- Helpers ----------
def extract_code_blocks(text: str):
    pattern = r"```(python|js|node|html|css)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [{"language": (lang or "text").lower(), "code": code.strip()} for lang, code in matches]

def simple_fallback(prompt: str) -> str:
    choices = [
        f"Hey — Chloe here. My main model is unavailable; quick thought about \"{prompt}\": try breaking it down into smaller steps.",
        f"Chloe (fallback): I can't reach the main brain right now, but regarding \"{prompt}\", ask me for more details and I'll help as much as I can.",
        f"Sorry — main model offline. For \"{prompt}\", I'd suggest starting with the basics and building up."
    ]
    return random.choice(choices)

def build_persona_prompt(persona: str) -> str:
    persona_map = {
        "default": "You are a helpful assistant named Chloe.",
        "teacher": "You are a calm teacher named Chloe.",
        "funny": "You are a witty assistant named Chloe."
    }
    return persona_map.get(persona, persona_map["default"])

# ---------- Home ----------
@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <html>
      <head><title>Chloe AI</title></head>
      <body style="font-family: sans-serif; text-align:center; padding:40px;">
        <h1>🤖 Chloe AI Backend</h1>
        <p>Use <code>/chat</code> POST endpoint to interact.</p>
        <p>Env: LOCAL_MODEL_PATH={bool(LOCAL_MODEL_PATH)}, HF_API_KEY set={bool(HF_API_KEY)}</p>
      </body>
    </html>
    """

# ---------- Chat endpoint ----------
@app.post("/chat")
async def chat(req: ChatRequest):
    persona_prompt = build_persona_prompt(req.persona)
    history = "\n".join([f"{'User' if m.role=='user' else 'Chloe'}: {m.content}" for m in req.messages])
    prompt = f"{persona_prompt}\n\n{history}\nChloe:"

    # Local model
    if local_llm:
        try:
            out = local_llm.create(prompt=prompt, max_tokens=256, temperature=0.7)
            text = ""
            if isinstance(out, dict):
                choices = out.get("choices")
                if choices:
                    text = "".join([c.get("text","") for c in choices])
                else:
                    text = out.get("text", "")
            else:
                text = str(out)
            if text:
                response = {"response": text}
                if req.code_mode:
                    response["code_blocks"] = extract_code_blocks(text)
                return JSONResponse(response)
        except Exception as e:
            logger.warning(f"Local LLaMA failed: {e}")

    # Hugging Face
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            url = f"https://api-inference.huggingface.co/models/{LOCAL_MODEL_NAME}"
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "top_p":0.95, "temperature":0.8}}
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        text = data[0].get("generated_text") or ""
                    else:
                        text = str(data)
                    response = {"response": text}
                    if req.code_mode:
                        response["code_blocks"] = extract_code_blocks(text)
                    return JSONResponse(response)
        except Exception as e:
            logger.warning(f"HF API failed: {e}")

    # Final fallback
    fallback = simple_fallback(req.messages[-1].content if req.messages else "request")
    return JSONResponse({"response": fallback})

# ---------- Stream Chat ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    persona_prompt = build_persona_prompt(req.persona)
    history = "\n".join([f"{'User' if m.role=='user' else 'Chloe'}: {m.content}" for m in req.messages])
    prompt = f"{persona_prompt}\n\n{history}\nChloe:"

    text = None
    if local_llm:
        try:
            out = local_llm.create(prompt=prompt, max_tokens=256, temperature=0.7)
            if isinstance(out, dict):
                choices = out.get("choices")
                if choices:
                    text = "".join([c.get("text","") for c in choices])
                else:
                    text = out.get("text","")
            else:
                text = str(out)
        except:
            text = None

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
        except:
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

# ---------- Image Generation ----------
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
                if isinstance(data, dict) and "image" in data:
                    return {"image_base64": data["image"]}
                if isinstance(data, list) and len(data) > 0 and "image_base64" in data[0]:
                    return {"image_base64": data[0]["image_base64"]}
                return {"result": data}
        except Exception as e:
            logger.warning(f"Image generation failed: {e}")
            raise HTTPException(status_code=503, detail="Image generation failed")
    raise HTTPException(status_code=503, detail="No image backend available")

# ---------- Image Analysis + Recommendations ----------
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    Blip2Processor = Blip2ForConditionalGeneration = None

BLIP_PROCESSOR = None
BLIP_MODEL = None

def init_vision():
    global BLIP_PROCESSOR, BLIP_MODEL
    if BLIP_MODEL is None and Blip2Processor and Blip2ForConditionalGeneration:
        try:
            model_name = "Salesforce/blip2-flan-t5-base"
            BLIP_PROCESSOR = Blip2Processor.from_pretrained(model_name)
            BLIP_MODEL = Blip2ForConditionalGeneration.from_pretrained(model_name)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            BLIP_MODEL.to(device)
            logger.info(f"✅ BLIP-2 vision model loaded on {device}")
        except Exception as e:
            logger.warning(f"Vision model load failed: {e}")
            BLIP_PROCESSOR = BLIP_MODEL = None
    return BLIP_PROCESSOR, BLIP_MODEL

def analyze_image(image_path: str) -> str:
    proc, model = init_vision()
    if not proc or not model:
        return "Chloe cannot analyze images right now."
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=64)
    caption = proc.decode(out_ids[0], skip_special_tokens=True)
    return f"Chloe sees: {caption}\n💡 Recommendation: Based on this, you might try something similar or explore related ideas."

@app.post("/image/analyze")
async def analyze_image_endpoint(file: bytes = None):
    if not file:
        raise HTTPException(status_code=400, detail="No image uploaded")
    tmp_path = f"/tmp/chloe_img_{uuid.uuid4().hex}.png"
    with open(tmp_path, "wb") as f:
        f.write(file)
    result = analyze_image(tmp_path)
    return JSONResponse({"analysis": result})

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
                for k, conn in list(active_ws_connections.items()):
                    if k != uid:
                        try:
                            await conn.send_json({"from": uid, "message": message})
                        except:
                            pass
    except WebSocketDisconnect:
        logger.info(f"WS disconnected {uid}")
    finally:
        active_ws_connections.pop(uid, None)

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
