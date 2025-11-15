# billy_allinone_render.py
# Updated — async-safe HF calls, fixed websocket streaming, cleaned undefined names.
import uvicorn
import asyncio
import os
import io
import uuid
import json
import time
import tempfile
from typing import Optional, List
from fastapi import WebSocket
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Hugging Face token (required for HF inference fallback)
HF_TOKEN = os.getenv("HF_TOKEN")
# Use a template, we'll fill model id in calls: hf_api_url.format(model_id)
HF_API_URL = "https://api-inference.huggingface.co/models/{}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Optional Supabase (keeps original behavior)
try:
    from supabase import create_client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    supabase = None

# Optional heavy local libs (guarded)
try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

# App init
app = FastAPI(title="Billy-AllInOne Render Edition")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CREATOR = {"name": "GoldBoy", "age": 17, "location": "England"}
PERSONALITY = f"You are Billy, a friendly assistant built by {CREATOR['name']}."

# MODEL defaults (you can change these or pass model ids to helpers)
DEFAULT_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_SUMMARY = "sshleifer/distilbart-cnn-12-6"
DEFAULT_STT = "openai/whisper-small"
DEFAULT_TTS = "espnet/kan-bayashi_ljspeech_vits"
DEFAULT_DETECT = "facebook/detr-resnet-50"
DEFAULT_CAPTION = "Salesforce/blip-image-captioning-large"
DEFAULT_VQA = "Salesforce/blip-vqa-base"
DEFAULT_INPAINT = "stabilityai/stable-diffusion-x4-inpainting"
DEFAULT_UPSCALE = "nateraw/real-esrgan"
DEFAULT_OCR = "microsoft/trocr-base-handwritten"
DEFAULT_MUSIC = "facebook/musicgen-small"
DEFAULT_3D = "openai/point-e"
DEFAULT_CODEGEN = "Salesforce/codegen-350M-multi"
DEFAULT_QA = "deepset/roberta-base-squad2"
DEFAULT_SENTIMENT = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TRANSLATE_FR = "Helsinki-NLP/opus-mt-en-fr"

# -----------------------
# Helpers
# -----------------------
async def hf_inference(model_id: str, inputs, params: dict = None, is_binary: bool = False, timeout: int = 120):
    """
    Async wrapper around HF Inference (router/inference) using httpx AsyncClient.
    model_id: full HF model id (e.g. 'tiiuae/falcon-7b')
    inputs: either text or bytes (if is_binary True)
    params: optional parameters dict passed under "parameters"
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN not set")
    url = HF_API_URL.format(model_id)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        if is_binary:
            # many HF endpoints accept raw bytes - pass as files
            files = {"file": ("file", inputs)}
            data = {}
            if params:
                data["parameters"] = json.dumps(params)
            resp = await client.post(url, headers=headers, data=data, files=files)
        else:
            body = {"inputs": inputs}
            if params:
                body["parameters"] = params
            resp = await client.post(url, headers=headers, json=body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"HF inference error: {resp.text}")
    # try JSON, else text
    try:
        return resp.json()
    except Exception:
        return resp.text

def log_event(name: str, payload: dict):
    if supabase:
        try:
            supabase.table("ai_logs").insert({"event": name, "payload": payload}).execute()
        except Exception:
            pass

# -----------------------
# Basic routes
# -----------------------
@app.get("/")
def index():
    return {"message": "Billy AllInOne — endpoints: /chat, /ws/stream (websocket), /translate, /summarize, /tts, /stt, /detect, /caption, /vqa, /inpaint, /upscale, /ocr, /music, /3d, /benchmark, /creator"}

@app.get("/health")
def health():
    return {"ok": True, "hf_token": bool(HF_TOKEN), "torch": bool(torch)}

@app.get("/creator")
def creator():
    return {"creator": CREATOR}

# -----------------------
# Text endpoints
# -----------------------
@app.post("/chat")
async def chat(prompt: str = Form(...), user_id: Optional[str] = Form("guest"), model_id: Optional[str] = Form(None)):
    """
    Simple chat endpoint. Calls HF model via hf_inference (async).
    Provide model_id if you want to override the default.
    """
    prompt_full = f"{PERSONALITY}\nUser: {prompt}\nBilly:"
    if HF_TOKEN:
        used = model_id or DEFAULT_CHAT_MODEL
        resp = await hf_inference(used, prompt_full, params={"max_new_tokens": 200})
        # HF response formats vary; try to extract text
        resp_text = ""
        if isinstance(resp, list) and len(resp) > 0:
            first = resp[0]
            if isinstance(first, dict) and "generated_text" in first:
                resp_text = first["generated_text"]
            else:
                resp_text = str(first)
        elif isinstance(resp, dict):
            if "generated_text" in resp:
                resp_text = resp["generated_text"]
            elif isinstance(resp.get("output", None), str):
                resp_text = resp["output"]
            else:
                resp_text = json.dumps(resp)[:5000]
        else:
            resp_text = str(resp)
        log_event("chat_hf", {"user": user_id, "prompt": prompt, "model": used})
        return {"source":"hf","response": resp_text}
    raise HTTPException(status_code=503, detail="No HF_TOKEN set")

@app.post("/translate")
async def translate(text: str = Form(...), to: str = Form("fr")):
    model_map = {"fr": DEFAULT_TRANSLATE_FR, "es": "Helsinki-NLP/opus-mt-en-es"}
    model = model_map.get(to, DEFAULT_TRANSLATE_FR)
    if HF_TOKEN:
        resp = await hf_inference(model, text, params={"max_new_tokens": 200})
        # inference formats vary
        if isinstance(resp, list) and isinstance(resp[0], dict):
            return {"source":"hf","translation": resp[0].get("translation_text", str(resp[0]))}
        return {"source":"hf","translation": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# WebSocket streaming endpoint
# -----------------------
@app.websocket("/ws/stream")
async def stream_socket(socket: WebSocket):
    """
    WebSocket streaming endpoint.
    Client must send JSON: {"prompt":"...","model_id":"optional-model-id"}
    The server will call HF inference (non-streaming) and stream the returned text
    back to the client in small character/word chunks to simulate streaming.
    """
    await socket.accept()
    try:
        message = await socket.receive_json()
        prompt = message.get("prompt")
        model = message.get("model_id") or DEFAULT_CHAT_MODEL

        if not prompt:
            await socket.send_json({"error": "No prompt provided"})
            await socket.close()
            return

        if not HF_TOKEN:
            await socket.send_json({"error": "HF_TOKEN not set on server"})
            await socket.close()
            return

        # Prepend personality to the prompt for context
        prompt_full = f"{PERSONALITY}\nUser: {prompt}\nBilly:"

        # Call HF asynchronously (single request) — HF router returns result synchronously
        hf_resp = await hf_inference(model, prompt_full, params={"max_new_tokens": 200}, timeout=240)

        # Extract text robustly
        resp_text = ""
        if isinstance(hf_resp, list) and len(hf_resp) > 0:
            first = hf_resp[0]
            if isinstance(first, dict) and "generated_text" in first:
                resp_text = first["generated_text"]
            else:
                resp_text = str(first)
        elif isinstance(hf_resp, dict):
            if "generated_text" in hf_resp:
                resp_text = hf_resp["generated_text"]
            elif "text" in hf_resp:
                resp_text = hf_resp["text"]
            else:
                # fallback: stringify
                resp_text = json.dumps(hf_resp)[:10000]
        else:
            resp_text = str(hf_resp)

        log_event("chat_stream_hf", {"prompt": prompt, "model": model})

        # Stream back in small chunks (word-by-word to be friendly)
        # You can adjust the chunking strategy here (characters, sentences, tokens, etc.)
        words = resp_text.split()
        for w in words:
            await socket.send_json({"delta": w + " "})
            # small sleep to allow client to render streaming parts
            await asyncio.sleep(0.02)

        # final done message with full text
        await socket.send_json({"done": True, "final": resp_text})
        await socket.close()
    except Exception as e:
        # avoid crashing connection: send error and close
        try:
            await socket.send_json({"error": str(e)})
        except Exception:
            pass
        await socket.close()

# -----------------------
# Summarize / QA / Codegen / Embeddings etc.
# -----------------------
@app.post("/summarize")
async def summarize(text: str = Form(...), model_id: Optional[str] = Form(None)):
    if HF_TOKEN:
        used = model_id or DEFAULT_SUMMARY
        resp = await hf_inference(used, text, params={"max_new_tokens": 120})
        return {"source":"hf","summary": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference(DEFAULT_SENTIMENT, text)
        return {"source":"hf","sentiment": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/qa")
async def qa(question: str = Form(...), context: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference(DEFAULT_QA, {"question": question, "context": context})
        return {"source":"hf","answer": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/codegen")
async def codegen(prompt: str = Form(...), model_id: Optional[str] = Form(None)):
    if HF_TOKEN:
        used = model_id or DEFAULT_CODEGEN
        resp = await hf_inference(used, prompt, params={"max_new_tokens": 300})
        return {"source":"hf","code": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Speech endpoints
# -----------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_STT, data, is_binary=True)
        return {"source":"hf","text": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/tts")
async def tts(text: str = Form(...), model_id: Optional[str] = Form(None)):
    if HF_TOKEN:
        used = model_id or DEFAULT_TTS
        resp = await hf_inference(used, text)
        return {"source":"hf","data": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Vision endpoints
# -----------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not Image:
        raise HTTPException(status_code=503, detail="PIL not available on server")
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_DETECT, data, is_binary=True)
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_CAPTION, data, is_binary=True)
        return {"source":"hf","caption": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/vqa")
async def vqa(file: UploadFile = File(...), question: str = Form(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_VQA, {"image": data, "question": question})
        return {"source":"hf","answer": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form("")):
    if HF_TOKEN:
        img_data = await image.read()
        mask_data = await mask.read()
        resp = await hf_inference(DEFAULT_INPAINT, {"image": img_data, "mask": mask_data, "prompt": prompt})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(2)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_UPSCALE, data, params={"scale": scale})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference(DEFAULT_OCR, data, is_binary=True)
        return {"source":"hf","text": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Music & 3D placeholders
# -----------------------
@app.post("/music")
async def music(prompt: str = Form(...), duration: int = Form(10)):
    if HF_TOKEN:
        resp = await hf_inference(DEFAULT_MUSIC, prompt, params={"duration": duration})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/3d")
async def text_to_3d(prompt: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference(DEFAULT_3D, prompt)
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Utility endpoints
# -----------------------
@app.post("/benchmark")
async def benchmark_model(model_id: str = Form(...), sample_input: str = Form("Hello world")):
    result = {"model": model_id}
    start = time.time()
    try:
        if HF_TOKEN:
            j = await hf_inference(model_id, sample_input, params={"max_new_tokens":64})
            result["source"] = "hf"
            result["output_sample"] = str(j)[:200]
        else:
            raise HTTPException(status_code=503, detail="HF token not set")
    except Exception as e:
        result["error"] = str(e)
    result["elapsed"] = time.time() - start
    return result

@app.post("/register")
def register(username: str = Form(...), email: str = Form(...)):
    if not supabase:
        return {"error":"Supabase not configured"}
    user_id = uuid.uuid4().hex
    try:
        supabase.table("users").insert({"id": user_id, "username": username, "email": email, "created_at": int(time.time())}).execute()
        return {"status":"ok","user_id": user_id}
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Main entry
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # use use_colors False to avoid weird render logs; uvicorn will keep process persistent
    uvicorn.run("billy_allinone_render:app", host="0.0.0.0", port=port, log_level="info")
