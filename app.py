# billy_allinone_render.py
import uvicorn
import requests
import json
import asyncio
import os, io, uuid, json, time, tempfile
from typing import Optional, List
from fastapi import WebSocket
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import threading

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b"

# Supabase (optional)
try:
    from supabase import create_client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
except:
    supabase = None

# Optional heavy local imports (guarded)
try:
    import torch
except: torch = None
try:
    from PIL import Image
except: Image = None

# App init
app = FastAPI(title="Billy-AllInOne Render Edition")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CREATOR = {"name": "GoldBoy", "age": 17, "location": "England"}
PERSONALITY = f"You are Billy, a friendly assistant built by {CREATOR['name']}."

# -----------------------
# Helpers
# -----------------------
async def hf_inference(model_id, inputs, params=None, is_binary=False):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN not set")
    url = HF_INFERENCE_URL.format(model_id)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient(timeout=120) as client:
        if is_binary:
            files = {"file": inputs}
            data = {"parameters": json.dumps(params or {})}
            resp = await client.post(url, headers=headers, data=data, files=files)
        else:
            payload = {"inputs": inputs}
            if params: payload["parameters"] = params
            resp = await client.post(url, headers=headers, json=payload)
    if resp.status_code >= 400: raise HTTPException(status_code=resp.status_code, detail=resp.text)
    try: return resp.json()
    except: return resp.text

def log_event(name: str, payload: dict):
    if supabase:
        try: supabase.table("ai_logs").insert({"event": name, "payload": payload}).execute()
        except: pass

# -----------------------
# Basic routes
# -----------------------
@app.get("/")
def index():
    return {"message": "Billy AllInOne — endpoints: /chat, /translate, /summarize, /tts, /stt, /detect, /caption, /vqa, /inpaint, /upscale, /ocr, /music, /3d, /benchmark, /creator"}

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
async def chat(prompt: str = Form(...), user_id: Optional[str] = Form("guest")):
    prompt_full = f"{PERSONALITY}\nUser: {prompt}\nBilly:"
    if HF_TOKEN:
        hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
        resp = await hf_inference(hf_model, prompt_full, params={"max_new_tokens":200})
        resp_text = ""
        if isinstance(resp, list) and len(resp) > 0:
            if isinstance(resp[0], dict) and "generated_text" in resp[0]:
                resp_text = resp[0]["generated_text"]
            else: resp_text = str(resp[0])
        elif isinstance(resp, dict) and resp.get("generated_text"):
            resp_text = resp["generated_text"]
        else: resp_text = str(resp)
        log_event("chat_hf", {"user": user_id, "prompt": prompt})
        return {"source":"hf","response": resp_text}
    raise HTTPException(status_code=503, detail="No HF_TOKEN set")

@app.post("/translate")
async def translate(text: str = Form(...), to: str = Form("fr")):
    model_map = {"fr":"Helsinki-NLP/opus-mt-en-fr","es":"Helsinki-NLP/opus-mt-en-es"}
    model = model_map.get(to,"Helsinki-NLP/opus-mt-en-fr")
    if HF_TOKEN:
        resp = await hf_inference(model, text, params={"max_new_tokens":200})
        if isinstance(resp, list) and isinstance(resp[0], dict):
            return {"source":"hf","translation": resp[0].get("translation_text", str(resp[0]))}
        return {"source":"hf","translation": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")


@app.websocket("/ws/stream")
async def stream_socket(socket: WebSocket):
    await socket.accept()

    try:
        message = await socket.receive_json()
        prompt = message.get("prompt")

        if not prompt:
            await socket.send_json({"error": "No prompt provided"})
            await socket.close()
            return

        # Call HuggingFace model
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200}
        }

        response = requests.post(HF_API_URL, headers=HF_HEADERS, data=json.dumps(payload))

        if response.status_code != 200:
            await socket.send_json({"error": f"HF error: {response.text}"})
            await socket.close()
            return

        data = response.json()
        resp_text = data[0]["generated_text"]

        # Stream word-by-word
        for word in resp_text.split():
            await socket.send_json({"delta": word + " "})
            await asyncio.sleep(0.03)

        await socket.send_json({"done": True})
        await socket.close()

    except Exception as e:
        await socket.send_json({"error": str(e)})
        await socket.close()
        
@app.post("/summarize")
async def summarize(text: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("sshleifer/distilbart-cnn-12-6", text, params={"max_new_tokens":120})
        return {"source":"hf","summary": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("distilbert-base-uncased-finetuned-sst-2-english", text)
        return {"source":"hf","sentiment": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/qa")
async def qa(question: str = Form(...), context: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("deepset/roberta-base-squad2", {"question": question, "context": context})
        return {"source":"hf","answer": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/codegen")
async def codegen(prompt: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("Salesforce/codegen-350M-multi", prompt, params={"max_new_tokens":300})
        return {"source":"hf","code": str(resp)}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Speech endpoints
# -----------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("openai/whisper-small", data, is_binary=True)
        return {"source":"hf","text": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/tts")
async def tts(text: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("espnet/kan-bayashi_ljspeech_vits", text)
        return {"source":"hf","data": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Vision endpoints
# -----------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not Image: raise HTTPException(status_code=503, detail="PIL not available")
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("facebook/detr-resnet-50", data, is_binary=True)
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("Salesforce/blip-image-captioning-large", data, is_binary=True)
        return {"source":"hf","caption": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/vqa")
async def vqa(file: UploadFile = File(...), question: str = Form(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("Salesforce/blip-vqa-base", {"image": data, "question": question})
        return {"source":"hf","answer": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form("")):
    if HF_TOKEN:
        img_data = await image.read()
        mask_data = await mask.read()
        resp = await hf_inference("stabilityai/stable-diffusion-x4-inpainting",
                                  {"image": img_data, "mask": mask_data, "prompt": prompt})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(2)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("nateraw/real-esrgan", data, params={"scale": scale})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if HF_TOKEN:
        data = await file.read()
        resp = await hf_inference("microsoft/trocr-base-handwritten", data, is_binary=True)
        return {"source":"hf","text": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

# -----------------------
# Music & 3D placeholders
# -----------------------
@app.post("/music")
async def music(prompt: str = Form(...), duration: int = Form(10)):
    if HF_TOKEN:
        resp = await hf_inference("facebook/musicgen-small", prompt, params={"duration": duration})
        return {"source":"hf","result": resp}
    raise HTTPException(status_code=503, detail="HF_TOKEN not set")

@app.post("/3d")
async def text_to_3d(prompt: str = Form(...)):
    if HF_TOKEN:
        resp = await hf_inference("openai/point-e", prompt)
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
    if not supabase: return {"error":"Supabase not configured"}
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
    uvicorn.run(app, host="0.0.0.0", port=port)
