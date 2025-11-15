# ===========================
# Billy Small-Model Edition
# Fully Render Compatible
# ===========================

import uvicorn
import os, json, uuid, time, asyncio
import httpx
from fastapi import FastAPI, Form, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# HuggingFace Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set!")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Small-model endpoints (Render friendly)
CHAT_MODEL = "google/flan-t5-large"
TRANSLATE_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATE_ES = "Helsinki-NLP/opus-mt-en-es"
SUMMARIZE_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
STT_MODEL = "openai/whisper-small"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DETECT_MODEL = "facebook/detr-resnet-50"
OCR_MODEL = "microsoft/trocr-base-handwritten"
VQA_MODEL = "Salesforce/blip-vqa-base"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------
# Helper HuggingFace request
# --------------------------
async def hf_call(model: str, inputs, params=None, binary=False):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF_TOKEN not configured")

    async with httpx.AsyncClient(timeout=120) as client:
        if binary:
            resp = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=HEADERS,
                content=inputs
            )
        else:
            payload = {"inputs": inputs}
            if params:
                payload["parameters"] = params

            resp = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=HEADERS,
                json=payload
            )

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    try:
        return resp.json()
    except:
        return resp.text


# --------------------------
# BASIC ROUTES
# --------------------------
@app.get("/")
def home():
    return {"msg": "Billy Small Model Edition — working 100%"}

@app.get("/health")
def health():
    return {"ok": True, "hf": bool(HF_TOKEN)}


# --------------------------
# CHAT
# --------------------------
@app.post("/chat")
async def chat(prompt: str = Form(...)):
    full_prompt = f"User: {prompt}\nAssistant:"
    resp = await hf_call(CHAT_MODEL, full_prompt, params={"max_new_tokens": 150})

    text = resp[0]["generated_text"] if isinstance(resp, list) else str(resp)
    return {"response": text}


# --------------------------
# STREAMING WEBSOCKET
# --------------------------
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    try:
        msg = await ws.receive_json()
        prompt = msg.get("prompt", "")

        if not prompt:
            await ws.send_json({"error": "No prompt"})
            return

        # Normal HF call first
        resp = await hf_call(CHAT_MODEL, f"User: {prompt}\nAssistant:", params={"max_new_tokens": 150})
        text = resp[0]["generated_text"]

        # Stream word by word
        for w in text.split():
            await ws.send_json({"delta": w + " "})
            await asyncio.sleep(0.02)

        await ws.send_json({"done": True})
        await ws.close()

    except Exception as e:
        await ws.send_json({"error": str(e)})
        await ws.close()


# --------------------------
# TRANSLATE
# --------------------------
@app.post("/translate")
async def translate(text: str = Form(...), to: str = Form("fr")):
    model = TRANSLATE_FR if to == "fr" else TRANSLATE_ES
    resp = await hf_call(model, text)
    return {"translation": resp[0]["translation_text"]}


# --------------------------
# SUMMARIZE
# --------------------------
@app.post("/summarize")
async def summarize(text: str = Form(...)):
    resp = await hf_call(SUMMARIZE_MODEL, text)
    return {"summary": resp}


# --------------------------
# SENTIMENT
# --------------------------
@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    resp = await hf_call(SENTIMENT_MODEL, text)
    return {"sentiment": resp}


# --------------------------
# SPEECH-TO-TEXT
# --------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    data = await file.read()
    resp = await hf_call(STT_MODEL, data, binary=True)
    return {"text": resp}


# --------------------------
# CAPTION IMAGE
# --------------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    data = await file.read()
    resp = await hf_call(CAPTION_MODEL, data, binary=True)
    return {"caption": resp}


# --------------------------
# OCR
# --------------------------
@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    resp = await hf_call(OCR_MODEL, data, binary=True)
    return {"text": resp}


# --------------------------
# VISION DETECTION
# --------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    resp = await hf_call(DETECT_MODEL, data, binary=True)
    return {"result": resp}


# --------------------------
# VQA
# --------------------------
@app.post("/vqa")
async def vqa(file: UploadFile = File(...), question: str = Form(...)):
    img_bytes = await file.read()
    resp = await hf_call(VQA_MODEL, {"image": img_bytes, "question": question})
    return {"answer": resp}


# --------------------------
# MAIN ENTRY
# --------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
