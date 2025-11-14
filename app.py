"""
Billy-AllInOne v1.1 — Unified AI backend (FastAPI)
By GoldBoy — includes Supabase integration and many optional endpoints.
Usage:
  - Set SUPABASE_URL & SUPABASE_KEY to enable Supabase features
  - Set HF_TOKEN to enable Hugging Face Inference API fallbacks
  - Optional local deps: torch, transformers, diffusers, whisper, faster-whisper, TTS, sentence-transformers, torchaudio, real-esrgan, openvino, etc.
Notes:
  - Each endpoint will report whether it's using a local model or HF Inference.
  - Heavy endpoints (3D modelling, video gen, pose, face recognition, medical imaging) are provided as placeholders that either call HF or require you to add your own model.
"""
import uvicorn
import os, io, uuid, json, time, tempfile, traceback
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import threading

# Optional heavy imports guarded
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
except Exception:
    pipeline = AutoTokenizer = AutoModelForSeq2SeqLM = AutoModelForCausalLM = None

try:
    import whisper
except Exception:
    whisper = None

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None

try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None

try:
    import httpx
except Exception:
    httpx = None

# Supabase client
try:
    from supabase import create_client, Client as SupabaseClient
except Exception:
    create_client = None
    SupabaseClient = None

# Computer vision local models (guarded)
try:
    import torchvision.transforms as T
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
except Exception:
    T = None
    fasterrcnn_resnet50_fpn = None

# Real-ESRGAN wrapper attempt
try:
    # placeholder import; most envs won't have it
    import realesrgan
except Exception:
    realesrgan = None

# Helper: read env
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face Inference API token (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")  # for voice cloning if you want
HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/v1/models/{}"

# Basic app
app = FastAPI(title="Billy-AllInOne (GoldBoy Edition)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Creator info + personality
CREATOR = {"name": "GoldBoy", "age": 17, "location": "England"}
PERSONALITY = f"You are Billy, a friendly assistant built by {CREATOR['name']} (age {CREATOR['age']}, {CREATOR['location']})."

# Supabase client init
supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)
else:
    print("⚠️ Supabase not configured (SUPABASE_URL/SUPABASE_KEY missing or supabase lib not installed)")

HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/v1/models/{}"

async def hf_inference(model_id, inputs, params=None, is_binary=False):
    """
    Hugging Face router-compatible inference.
    Supports:
      - text (JSON)
      - image/audio bytes (binary multipart)
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing")

    url = HF_INFERENCE_URL.format(model_id)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    async with httpx.AsyncClient(timeout=120) as client:
        if is_binary:
            # For images/audio
            files = {"file": inputs}
            data = {"parameters": json.dumps(params or {})}
            resp = await client.post(url, headers=headers, data=data, files=files)
        else:
            # For text
            payload = {"inputs": inputs}
            if params:
                payload["parameters"] = params
            resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except:
        return resp.text
# Small local model loader examples (guarded)
_local_models = {}

def load_local_nlp():
    """Load light local text models (if transformers installed)"""
    if "nlp" in _local_models:
        return _local_models["nlp"]
    if AutoTokenizer and AutoModelForSeq2SeqLM:
        try:
            tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            _local_models["nlp"] = (tok, model)
            print("✅ Loaded local flan-t5-small")
            return _local_models["nlp"]
        except Exception as e:
            print("⚠️ Local flan load failed:", e)
    return None

def available_local_det():
    if fasterrcnn_resnet50_fpn and T:
        try:
            if "det" not in _local_models:
                det = fasterrcnn_resnet50_fpn(pretrained=True).eval()
                _local_models["det"] = det
                print("✅ Local detector ready")
            return _local_models["det"]
        except Exception as e:
            print("⚠️ Local detector failed:", e)
    return None

# Simple logging to Supabase table 'ai_logs' if available
def log_event(name: str, payload: dict):
    if not supabase:
        return
    try:
        supabase.table("ai_logs").insert({"event": name, "payload": payload}).execute()
    except Exception as e:
        print("⚠️ supabase log failed:", e)

def start_gradio():
    import requests

    BASE = "http://localhost:8080"

    def chat_ui(prompt):
        resp = requests.post(f"{BASE}/chat", data={"prompt": prompt, "user_id": "guest"}).json()
        return resp.get("response", "[No response]")

    with gr.Blocks() as demo:
        prompt_input = gr.Textbox(label="Prompt")
        chat_output = gr.Textbox(label="Response")
        prompt_input.submit(chat_ui, prompt_input, chat_output)
    demo.launch(server_name="0.0.0.0", server_port=8080)  # Gradio on a separate port

# -------------------------
# ROUTES: Basic & Health
# -------------------------
@app.get("/")
def index():
    return {"message": "Billy AllInOne — endpoints: /chat, /tts, /stt, /translate, /detect, /inpaint, /upscale, /ocr, /vqa, /codegen, /qa, /music, /3d, /benchmark (optional)"}

@app.get("/health")
def health():
    return {"ok": True, "supabase": bool(supabase), "hf_token": bool(HF_TOKEN), "torch": bool(torch)}

# -------------------------
# TEXT: chat, translate, summarize, sentiment, QA, codegen
# -------------------------
@app.post("/chat")
async def chat(prompt: str = Form(...), user_id: Optional[str] = Form("guest")):
    """
    Unified chat endpoint:
      1. Tries local Flan-T5 if installed
      2. Falls back to Hugging Face Inference API if HF_TOKEN is set
      3. Safely handles HF JSON output to avoid 'detail not found'
    """
    try:
        # --- 1. Load memory from Supabase if available ---
        memory_text = ""
        if supabase:
            try:
                rows = supabase.table("chat_memory").select("*")\
                    .eq("user_id", user_id).order("created_at", {"ascending": False}).limit(5).execute()
                data = getattr(rows, 'data', []) or []
                memory_text = "\n".join([f"User:{r.get('prompt','')}\nBot:{r.get('response','')}" for r in data])
            except Exception:
                memory_text = ""

        # --- 2. Build full prompt ---
        prompt_full = f"{PERSONALITY}\nPrevious:\n{memory_text}\nUser: {prompt}\nBilly:"

        # --- 3. Try local model first ---
        local = load_local_nlp()
        if local:
            try:
                tok, model = local
                inputs = tok(prompt_full, return_tensors="pt", truncation=True)
                outputs = model.generate(**inputs, max_new_tokens=200)
                resp = tok.decode(outputs[0], skip_special_tokens=True)

                log_event("chat_local", {"user": user_id, "prompt": prompt})

                # store memory
                if supabase:
                    try:
                        supabase.table("chat_memory").insert({
                            "user_id": user_id, 
                            "prompt": prompt, 
                            "response": resp, 
                            "created_at": int(time.time())
                        }).execute()
                    except Exception:
                        pass

                return {"source": "local", "response": resp}
            except Exception as e:
                print("⚠️ Local model failed, falling back to HF:", e)

        # --- 4. Hugging Face fallback ---
        if HF_TOKEN:
  hf_model = "meta-llama/Llama-3-7b-chat-hf" # choose any valid HF model
            try:
                j = await hf_inference(hf_model, prompt_full, params={"max_new_tokens":200})

                # --- Safe parsing of HF output ---
                resp = ""
                if isinstance(j, list) and len(j) > 0:
                    if isinstance(j[0], dict) and "generated_text" in j[0]:
                        resp = j[0]["generated_text"]
                    else:
                        resp = str(j[0])
                elif isinstance(j, dict) and j.get("generated_text"):
                    resp = j["generated_text"]
                else:
                    resp = str(j)

                log_event("chat_hf", {"user": user_id, "prompt": prompt})

                # store memory
                if supabase:
                    try:
                        supabase.table("chat_memory").insert({
                            "user_id": user_id,
                            "prompt": prompt,
                            "response": resp,
                            "created_at": int(time.time())
                        }).execute()
                    except Exception:
                        pass

                return {"source": "hf", "response": resp}

            except Exception as e:
                # HF call failed
                print("⚠️ HF inference failed:", e)
                raise HTTPException(status_code=503, detail=f"HF call failed: {str(e)}")

        # --- 5. No model available ---
        raise HTTPException(status_code=503, detail="No local model available and HF_TOKEN not set")

    except Exception as e:
        # unexpected error
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate(text: str = Form(...), to: str = Form("fr")):
    """
    Simple translation endpoint:
      - local pipeline if available
      - HF fallback (Helsinki models)
    """
    try:
        # Prefer local pipeline
        if pipeline:
            try:
                # choose model based on `to`
                model_map = {"fr": "Helsinki-NLP/opus-mt-en-fr", "es": "Helsinki-NLP/opus-mt-en-es"}
                model = model_map.get(to, "Helsinki-NLP/opus-mt-en-fr")
                p = pipeline("translation_en_to_XX", model=model)  # may fail; pipelines vary
                out = p(text, max_length=400)
                return {"source": "local", "translation": out[0]["translation_text"]}
            except Exception:
                pass
        # HF fallback
        if HF_TOKEN:
            model = {"fr":"Helsinki-NLP/opus-mt-en-fr","es":"Helsinki-NLP/opus-mt-en-es"}.get(to,"Helsinki-NLP/opus-mt-en-fr")
            j = await hf_inference(model, text, params={"max_new_tokens":200})
            # HF text response parsing
            if isinstance(j, list) and isinstance(j[0], dict):
                if j[0].get("translation_text"): return {"source":"hf","translation": j[0]["translation_text"]}
            return {"source":"hf","translation": str(j)}
        raise HTTPException(status_code=503, detail="No translator available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    try:
        if pipeline:
            try:
                p = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                out = p(text, max_length=120, min_length=30, do_sample=False)
                return {"source":"local", "summary": out[0]["summary_text"]}
            except Exception:
                pass
        if HF_TOKEN:
            j = await hf_inference("sshleifer/distilbart-cnn-12-6", text, params={"max_new_tokens":120})
            return {"source":"hf", "summary": str(j)}
        raise HTTPException(status_code=503, detail="Summarization model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    try:
        if pipeline:
            p = pipeline("sentiment-analysis")
            out = p(text)
            return {"source":"local","sentiment": out}
        if HF_TOKEN:
            j = await hf_inference("distilbert-base-uncased-finetuned-sst-2-english", text)
            return {"source":"hf","sentiment": j}
        raise HTTPException(status_code=503, detail="Sentiment model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def qa(question: str = Form(...), context: str = Form(...)):
    """
    Question answering over provided context (extractive).
    """
    try:
        if pipeline:
            p = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            out = p(question=question, context=context)
            return {"source":"local","answer": out}
        if HF_TOKEN:
            j = await hf_inference("deepset/roberta-base-squad2", {"question": question, "context": context})
            return {"source":"hf","answer": j}
        raise HTTPException(status_code=503, detail="QA model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/codegen")
async def codegen(prompt: str = Form(...), lang: str = Form("python")):
    """
    Simple code generation endpoint. Uses a small code model or HF.
    """
    try:
        # Prefer small local codegen if installed
        if AutoModelForCausalLM and AutoTokenizer:
            try:
                model_name = "Salesforce/codegen-350M-multi"
                tok = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                inputs = tok(prompt, return_tensors="pt")
                outs = model.generate(**inputs, max_new_tokens=300)
                code = tok.decode(outs[0], skip_special_tokens=True)
                return {"source":"local","code": code}
            except Exception:
                pass
        if HF_TOKEN:
            model = "Salesforce/codegen-350M-multi"
            j = await hf_inference(model, prompt, params={"max_new_tokens":300})
            return {"source":"hf","code": str(j)}
        raise HTTPException(status_code=503, detail="Code model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# SPEECH: TTS, STT, Voice clone
# -------------------------
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Speech-to-text:
      - tries local whisper (if installed)
      - else HF or reports not available
    """
    try:
        data = await file.read()
        tmp = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
        with open(tmp, "wb") as f:
            f.write(data)
        if whisper:
            m = whisper.load_model("tiny")
            o = m.transcribe(tmp)
            os.remove(tmp)
            return {"source":"local","text": o.get("text","")}
        if HF_TOKEN:
            # HF STT model e.g. openai/whisper-small or facebook/wav2vec2
            j = await hf_inference("openai/whisper-small", open(tmp, "rb").read())
            return {"source":"hf","text": j}
        os.remove(tmp)
        raise HTTPException(status_code=503, detail="STT not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def tts(text: str = Form(...), voice: Optional[str] = Form(None)):
    """
    TTS:
      - tries CoquiTTS if installed
      - fallback: HF TTS via 'tts' models or raise
    """
    try:
        out_path = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
        if CoquiTTS:
            tts = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=False)
            tts.tts_to_file(text=text, speaker=voice or None, file_path=out_path)
            return FileResponse(out_path, media_type="audio/mpeg")
        if HF_TOKEN:
            # HF TTS endpoint (e.g., facebook/fastspeech2-* not always available via inference endpoint)
            # We'll call a generic model id if available
            resp = await hf_inference("espnet/kan-bayashi_ljspeech_vits", text)
            # HF returns base64 audio or bytes depending on model; we return JSON
            return {"source":"hf","data": resp}
        raise HTTPException(status_code=503, detail="TTS backend not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_clone")
async def voice_clone(text: str = Form(...), seed_audio: UploadFile = File(None)):
    """
    Voice cloning: Attempt ElevenLabs if ELEVEN_API_KEY present, else suggest enabling.
    This endpoint will return a URL or bytes depending on integration.
    """
    if ELEVEN_API_KEY:
        # Implementation would call ElevenLabs API. Provide instructions only.
        return {"status":"ok","note":"ElevenLabs voice cloning is available if you wire ELEVEN_API_KEY. Implement API POST call."}
    return {"error":"Voice cloning requires ElevenLabs or other commercial API. Set ELEVEN_API_KEY"}

# -------------------------
# VISION: detect, caption, vqa, inpaint, background removal, upscaling, ocr
# -------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Object detection: uses local torchvision fasterrcnn if available, else HF vision-detector if HF_TOKEN set.
    """
    try:
        data = await file.read()
        tmp = f"/tmp/{uuid.uuid4().hex}.jpg"
        with open(tmp, "wb") as f:
            f.write(data)
        # Local
        det = available_local_det()
        if det and Image:
            img = Image.open(tmp).convert("RGB")
            transform = T.Compose([T.ToTensor()])
            tensor = transform(img)
            with torch.no_grad():
                preds = det([tensor])[0]
            out = []
            for box, score, lab in zip(preds["boxes"], preds["scores"], preds["labels"]):
                if float(score) > 0.5:
                    out.append({"box":[float(x) for x in box], "score": float(score), "label": int(lab)})
            return {"source":"local", "detections": out}
        # HF fallback -- use "facebook/detr-resnet-50" or "google/vision-detector" via inference
        if HF_TOKEN:
            # HF inference expects bytes
            resp = await hf_inference("facebook/detr-resnet-50", open(tmp, "rb").read())
            return {"source":"hf", "result": resp}
        raise HTTPException(status_code=503, detail="No detector available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    """
    Image captioning via BLIP (local) or HF.
    """
    try:
        data = await file.read()
        tmp = f"/tmp/{uuid.uuid4().hex}.jpg"
        with open(tmp, "wb") as f:
            f.write(data)
        # local BLIP
        try:
            if pipeline:
                p = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
                out = p(tmp)
                return {"source":"local","caption": out[0]["generated_text"] if isinstance(out, list) and out else out}
        except Exception:
            pass
        if HF_TOKEN:
            out = await hf_inference("Salesforce/blip-image-captioning-large", open(tmp, "rb").read(), is_binary=True)
            return {"source":"hf","caption": out}
        raise HTTPException(status_code=503, detail="Caption model not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vqa")
async def vqa(file: UploadFile = File(...), question: str = Form(...)):
    """
    Visual Question Answering: local BLIP VQA or HF fallback.
    """
    try:
        tmp = f"/tmp/{uuid.uuid4().hex}.jpg"
        with open(tmp, "wb") as f:
            f.write(await file.read())
        if pipeline:
            try:
                p = pipeline("vqa", model="Salesforce/blip-vqa-base")
                out = p(tmp, question)
                return {"source":"local","answer": out}
            except Exception:
                pass
        if HF_TOKEN:
            out = await hf_inference("Salesforce/blip-vqa-base", {"image": open(tmp, "rb").read(), "question": question})
            return {"source":"hf","answer": out}
        raise HTTPException(status_code=503, detail="VQA backend not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form("")):
    """
    Image editing/inpainting:
      - uses diffusers in local env if installed
      - else fallback to HF image-edit models via inference
    """
    try:
        # Save
        img_tmp = f"/tmp/{uuid.uuid4().hex}_img.png"
        mask_tmp = f"/tmp/{uuid.uuid4().hex}_mask.png"
        with open(img_tmp, "wb") as f:
            f.write(await image.read())
        with open(mask_tmp, "wb") as f:
            f.write(await mask.read())
        # Attempt HF
        if HF_TOKEN:
            # There are HF image-edit endpoints (e.g., stability-inpainting)
            out = await hf_inference("stabilityai/stable-diffusion-x4-inpainting", {"image": open(img_tmp, "rb").read(), "mask": open(mask_tmp, "rb").read(), "prompt": prompt})
            return {"source":"hf","result": out}
        raise HTTPException(status_code=503, detail="Inpainting requires local diffusers or HF_TOKEN")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_bg")
async def remove_bg(image: UploadFile = File(...)):
    """
    Background removal:
      - uses HF 'remove-background' models if available, otherwise requires external service.
    """
    try:
        tmp = f"/tmp/{uuid.uuid4().hex}.png"
        with open(tmp, "wb") as f:
            f.write(await image.read())
        if HF_TOKEN:
            out = await hf_inference("photoroom/background-removal", open(tmp, "rb").read(), is_binary=True)
            return {"source":"hf", "result": out}
        return {"error": "Enable HF_TOKEN or provide local background removal model (u2net/robust)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upscale")
async def upscale(image: UploadFile = File(...), scale: int = Form(2)):
    """
    Image upscaling:
      - local Real-ESRGAN wrapper if installed (realesrgan)
      - HF fallback (e.g., xinntao/Real-ESRGAN)
    """
    try:
        tmp = f"/tmp/{uuid.uuid4().hex}.png"
        with open(tmp, "wb") as f:
            f.write(await image.read())
        if realesrgan:
            # placeholder usage
            return {"source":"local","note":"Run realesrgan upscaler here (local installation required)."}
        if HF_TOKEN:
            out = await hf_inference("nateraw/real-esrgan", open(tmp, "rb").read(), params={"scale": scale})
            return {"source":"hf", "result": out}
        raise HTTPException(status_code=503, detail="Upscaler not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    OCR: uses HF TrOCR model or local pipeline
    """
    try:
        tmp = f"/tmp/{uuid.uuid4().hex}.png"
        with open(tmp, "wb") as f:
            f.write(await file.read())
        if HF_TOKEN:
            out = await hf_inference("microsoft/trocr-base-handwritten", open(tmp, "rb").read(), is_binary=True)
            return {"source":"hf","text": out}
        # No local fallback in this minimal script
        raise HTTPException(status_code=503, detail="OCR requires HF_TOKEN or local trocr")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# MUSIC
# -------------------------
@app.post("/music")
async def music(prompt: str = Form(...), duration: int = Form(10)):
    """
    Music generation:
      - Try MusicGen if installed
      - Else recommend using Hugging Face AudioGen/MusicGen via HF_TOKEN
    """
    try:
        # Local heavy music libs likely not present; we rely on HF inference
        if HF_TOKEN:
            out = await hf_inference("facebook/musicgen-small", prompt, params={"duration": duration})
            return {"source":"hf","result": out}
        return {"error":"Music generation requires MusicGen locally or HF_TOKEN for remote inference"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# 3D modeling & character animation placeholders
# -------------------------
@app.post("/3d")
async def text_to_3d(prompt: str = Form(...)):
    """
    Text-to-3D placeholder:
      - This is an emerging area. Use HF 3D models or external APIs (e.g., DreamFusion-like or Point-E).
      - If HF_TOKEN available, call a 3D model inference (if present) or return instructions.
    """
    if HF_TOKEN:
        try:
            out = await hf_inference("openai/point-e", prompt)
            return {"source":"hf","result": out}
        except Exception:
            return {"error":"HF 3D model call failed; you may need to host Point-E locally or use a provider."}
    return {"error":"Text-to-3D requires a specialized model (Point-E, DreamFusion). Add HF_TOKEN or run locally."}

@app.post("/character_animation")
async def character_animation(prompt: str = Form(...)):
    """
    Character animation placeholder: integrates mocap -> rig -> animate pipeline. Very project-specific.
    """
    return {"error":"Character animation is project-specific. Suggest using Blender + retargeting scripts or external API. I can scaffold if you want."}

# -------------------------
# ANALYSIS: medical imaging, financial (stubs & safe advice)
# -------------------------
@app.post("/medical/scan_analysis")
async def medical_scan_analysis(file: UploadFile = File(...)):
    """
    Medical imaging: This is sensitive and high-stakes.
    Provide only a placeholder that returns a recommendation to use validated, certified models.
    """
    return {"error":"Medical diagnostics requires regulatory-approved models and clinical validation. Do not use for diagnosis. I can show how to connect to a validated model or run research-only models with disclaimers."}

@app.post("/financial/analyze")
async def financial_analyze(text: str = Form(...)):
    """
    Financial analysis: sentiment, ratio extraction, summary
    """
    try:
        # reuse sentiment + summarization pipelines
        s = await sentiment(text=text)
        su = await summarize(text=text)
        return {"sentiment": s, "summary": su}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# OTHER: benchmarking, fine-tune helpers, dataset creation
# -------------------------
@app.post("/benchmark")
async def benchmark_model(model_id: str = Form(...), sample_input: str = Form("Hello world")):
    """
    Simple benchmark helper: times HF inference or local model generation
    """
    result = {"model": model_id}
    start = time.time()
    try:
        if HF_TOKEN:
            j = await hf_inference(model_id, sample_input, params={"max_new_tokens": 64})
            result["source"] = "hf"
            result["output_sample"] = str(j)[:200]
        else:
            # local attempt: use pipeline if possible
            if pipeline:
                p = pipeline("text-generation", model=model_id)
                out = p(sample_input, max_length=80)
                result["source"] = "local"
                result["output_sample"] = out[0]["generated_text"]
            else:
                raise HTTPException(status_code=503, detail="No HF token and pipeline not available")
    except Exception as e:
        result["error"] = str(e)
    result["elapsed"] = time.time() - start
    return result

@app.post("/finetune/create_job")
def finetune_job(dataset_name: str = Form(...), model: str = Form("meta-llama/Llama-2-7b-chat")):
    """
    Fine-tuning helper stub: just records request and instructs how to proceed.
    Actual fine-tuning is infra-heavy and differs by model/provider.
    """
    # store request to supabase if available
    if supabase:
        try:
            supabase.table("ft_jobs").insert({"dataset": dataset_name, "model": model, "created_at": int(time.time())}).execute()
        except Exception:
            pass
    return {"status":"queued","note":"This is a helper stub. For actual fine-tuning use Hugging Face Training or LoRA adapters; I can scaffold training scripts."}

@app.post("/dataset/create")
async def dataset_create(files: List[UploadFile] = File(...), dataset_name: str = Form(...)):
    """
    Create a simple dataset on disk (for manual fine-tuning later). Saves to /tmp/<dataset_name>/
    """
    base = f"/tmp/dataset_{dataset_name}_{uuid.uuid4().hex}"
    os.makedirs(base, exist_ok=True)
    saved = []
    for f in files:
        path = os.path.join(base, f.filename)
        with open(path, "wb") as fh:
            fh.write(await f.read())
        saved.append(path)
    return {"created": True, "storage_path": base, "files": saved}

# -------------------------
# DATA VISUALISATION: create charts from JSON / CSV
# -------------------------
@app.post("/viz")
async def viz(data_json: str = Form(...), chart_type: str = Form("line")):
    """
    Simple data visualization: accepts JSON {"x":[...], "y":[...]} or array of objects.
    Returns PNG image file.
    """
    import matplotlib.pyplot as plt
    import base64
    try:
        data = json.loads(data_json)
        fig, ax = plt.subplots(figsize=(8,4))
        if chart_type == "line":
            x = data.get("x")
            y = data.get("y")
            ax.plot(x, y)
        elif chart_type == "bar":
            labels = [str(i) for i in range(len(data.get("y",[])))]
            ax.bar(labels, data.get("y",[]))
        else:
            ax.text(0.5, 0.5, "Chart type not supported", ha="center")
        ax.set_title(data.get("title","Chart"))
        tmp = f"/tmp/viz_{uuid.uuid4().hex}.png"
        fig.savefig(tmp, bbox_inches="tight")
        plt.close(fig)
        return FileResponse(tmp, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# PLACEHOLDERS: pose estimation, face recognition, anomaly detection, recommender
# -------------------------
@app.post("/pose_estimate")
def pose_estimate(file: UploadFile = File(...)):
    return {"error":"Pose estimation endpoint requires MoveNet/MediaPipe or OpenPose integrations. I can add it on request."}

@app.post("/face_recognition")
def face_recognition(file: UploadFile = File(...)):
    return {"error":"Face recognition is privacy sensitive. I can add a local face-rec library integration (face_recognition) if you accept privacy/legal considerations."}

@app.post("/anomaly_detect")
def anomaly_detect(data_json: str = Form(...)):
    return {"note":"Anomaly detection is data-specific; I can suggest IsolationForest, LSTM, or feature-based pipelines. Provide sample data to scaffold."}

@app.post("/recommend")
def recommend(user_id: str = Form(...), k: int = Form(5)):
    """
    Simple item-based placeholder recommender: return k random demo items.
    """
    demo = [{"id": f"item_{i}", "title": f"Demo Item {i}"} for i in range(1, 21)]
    import random
    return {"user": user_id, "recommendations": random.sample(demo, k)}

# -------------------------
# UTILITY: register user (supabase), logs
# -------------------------
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

# -------------------------
# FINAL: expose creator info and tips
# -------------------------
@app.get("/creator")
def creator():
    return {"creator": CREATOR, "note": "Include this info when the model is asked who built it."}


# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    threading.Thread(target=start_gradio, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
