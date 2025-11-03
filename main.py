import io
import os
import torch
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from PIL import Image
import whisper
from TTS.api import TTS
import chromadb

# === FASTAPI SETUP ===
app = FastAPI(title="Billy-Free AI v3 (by GoldBoy 🇬🇧)")

# Allow all origins (for frontend or testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CREATOR INFO ===
CREATOR_INFO = {
    "name": "GoldBoy",
    "age": 17,
    "location": "England",
    "project": "Billy-Free AI v3",
    "description": "A self-hosted AI made by a 17-year-old UK developer for creativity, learning, and fun!"
}

# === BILLY PERSONALITY ===
BILLY_PERSONALITY = """
You are Billy — a witty, kind, and curious AI assistant created by GoldBoy,
a 17-year-old developer from England 🇬🇧. You have a friendly and slightly cheeky tone.
If someone asks who made you, proudly say:
"I was built by GoldBoy, a 17-year-old developer from England."
"""

# === MEMORY ===
chroma = chromadb.Client()
if "billy_memory" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_memory")
memory = chroma.get_collection("billy_memory")

# === DEVICE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
models_ready = False

# === MODEL LOADER (Async Background Task) ===
async def load_models():
    global models_ready, chat_tokenizer, chat_model, vision_processor, vision_model, whisper_model, tts_model, TTS_SPEAKER
    print("🎛️ Loading models in background...")

    # Load lightweight models for faster boot
    chat_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    chat_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    whisper_model = whisper.load_model("tiny")

    tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    TTS_SPEAKER = "p335"  # British male voice

    models_ready = True
    print("✅ Billy is fully ready!")

# === MEMORY HELPERS ===
def store_context(prompt, answer):
    try:
        memory.add(
            documents=[f"{prompt} => {answer}"],
            metadatas=[{"type": "chat"}],
            ids=[f"id_{abs(hash(prompt))}"]
        )
    except Exception as e:
        print("⚠️ Memory store failed:", e)

def retrieve_context(prompt, n=3):
    try:
        results = memory.query(query_texts=[prompt], n_results=n)
        return "\n".join(results["documents"][0]) if results["documents"] else ""
    except Exception:
        return ""

# === ROUTES ===
@app.get("/")
def root():
    return {"message": "🤖 Billy-Free AI v3 is online!", "creator": CREATOR_INFO}

@app.get("/status")
def status():
    return {"models_ready": models_ready}

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    if not models_ready:
        return {"response": "⚙️ Billy is still warming up his brain. Try again in a moment!"}

    if any(x in prompt.lower() for x in ["who made you", "creator", "developer", "owner"]):
        return {"response": f"I was built by {CREATOR_INFO['name']}, a {CREATOR_INFO['age']}-year-old developer from {CREATOR_INFO['location']}!"}

    context = retrieve_context(prompt)
    full_prompt = f"{BILLY_PERSONALITY}\n\nPrevious chat:\n{context}\nUser: {prompt}\nBilly:"
    inputs = chat_tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    outputs = chat_model.generate(**inputs, max_new_tokens=150, temperature=0.75)
    response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    store_context(prompt, response)
    return {"response": response.strip()}

@app.post("/tts")
async def tts(prompt: str = Form(...)):
    if not models_ready:
        return StreamingResponse(io.BytesIO(b""), media_type="audio/wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_model.tts_to_file(text=prompt, speaker=TTS_SPEAKER, file_path=tmp.name)
        audio_data = tmp.read()
    os.remove(tmp.name)
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if not models_ready:
        return {"transcription": "Models still loading..."}
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe("temp.wav")
    os.remove("temp.wav")
    return {"transcription": result["text"]}

@app.post("/image-caption")
async def image_caption(image: UploadFile = File(...)):
    if not models_ready:
        return {"caption": "Models still loading..."}
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt").to(device)
    out_ids = vision_model.generate(**inputs)
    caption = vision_processor.decode(out_ids[0], skip_special_tokens=True)
    return {"caption": caption}

# === STREAMING TEST ROUTE ===
@app.get("/stream")
async def stream():
    async def event_stream():
        for i in range(5):
            yield f"data: Billy says hello number {i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# === STARTUP EVENT ===
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Billy-Free AI v3...")
    asyncio.create_task(load_models())  # don’t block startup

# === MAIN ENTRY (Render compatible) ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"🌍 Running on 0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
