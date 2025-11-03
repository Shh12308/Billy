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

# === APP SETUP ===
app = FastAPI(title="Billy-Free AI v2 (by GoldBoy)")

# Allow all origins for demo / frontend use
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
    "project": "Billy-Free AI v2",
    "description": "An open, self-hosted AI project built for learning and creative experimentation."
}

# === MEMORY (ChromaDB) ===
chroma = chromadb.Client()
if "billy_memory" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_memory")
memory = chroma.get_collection("billy_memory")

# === DEVICE ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODELS ===
print("Loading models... this may take a minute.")

# 1️⃣ Chat/Text Generation — Flan-T5-Small (light + smart)
chat_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
chat_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# 2️⃣ Image Captioning — BLIP base
vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# 3️⃣ Speech Recognition — Whisper tiny
whisper_model = whisper.load_model("tiny")

# 4️⃣ Text-to-Speech — lightweight Tacotron2
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

print("✅ All models loaded successfully.")

# === MEMORY HELPERS ===
def store_context(prompt, answer):
    try:
        memory.add(
            documents=[f"{prompt} => {answer}"],
            metadatas=[{"type": "chat"}],
            ids=[f"id_{abs(hash(prompt))}"]
        )
    except Exception as e:
        print("Memory store failed:", e)

def retrieve_context(prompt, n=3):
    try:
        results = memory.query(query_texts=[prompt], n_results=n)
        return "\n".join(results["documents"][0]) if results["documents"] else ""
    except Exception:
        return ""

# === ROUTES ===
@app.get("/")
def root():
    return {
        "message": "🤖 Billy-Free AI v2 (Render Optimized) is online!",
        "creator": CREATOR_INFO
    }

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    # Add built-in self-awareness
    if any(x in prompt.lower() for x in ["who made you", "creator", "developer", "owner"]):
        return {"response": f"I was created by {CREATOR_INFO['name']}, a {CREATOR_INFO['age']}-year-old developer from {CREATOR_INFO['location']}."}

    context = retrieve_context(prompt)
    full_prompt = f"{context}\nUser: {prompt}\nAssistant:" if context else f"User: {prompt}\nAssistant:"
    inputs = chat_tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    outputs = chat_model.generate(**inputs, max_new_tokens=120, temperature=0.7)
    text = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    store_context(prompt, text)
    return {"response": text.strip()}

@app.post("/tts")
async def tts(prompt: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        tts_model.tts_to_file(text=prompt, file_path=temp_file.name)
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
    os.remove(temp_file.name)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe("temp.wav")
    os.remove("temp.wav")
    return {"transcription": result["text"]}

@app.post("/image-caption")
async def image_caption(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt").to(device)
    output_ids = vision_model.generate(**inputs)
    caption = vision_processor.decode(output_ids[0], skip_special_tokens=True)
    return {"caption": caption}

@app.get("/stream")
async def stream():
    async def event_stream():
        for i in range(10):
            yield f"data: Stream update #{i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/memory")
def get_memory():
    try:
        docs = memory.get()["documents"]
        return {"memory": docs}
    except Exception as e:
        return {"error": str(e)}

# === RUN APP ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Render sets this automatically
    print(f"Starting Billy-Free AI v2 by {CREATOR_INFO['name']} on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
