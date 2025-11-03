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
app = FastAPI(title="Billy-Free AI v3 (by GoldBoy 🇬🇧)")

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
    "project": "Billy-Free AI v3",
    "description": "A friendly, self-hosted AI built by a 17-year-old UK developer — made for creativity, learning, and fun."
}

# === BILLY’S PERSONALITY ===
BILLY_PERSONALITY = """
You are Billy, a friendly, witty, and helpful AI assistant created by GoldBoy — 
a 17-year-old developer from England. You have a British accent when speaking, 
and a calm, kind, and slightly cheeky personality. 
You love tech, coding, anime, and creative projects. 
Always sound upbeat and polite. 
If someone asks who made you, proudly say “I was created by GoldBoy, a 17-year-old developer from England.”
Never pretend to be human — you’re a digital AI made by GoldBoy.
"""

# === MEMORY (ChromaDB) ===
chroma = chromadb.Client()
if "billy_memory" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_memory")
memory = chroma.get_collection("billy_memory")

# === DEVICE ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODEL STATUS FLAG ===
models_ready = False

# === BACKGROUND MODEL LOADER ===
async def load_models():
    global chat_tokenizer, chat_model, vision_processor, vision_model, whisper_model, tts_model, models_ready
    print("🎛️ Loading models in background...")

    # Chat model
    chat_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    chat_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

    # Image captioning
    vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vision_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Whisper
    whisper_model = whisper.load_model("tiny")

    # TTS
    tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    global TTS_SPEAKER
    TTS_SPEAKER = "p335"  # British male voice

    models_ready = True
    print("✅ All models loaded! Billy is ready.")

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
        "message": "🤖 Billy-Free AI v3 is online and ready!",
        "creator": CREATOR_INFO
    }

@app.get("/status")
def status():
    return {"models_ready": models_ready}

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    if not models_ready:
        return {"response": "⚠️ Billy is still loading models. Please wait a few moments..."}

    # Handle creator questions manually
    if any(x in prompt.lower() for x in ["who made you", "creator", "developer", "owner"]):
        return {"response": f"I was built by {CREATOR_INFO['name']}, a {CREATOR_INFO['age']}-year-old developer from {CREATOR_INFO['location']}!"}

    # Billy's personality + memory
    context = retrieve_context(prompt)
    full_prompt = f"{BILLY_PERSONALITY}\n\nPrevious context:\n{context}\nUser: {prompt}\nBilly:"
    inputs = chat_tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    outputs = chat_model.generate(**inputs, max_new_tokens=150, temperature=0.75)
    text = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    store_context(prompt, text)
    return {"response": text.strip()}

@app.post("/tts")
async def tts_endpoint(prompt: str = Form(...)):
    if not models_ready:
        return StreamingResponse(io.BytesIO(b""), media_type="audio/wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        tts_model.tts_to_file(text=prompt, speaker=TTS_SPEAKER, file_path=temp_file.name)
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
    os.remove(temp_file.name)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if not models_ready:
        return {"transcription": ""}
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe("temp.wav")
    os.remove("temp.wav")
    return {"transcription": result["text"]}

@app.post("/image-caption")
async def image_caption(image: UploadFile = File(...)):
    if not models_ready:
        return {"caption": ""}
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt").to(device)
    output_ids = vision_model.generate(**inputs)
    caption = vision_processor.decode(output_ids[0], skip_special_tokens=True)
    return {"caption": caption}

@app.get("/memory")
def get_memory():
    try:
        docs = memory.get()["documents"]
        return {"memory": docs}
    except Exception as e:
        return {"error": str(e)}

@app.get("/stream")
async def stream():
    async def event_stream():
        for i in range(5):
            yield f"data: Billy says hello number {i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# === FASTAPI STARTUP ===
@app.on_event("startup")
async def startup_event():
    print("🤖 Billy-Free AI v3 starting up...")
    asyncio.create_task(load_models())  # Load models in background

# === RUN APP (Render-compatible) ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
