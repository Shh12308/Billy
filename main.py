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
    "description": "An open, self-hosted AI built for creativity, learning, and conversation."
}

# === BILLY’S PERSONALITY ===
BILLY_PERSONALITY = """
You are Billy, a friendly AI assistant created by GoldBoy, a 17-year-old developer from England.
You are casual, kind, and helpful — you like to chat about tech, coding, media, and creativity.
If someone asks who made you, always say it was GoldBoy.
If you don’t know something, admit it politely.
Keep responses short and conversational, like a friend texting.
Never pretend to be human — you’re a digital assistant designed by GoldBoy.
"""

# === MEMORY (ChromaDB) ===
chroma = chromadb.Client()
if "billy_memory" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_memory")
memory = chroma.get_collection("billy_memory")

# === DEVICE ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODELS ===
print("Loading models... this may take a minute.")

# 1️⃣ Chat/Text Generation — Flan-T5-Small (free + fast)
chat_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
chat_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# 2️⃣ Image Captioning — BLIP base
vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# 3️⃣ Speech Recognition — Whisper tiny
whisper_model = whisper.load_model("tiny")

# 4️⃣ Text-to-Speech — Tacotron2
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
        "message": "🤖 Billy-Free AI v2 is online and ready to chat!",
        "creator": CREATOR_INFO
    }

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    # Handle creator questions manually
    if any(x in prompt.lower() for x in ["who made you", "creator", "developer", "owner"]):
        return {"response": f"I was built by {CREATOR_INFO['name']}, a {CREATOR_INFO['age']}-year-old developer from {CREATOR_INFO['location']}!"}

    # Add Billy's personality and memory
    context = retrieve_context(prompt)
    full_prompt = f"{BILLY_PERSONALITY}\n\nPrevious:\n{context}\nUser: {prompt}\nBilly:"
    inputs = chat_tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
    outputs = chat_model.generate(**inputs, max_new_tokens=120, temperature=0.75)
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
            yield f"data: Billy says hi #{i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# === RUN APP ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Billy-Free AI v2 by {CREATOR_INFO['name']} on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
