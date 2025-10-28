import io
import os
import torch
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import whisper
from TTS.api import TTS
import chromadb

# === APP SETUP ===
app = FastAPI(title="Billy-Free AI v2")

# Allow all origins for demo / frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MEMORY (ChromaDB) ===
chroma = chromadb.Client()
if "billy_memory" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_memory")
memory = chroma.get_collection("billy_memory")

# === MODELS ===
device = "cuda" if torch.cuda.is_available() else "cpu"

chat_model = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    device=0 if device == "cuda" else -1
)

vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

whisper_model = whisper.load_model("tiny")
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# === HELPERS ===
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
    return {"message": "🤖 Billy-Free AI v2 is online and ready!"}

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    context = retrieve_context(prompt)
    full_prompt = f"{context}\n{prompt}" if context else prompt
    output = chat_model(full_prompt, max_length=180, do_sample=True, temperature=0.8)[0]["generated_text"]
    store_context(prompt, output)
    return {"response": output.strip()}

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
    return {"transcription": result["text"]}

@app.post("/image-caption")
async def image_caption(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt").to(device)
    output_ids = vision_model.generate(**inputs)
    caption = vision_processor.decode(output_ids[0], skip_special_tokens=True)
    return {"caption": caption}

@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base",
        torch_dtype=torch.float32
    )
    pipe.to(device)
    image = pipe(prompt, guidance_scale=7.5).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/stream")
async def stream():
    async def event_stream():
        for i in range(10):
            yield f"data: Stream update #{i}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/memory")
def get_memory():
    docs = memory.get()["documents"]
    return {"memory": docs}

# === RUN APP ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Use Render port, fallback to 8080 locally
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
