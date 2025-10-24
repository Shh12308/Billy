# main.py
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import pipeline, AutoProcessor, BlipForConditionalGeneration
from langdetect import detect
from TTS.api import TTS
import whisper
import chromadb

# === App init ===
app = FastAPI(title="Billy-Free AI")

# === Memory DB ===
chroma = chromadb.Client()
if "billy_facts" not in [c.name for c in chroma.list_collections()]:
    chroma.create_collection("billy_facts")
memory = chroma.get_collection("billy_facts")

# === Text Generation (Small Model) ===
chat_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

# === Vision (Image Captioning) ===
vision_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === STT / Whisper ===
whisper_model = whisper.load_model("tiny")

# === TTS / Coqui ===
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# === Helpers ===
def store_context(prompt, answer):
    memory.add(
        documents=[f"{prompt} => {answer}"],
        metadatas=[{"source": "chat"}],
        ids=[f"id_{hash(prompt)}"]
    )

def retrieve_context(prompt, n=3):
    results = memory.query(query_texts=[prompt], n_results=n)
    return results["documents"][0] if results["documents"] else []

# === Routes ===

@app.get("/")
def root():
    return {"message": "✅ Billy-Free AI is online!"}

@app.post("/chat")
def chat(prompt: str = Form(...)):
    # retrieve memory
    context = retrieve_context(prompt)
    full_prompt = (context + "\n" if context else "") + prompt
    # generate response
    output = chat_model(full_prompt, max_length=200, do_sample=True)[0]['generated_text']
    # store in memory
    store_context(prompt, output)
    return {"response": output}

@app.post("/tts")
def tts(prompt: str = Form(...)):
    with io.BytesIO() as f:
        tts_model.tts_to_file(text=prompt, file_path="temp.wav")
        with open("temp.wav", "rb") as wav:
            audio_bytes = wav.read()
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

@app.post("/stt")
def stt(file: UploadFile = File(...)):
    audio_bytes = file.file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe("temp.wav")
    return {"transcription": result["text"]}

@app.post("/image-caption")
def image_caption(image: UploadFile = File(...)):
    img_bytes = image.file.read()
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = vision_processor(images=img, return_tensors="pt")
    output_ids = vision_model.generate(**inputs)
    caption = vision_processor.decode(output_ids[0], skip_special_tokens=True)
    return {"caption": caption}

@app.get("/memory")
def get_memory():
    docs = memory.get()["documents"]
    return {"memory": docs}
