from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from datetime import datetime
from supabase import create_client, Client
import os
import time

load_dotenv()

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Limits
MAX_MESSAGES_PER_DAY = 75
MAX_CHARACTERS = 50

# Model setup
model_id = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
model.eval()

# FastAPI app
app = FastAPI()

class MessageRequest(BaseModel):
    user_id: str
    message: str

def log_message(user_id: str, text: str):
    supabase.table("messages").insert({
        "user_id": user_id,
        "message": text,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def check_limit(user_id: str):
    today = datetime.utcnow().date().isoformat()
    res = supabase.table("messages") \
        .select("*", count='exact') \
        .eq("user_id", user_id) \
        .gte("created_at", today) \
        .execute()
    return res.count >= MAX_MESSAGES_PER_DAY

def validate_input(text: str):
    return len(text) <= MAX_CHARACTERS

@app.post("/chat")
async def chat(data: MessageRequest):
    user_id = data.user_id
    message = data.message.strip()

    if not validate_input(message):
        return JSONResponse({"error": "Message too long. Limit: 50 characters."}, status_code=400)
    if check_limit(user_id):
        return JSONResponse({"error": "Daily limit reached. Upgrade to Zynara."}, status_code=403)

    # Log message
    log_message(user_id, message)

    prompt = f"User: {message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 200,
        "temperature": 0.7,
        "do_sample": True,
        "streamer": streamer
    }

    def generate():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token

    return StreamingResponse(generate(), media_type="text/plain")
