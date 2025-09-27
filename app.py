# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from openai import OpenAI
import os

# --- Config ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # set this in Render dashboard
if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="Billy AI Chat", version="1.0.0")

# --- CORS for frontend ---
origins = ["*"]  # Change to your frontend URL for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ChatRequest(BaseModel):
    prompt: str
    chat_history: List[Tuple[str, str]] = []  # list of (user, assistant) messages

class ChatResponse(BaseModel):
    response: str

# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Billy AI chatbot running!"}

@app.get("/status")
def status():
    return {"status": "AI running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint: sends user prompt + history to OpenAI and returns response
    """
    # Build conversation for OpenAI
    messages = []
    for user_msg, ai_msg in req.chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": req.prompt})

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        text = resp.choices[0].message.content.strip()
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
