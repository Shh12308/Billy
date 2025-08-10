import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading

# ----------------------------
# 1️⃣ Configuration
# ----------------------------
# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Render Dashboard

# ✅ Use a small model for cheap hosting
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Load model
print(f"🚀 Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ----------------------------
# 2️⃣ FastAPI App
# ----------------------------
app = FastAPI(title="Billy AI API")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Billy AI is running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream Billy's response back to the client."""
    prompt = f"You are Billy, a helpful AI assistant. User says: {req.message}"

    # Streaming setup
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Run generation in background thread
    thread = threading.Thread(target=model.generate, kwargs={
        "input_ids": inputs.input_ids,
        "max_new_tokens": 200,
        "temperature": 0.7,
        "streamer": streamer
    })
    thread.start()

    def token_stream():
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")

# ----------------------------
# 3️⃣ Local Debug (Gradio)
# ----------------------------
if __name__ == "__main__":
    import gradio as gr

    def respond(message):
        return "".join(chat(ChatRequest(message=message)))

    demo = gr.ChatInterface(fn=respond, title="Billy AI")
    demo.launch()
