import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
FRONTEND_API_KEY = os.getenv("FRONTEND_API_KEY", "changeme")
MODEL_ID = os.getenv("MODEL_ID", "gpt2")

app = FastAPI(title="Mini Billy AI")

# Load small model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != FRONTEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/generate", dependencies=[Depends(api_key_auth)])
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

@app.get("/health")
def health():
    return {"status": "ok"}
