import os
import httpx
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
app = FastAPI()
logger = logging.getLogger("mixtral-groq-server")
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    messages: List[Dict]
    provider: Optional[str] = "groq"
    model: Optional[str] = "mistral-saba-24b"


# ------------------------------------------------------------
# OPENROUTER MODERATION
# ------------------------------------------------------------
async def openrouter_moderate(text: str):
    url = "https://openrouter.ai/api/v1/moderations"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai-moderation-latest",
        "input": text
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


async def moderate_text(text: str):
    if not text:
        return True, None

    # quick built-in rule check
    banned = ["bomb", "kill", "terror", "rape", "shoot"]
    if any(w in text.lower() for w in banned):
        return False, "Blocked by rule-based safety"

    # OpenRouter moderation
    if OPENROUTER_API_KEY:
        try:
            res = await openrouter_moderate(text)
            categories = res.get("results", [{}])[0].get("categories", {})
            flagged = any(categories.get(cat, False) for cat in categories)
            if flagged:
                return False, "Blocked by OpenRouter moderation"
        except Exception:
            logger.exception("OpenRouter moderation error — permissive fallback")
            return True, None

    return True, None


# ------------------------------------------------------------
# GROQ CHAT COMPLETION
# ------------------------------------------------------------
async def groq_chat_completion(messages: List[Dict], model: str):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,         # REQUIRED by Groq
        "temperature": 0.2
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.error(f"Groq HTTP error ({model}): {r.text}")
        r.raise_for_status()
        return r.json()


# ------------------------------------------------------------
# MAIN CHAT HANDLER
# ------------------------------------------------------------
@app.post("/chat")
async def provider_chat(data: ChatRequest):
    text_for_safety = " ".join([m.get("content", "") for m in data.messages])
    allowed, reason = await moderate_text(text_for_safety)
    if not allowed:
        return {"error": reason}

    if data.provider == "groq":
        try:
            response = await groq_chat_completion(data.messages, data.model)
            return response
        except Exception:
            logger.exception("Groq API error")
            return {"error": "Groq provider failure"}

    # fallback: echo
    return {"output": "provider not supported"}


# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "Groq + Mistral router running"}
