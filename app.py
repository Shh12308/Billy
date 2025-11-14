import os
import httpx
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your websites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

HF_API_URL = "https://router.huggingface.co/hf-inference/{model_id}"


async def hf_inference(model_id, prompt, params=None):
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="HF token missing")

    url = HF_API_URL.format(model_id=model_id)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            json={"inputs": prompt, "parameters": params or {}},
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"HF error {resp.status_code}: {resp.text}"
        )

    return resp.json()


@app.post("/chat")
async def chat(prompt: str = Form(...), user_id: str = Form("guest")):
    result = await hf_inference(
        "google/gemma-2b-it",  # CHANGE MODEL HERE IF YOU WANT
        prompt,
        params={"max_new_tokens": 200},
    )

    # HF returns {"generated_text": "..."}
    text = result[0].get("generated_text", "")

    return {"reply": text}


@app.get("/health")
def health():
    return {
        "ok": True,
        "hf_token": bool(HF_TOKEN)
    }


# IMPORTANT: DO NOT RUN UVICORN HERE – RENDER HANDLES IT.
# This must stay empty.
if __name__ == "__main__":
    pass
