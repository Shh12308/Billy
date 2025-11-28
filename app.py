from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################################
#                CONFIG
############################################################

GROQ_API_KEY = "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"

CHAT_MODEL = "llama-3.1-8b-instant"
IMAGE_MODEL = "flux"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "whisper-large-v3"

############################################################
#          FIXED — GROQ STREAMING FUNCTION
############################################################

async def groq_stream(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:

            # If Groq responds with error
            if r.status_code != 200:
                yield f"data: ERROR {r.status_code}\n\n"
                return

            async for raw_line in r.aiter_lines():
                if not raw_line:
                    continue

                # Groq sends: "data: {...}"
                if raw_line.startswith("data: "):
                    data = raw_line[len("data: "):]

                    if data.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return

                    # Send JSON chunk to FE
                    yield f"data: {data}\n\n"

                # Ignore event:, empty lines, etc.
                else:
                    continue

############################################################
#            STREAM ROUTE — WORKING
############################################################

@app.get("/stream")
async def stream_chat(prompt: str):
    return StreamingResponse(
        groq_stream(prompt),
        media_type="text/event-stream"
    )


############################################################
#                   BASIC CHAT
############################################################

@app.post("/chat")
async def basic_chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
    try:
        return r.json()
    except:
        return {"error": "Chat failed"}


############################################################
#                   IMAGE GENERATION
############################################################

@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/images/generations",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt
            }
        )

    try:
        img = r.json()["data"][0]["b64_json"]
        return {"image": img}
    except:
        return JSONResponse({"error": "Image generation failed"}, status_code=400)


############################################################
#                     TTS
############################################################

@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    text = body.get("text", "")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/audio/speech",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": TTS_MODEL,
                "voice": "alloy",
                "input": text
            }
        )

    try:
        audio = r.json()["audio"]
        return {"audio": audio}
    except:
        return JSONResponse({"error": "TTS failed"}, status_code=400)


############################################################
#                      STT
############################################################

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    b64_data = base64.b64encode(audio_bytes).decode()

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": STT_MODEL,
                "file": b64_data,
                "format": "base64"
            }
        )

    try:
        return r.json()
    except:
        return JSONResponse({"error": "STT failed"}, status_code=400)


############################################################
#                     ROOT
############################################################

@app.get("/")
async def root():
    return {"status": "Billy AI backend running OK"}


############################################################
#             LOCAL RUN
############################################################

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
