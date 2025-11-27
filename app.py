from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import base64
import uvicorn

############################################################
#                   CONFIGURATION
############################################################

app = FastAPI()

# Enable CORS for your frontend hosted anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = "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"
CHAT_MODEL = "llama-3.1-8b-instant"
IMAGE_MODEL = "flux"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "whisper-large-v3"


############################################################
#                     UTIL FUNCTIONS
############################################################

async def groq_stream(prompt: str):
    """
    Connects to Groq streaming chat completions.
    Yields chunks of tokens for SSE.
    """
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
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                yield f"data: ERROR {response.status_code}\n\n"
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line.replace("data: ", "")
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {data}\n\n"


############################################################
#                   STREAMING CHAT ENDPOINT
############################################################

@app.get("/stream")
async def stream_chat(prompt: str):
    """
    SSE (Server Sent Events) endpoint for streaming chat replies.
    Your frontend uses EventSource() to connect here.
    """
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")


############################################################
#                   BASIC CHAT ENDPOINT
############################################################

@app.post("/chat")
async def basic_chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    url = "https://api.groq.com/openai/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

    if r.status_code != 200:
        return JSONResponse({"error": "Failed"}, status_code=400)

    return r.json()


############################################################
#                   IMAGE GENERATION
############################################################

@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    url = "https://api.groq.com/openai/v1/images/generations"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt
            }
        )

    try:
        data = resp.json()
        img = data["data"][0]["b64_json"]
        return {"image": img}
    except:
        return JSONResponse({"error": "Image generation failed"}, status_code=400)


############################################################
#                   TEXT TO SPEECH (TTS)
############################################################

@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    text = body.get("text", "")

    url = "https://api.groq.com/openai/v1/audio/speech"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": TTS_MODEL,
                "voice": "alloy",
                "input": text
            }
        )

    if r.status_code != 200:
        return JSONResponse({"error": "TTS failed"}, status_code=400)

    # Groq returns base64 audio
    data = r.json()
    audio_b64 = data["audio"]

    return {"audio": audio_b64}


############################################################
#                   SPEECH TO TEXT (STT)
############################################################

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Uploads an audio file and returns speech-to-text transcription.
    """
    audio_bytes = await file.read()
    b64_audio = base64.b64encode(audio_bytes).decode()

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": STT_MODEL,
                "file": b64_audio,
                "format": "base64"
            }
        )

    if r.status_code != 200:
        return JSONResponse({"error": "STT failed"}, status_code=400)

    return r.json()


############################################################
#                   HEALTH CHECK
############################################################

@app.get("/")
async def root():
    return {"status": "Billy AI backend running"}


############################################################
#                   LOCAL UVICORN RUN (OPTIONAL)
############################################################

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
