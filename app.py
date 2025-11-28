from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_MODEL = "llama-3.1-8b-instant"
IMAGE_MODEL = "gpt-image-1"
TTS_MODEL = "gpt-4o-mini-tts"
STT_MODEL = "whisper-large-v3"


############################################################
# STREAMING CHAT (NO API KEY)
############################################################

async def groq_stream(prompt: str):

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:

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


@app.get("/stream")
async def stream_chat(prompt: str):
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")


############################################################
# BASIC CHAT
############################################################

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    url = "https://api/groq.com/openai/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

    return r.json()


############################################################
# IMAGE GENERATION
############################################################

@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")

    url = "https://api.groq.com/openai/v1/images/generations"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt
            }
        )

    try:
        data = r.json()
        img = data["data"][0]["b64_json"]
        return {"image": img}
    except:
        return JSONResponse({"error": "Image generation failed"}, status_code=400)


############################################################
# TTS
############################################################

@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    text = body.get("text", "")

    url = "https://api.groq.com/openai/v1/audio/speech"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={
                "model": TTS_MODEL,
                "voice": "alloy",
                "input": text
            }
        )

    data = r.json()
    return {"audio": data["audio"]}


############################################################
# STT
############################################################

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={
                "model": STT_MODEL,
                "file": audio_b64,
                "format": "base64"
            }
        )

    return r.json()


@app.get("/")
async def ok():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
