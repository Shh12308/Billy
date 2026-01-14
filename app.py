import os
import io
import json
import uuid
import numpy as np
import base64
import time
import asyncio
import logging
import subprocess
import tempfile
import cv2
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from io import BytesIO

import httpx
import aiohttp
import torch
from PIL import Image
from fastapi import BackgroundTasks, FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends, Response
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from supabase import create_client
from ultralytics import YOLO
from torchvision import models, transforms

# Fix: Import utils with proper error handling
try:
    import utils
    from utils import safe_system_prompt
except ImportError:
    # Create a placeholder if utils is not available
    def safe_system_prompt(prompt):
        return prompt
    
    class UtilsPlaceholder:
        pass
    utils = UtilsPlaceholder()

YOLO_OBJECTS = None
YOLO_FACES = None
YOLO_DEVICE = "cpu"

def get_yolo_objects():
    global YOLO_OBJECTS
    if YOLO_OBJECTS is None:
        YOLO_OBJECTS = YOLO("yolov8n.pt")
        YOLO_OBJECTS.to(YOLO_DEVICE)
    return YOLO_OBJECTS

def get_yolo_faces():
    global YOLO_FACES
    if YOLO_FACES is None:
        YOLO_FACES = YOLO("yolov8n-face.pt")
        YOLO_FACES.to(YOLO_DEVICE)
    return YOLO_FACES

# ---------- ENV KEYS ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")
    
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Initialize Supabase tables
def init_supabase_tables():
    try:
        # Create memory table
        supabase.rpc("create_memory_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create conversations table
        supabase.rpc("create_conversations_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create messages table
        supabase.rpc("create_messages_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create artifacts table
        supabase.rpc("create_artifacts_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create active_streams table
        supabase.rpc("create_active_streams_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create memories table
        supabase.rpc("create_memories_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create images table
        supabase.rpc("create_images_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create vision_history table
        supabase.rpc("create_vision_history_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create cache table
        supabase.rpc("create_cache_table").execute()
    except:
        pass  # Table might already exist
    
    try:
        # Create usage table
        supabase.rpc("create_usage_table").execute()
    except:
        pass  # Table might already exist

# Initialize tables on startup
init_supabase_tables()

groq_client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("zynara-server")

app = FastAPI(
    title="ZyNaraAI1.0 Multimodal Server",
    redirect_slashes=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SSE HELPER ----------------
def sse(obj: dict) -> str:
    """
    Formats a dict as a Server-Sent Event (SSE) message.
    """
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ---------- ENV KEYS ----------
# strip GROQ API key in case it contains whitespace/newlines
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is not None:
    GROQ_API_KEY = GROQ_API_KEY.strip()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL")
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

# Quick log so you can confirm key presence without printing the key itself
logger.info(f"GROQ key present: {bool(GROQ_API_KEY)}")

# -------------------
# Models
# -------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"  # Added missing URL

# TTS/STT are handled via ElevenLabs now
TTS_MODEL = None
STT_MODEL = None

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldYLocks",
    "age": 17,
    "country": "England",
    "projects": ["MZ", "LS", "SX", "CB"],
    "socials": { "discord":"@nexisphere123_89431", "twitter":"@NexiSphere"},
    "bio": "Created by GoldBoy (17, England). Projects: MZ, LS, SX, CB. Socials: Discord @nexisphere123_89431 Twitter @NexiSphere."
}

JUDGE0_LANGUAGES = {
    # --- C / C++ ---
    "c": 50,
    "c_clang": 49,
    "cpp": 54,
    "cpp_clang": 53,

    # --- Java ---
    "java": 62,

    # --- Python ---
    "python": 71,
    "python2": 70,
    "micropython": 79,

    # --- JavaScript / TS ---
    "javascript": 63,
    "nodejs": 63,
    "typescript": 74,

    # --- Go ---
    "go": 60,

    # --- Rust ---
    "rust": 73,

    # --- C# / .NET ---
    "csharp": 51,
    "fsharp": 87,
    "dotnet": 51,

    # --- PHP ---
    "php": 68,

    # --- Ruby ---
    "ruby": 72,

    # --- Swift ---
    "swift": 83,

    # --- Kotlin ---
    "kotlin": 78,

    # --- Scala ---
    "scala": 81,

    # --- Objective-C ---
    "objective_c": 52,

    # --- Bash / Shell ---
    "bash": 46,
    "sh": 46,

    # --- PowerShell ---
    "powershell": 88,

    # --- Perl ---
    "perl": 85,

    # --- Lua ---
    "lua": 64,

    # --- R ---
    "r": 80,

    # --- Dart ---
    "dart": 75,

    # --- Julia ---
    "julia": 84,

    # --- Haskell ---
    "haskell": 61,

    # --- Elixir ---
    "elixir": 57,

    # --- Erlang ---
    "erlang": 58,

    # --- OCaml ---
    "ocaml": 65,

    # --- Crystal ---
    "crystal": 76,

    # --- Nim ---
    "nim": 77,

    # --- Zig ---
    "zig": 86,

    # --- Assembly ---
    "assembly": 45,

    # --- COBOL ---
    "cobol": 55,

    # --- Fortran ---
    "fortran": 59,

    # --- Prolog ---
    "prolog": 69,

    # --- Scheme ---
    "scheme": 82,

    # --- Common Lisp ---
    "lisp": 66,

    # --- Brainf*ck ---
    "brainfuck": 47,

    # --- V ---
    "vlang": 91,

    # --- Groovy ---
    "groovy": 56,

    # --- Hack ---
    "hack": 67,

    # --- Pascal ---
    "pascal": 67,

    # --- Scratch ---
    "scratch": 92,

    # --- Solidity ---
    "solidity": 94,

    # --- SQL ---
    "sql": 82,

    # --- Text / Plain ---
    "plain_text": 43,
    "text": 43,
}

JUDGE0_URL = "https://judge0-ce.p.rapidapi.com"
JUDGE0_KEY = os.getenv("JUDGE0_API_KEY")

if not JUDGE0_KEY:
    logger.warning("⚠️ Judge0 key not set — code execution disabled")

if not JUDGE0_KEY:
    logger.warning("Code execution disabled (missing Judge0 API key)")

async def run_code_judge0(code: str, language_id: int):
    payload = {
        "language_id": language_id,
        "source_code": code
    }

    headers = {
        "X-RapidAPI-Key": JUDGE0_KEY,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30) as client:
        submit = await client.post(
            "https://judge0-ce.p.rapidapi.com/submissions?wait=false",
            json=payload,
            headers=headers
        )

        if submit.status_code == 403:
            return {
                "error": "Judge0 execution blocked (403). Check RapidAPI key or plan."
            }

        submit.raise_for_status()
        return submit.json()

# Fix: Moved generate_ai_response function before it's used
async def generate_ai_response(conversation_id: str, user_id: str, messages: list):
    """
    Generates an AI response using Groq API in the background.
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 1500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_URL, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                ai_message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"[Conversation {conversation_id}] AI response: {ai_message}")
                return ai_message
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, I couldn't generate a response at this time."

# Fix: Moved stream_llm function before it's used
async def stream_llm(user_id, conversation_id, messages):
    assistant_reply = ""

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 1500,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            GROQ_URL,
            headers=get_groq_headers(),
            json=payload,
        ) as response:

            async for line in response.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data = line.replace("data:", "", 1).strip()

                if not data:
                    continue

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = chunk["choices"][0]["delta"]

                # -------------------------
                # TOOL CALLS
                # -------------------------
                if "tool_calls" in delta:
                    async for item in handle_tools(user_id, messages, delta):
                        yield item
                    continue

                # -------------------------
                # NORMAL TEXT STREAMING
                # -------------------------
                content = delta.get("content")
                if content:
                    # 🚫 Prevent tool leakage
                    if "<function=" in content:
                        pass
                    else:
                        assistant_reply += content
                        yield sse({
                            "type": "token",
                            "text": content
                        })

    async def run(call):
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])

        if name == "web_search":
            return name, await duckduckgo_search(args["query"])
        if name == "run_code":
            return name, await run_code_safely(args["task"])

    results = await asyncio.gather(*(run(c) for c in calls))

    for name, result in results:
        messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })
        yield sse({"type": "tool", "tool": name, "result": result})

# Fix: Moved persist_reply function before it's used
async def persist_reply(user_id, conversation_id, text):
    try:
        supabase.table("messages").insert({
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": text,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        supabase.table("memories").insert({
            "user_id": user_id,
            "conversation_id": conversation_id,
            "content": text[:500],
            "importance": score_memory(text),
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        await decay_memories(user_id)
    except Exception as e:
        logger.error(f"Failed to persist reply: {e}")

# Fix: Moved score_memory function before it's used
def score_memory(text: str) -> int:
    if any(k in text.lower() for k in ["name", "preference", "goal"]):
        return 5
    return 2

# Fix: Moved decay_memories function before it's used
async def decay_memories(user_id):
    try:
        supabase.rpc("decay_memories", {"uid": user_id}).execute()
    except Exception as e:
        logger.error(f"Failed to decay memories: {e}")

# Fix: Moved get_groq_headers function before it's used
def get_groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

# Fix: Moved run_code_safely function before it's used
async def run_code_safely(prompt: str):
    """Helper for streaming /ask/universal."""
    # Default to python if not specified for this helper
    language = "python" 
    
    # 1. Generate code
    code_prompt = f"Write a complete {language} program to: {prompt}"
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": code_prompt}],
        "max_tokens": 2048
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json=payload
        )
        r.raise_for_status()
        code = r.json()["choices"][0]["message"]["content"]

    # 2. Run code
    lang_id = JUDGE0_LANGUAGES.get(language, 71)
    execution = await run_code_judge0(code, lang_id)
    
    return {"code": code, "execution": execution}

# Fix: Moved duckduckgo_search function before it's used
async def duckduckgo_search(q: str):
    """
    Use DuckDuckGo Instant Answer API (no API key required).
    Returns a simple structured result with abstract, answer and a list of related topics.
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html":1, "skip_disambig": 1}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        results = []
        # RelatedTopics can contain nested topics or single items; handle both.
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict):
                # Some items are like {"Text": "...", "FirstURL": "..."}
                if item.get("Text"):
                    results.append({"title": item.get("Text"), "url": item.get("FirstURL")})
                # Some are category blocks with "Topics" list
                elif item.get("Topics"):
                    for t in item.get("Topics", []):
                        if t.get("Text"):
                            results.append({"title": t.get("Text"), "url": t.get("FirstURL")})
        # Limit results to a reasonable number
        results = results[:10]

        return {
            "query": q,
            "abstract": data.get("AbstractText"),
            "answer": data.get("Answer"),
            "results": results
            }

# Fix: Moved get_or_create_user function before it's used
async def get_or_create_user(req: Request, res: Response) -> str:
    user_id = req.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        res.set_cookie(
            key="user_id",
            value=user_id,
            httponly=True,
            samesite="lax"
        )
    return user_id

# Fix: Moved cache_result function before it's used
def cache_result(prompt: str, provider: str, result: dict):
    # Store cache in Supabase
    try:
        supabase.table("cache").insert({
            "prompt": prompt,
            "provider": provider,
            "result": json.dumps(result),
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to cache result: {e}")

# Fix: Moved get_cached_result function before it's used
def get_cached_result(prompt: str, provider: str) -> Optional[dict]:
    try:
        response = supabase.table("cache").select("result").eq("prompt", prompt).eq("provider", provider).order("created_at", desc=True).limit(1).execute()
        if response.data:
            return json.loads(response.data[0]["result"])
    except Exception as e:
        logger.error(f"Failed to get cached result: {e}")
    return None

# Fix: Moved get_system_prompt function before it's used
def get_system_prompt(user_message: Optional[str] = None) -> str:
    base = "You are ZynaraAI1.0: helpful, concise, friendly, and focus entirely on what the user asks. Do not reference your creator or yourself unless explicitly asked."
    if user_message:
        base += f" The user said: \"{user_message}\". Tailor your response to this."
    return base

# Fix: Moved build_contextual_prompt function before it's used
def build_contextual_prompt(user_id: str, message: str) -> str:
    try:
        # Get user memory
        memory_response = supabase.table("memory").select("key, value").eq("user_id", user_id).order("updated_at", desc=True).limit(5).execute()
        memory_rows = memory_response.data if memory_response.data else []
        
        # Get conversation history for context
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        if conv_response.data:
            conv_id = conv_response.data[0]["id"]
            msg_response = supabase.table("messages").select("content").eq("conversation_id", conv_id).order("created_at", desc=True).limit(10).execute()
            msg_rows = msg_response.data if msg_response.data else []
        else:
            msg_rows = []
        
        context = "\n".join(f"{row['key']}: {row['value']}" for row in memory_rows)
        msg_context = "\n".join(f"Previous: {row['content']}" for row in msg_rows)
        
        return f"""You are ZyNaraAI1.0: helpful, concise, friendly. Focus on exactly what the user wants.
User context:
{context}

Recent conversation:
{msg_context}

User message: {message}"""
    except Exception as e:
        logger.error(f"Failed to build contextual prompt: {e}")
        return f"You are ZyNaraAI1.0: helpful, concise, friendly. Focus on exactly what the user wants.\n\nUser message: {message}"

# Fix: Moved check_permission function before it's used
def check_permission(role: str, tool_name: str) -> bool:
    permissions = {
        "user": {"web_search"},
        "admin": {"web_search", "run_code"},
        "system": {"web_search", "run_code"}
    }
    return tool_name in permissions.get(role, set())

# Fix: Moved get_or_create_conversation_id function before it's used
def get_or_create_conversation_id(supabase, user_id: str) -> str:
    # Try to get most recent conversation
    res = (
        supabase.table("conversations")
        .select("id")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(1)
        .execute()
    )

    if res.data:
        return res.data[0]["id"]

    # Create new conversation
    conversation_id = str(uuid.uuid4())
    supabase.table("conversations").insert({
        "id": conversation_id,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }).execute()

    return conversation_id

# Fix: Moved build_system_prompt function before it's used
def build_system_prompt(artifact: Union[str, None]):
    base = """
You are a helpful AI assistant.

You are in an ongoing conversation.
You MUST maintain continuity.
You MUST respect prior context.

Rules:
- Do not reset unless asked
- If user says "it", "that", "the last thing", infer correctly
- If an artifact exists, modify it instead of starting over
"""
    if artifact:
        base += f"""

Current working artifact:
-------------------------
{artifact}
-------------------------
You are editing this artifact.
Return the FULL updated version.
"""
    return base

# Fix: Moved unique_filename function before it's used
def unique_filename(ext="png"):
    return f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"

# Fix: Moved upload_to_supabase function before it's used
def upload_to_supabase(
    file_bytes: bytes,
    filename: str,
    bucket: str = "ai-images",
    content_type: str = "application/octet-stream"
) -> str:
    """
    Upload a file (image or video) to Supabase storage.
    """
    supabase.storage.from_(bucket).upload(
        path=filename,
        file=file_bytes,
        file_options={"content-type": content_type}
    )
    return filename

# Fix: Moved upload_image_to_supabase function before it's used
def upload_image_to_supabase(image_bytes: bytes, filename: str):
    upload = supabase.storage.from_("ai-images").upload(
        filename,
        image_bytes,
        {"content-type": "image/png"}
    )

    if upload.get("error"):
        raise Exception(upload["error"]["message"])

    return upload

# Fix: Moved save_image_record function before it's used
def save_image_record(user_id, prompt, path, is_nsfw):
    try:
        supabase.table("images").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "prompt": prompt,
            "image_path": path,
            "is_nsfw": is_nsfw,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save image record: {e}")

# Fix: Moved route_query function before it's used
async def route_query(user_id: str, query: str):
    q = query.lower()

    PERSONAL = ["who am i", "what did i say", "my name", "about me"]
    if any(k in q for k in PERSONAL):
        return "memory"

    try:
        memories = supabase.rpc("search_memories", {
            "uid": user_id,
            "q": query,
            "limit": 3
        }).execute().data

        if memories and memories[0]["score"] > 0.75:
            return "memory"
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")

    return "search"

# Fix: Moved get_user_id_from_cookie function before it's used
async def get_user_id_from_cookie(request: Request, response: Response) -> str:
    return await get_or_create_user(request, response)

# Fix: Moved nsfw_check function before it's used
async def nsfw_check(prompt: str) -> bool:
    if not OPENAI_API_KEY: return False
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.openai.com/v1/moderations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "omni-moderation-latest", "input": prompt}
        )
        r.raise_for_status()
        result = r.json()["results"][0]
        return result["flagged"]

# Fix: Moved get_or_create_conversation function before it's used
async def get_or_create_conversation(user_id: str, conversation_id: Union[str, None]):
    if conversation_id:
        return conversation_id

    conv_id = str(uuid.uuid4())
    try:
        supabase.table("conversations").insert({
            "id": conv_id,
            "user_id": user_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
    
    return conv_id

# Fix: Moved load_artifact function before it's used
async def load_artifact(conversation_id: str):
    try:
        response = supabase.table("artifacts").select("*").eq("conversation_id", conversation_id).limit(1).execute()
        if response.data:
            row = response.data[0]
            return {
                "id": row["id"],
                "conversation_id": row["conversation_id"],
                "type": row["type"],
                "content": row["content"]
            }
    except Exception as e:
        logger.error(f"Failed to load artifact: {e}")
    return None

# Fix: Moved summarize_conversation function before it's used
async def summarize_conversation(conversation_id: str):
    try:
        response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(40).execute()
        rows = response.data if response.data else []
        
        if not rows:
            return

        text = "\n".join(f"{row['role']}: {row['content']}" for row in rows)

        payload = {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Summarize this conversation briefly. "
                        "Capture important facts, preferences, and ongoing work."
                    )
                },
                {"role": "user", "content": text}
            ],
            "max_tokens": 200
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            summary = r.json()["choices"][0]["message"]["content"]

        supabase.table("conversations").update({
            "summary": summary,
            "updated_at": datetime.now().isoformat()
        }).eq("id", conversation_id).execute()
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")

# Fix: Moved get_user_from_request function before it's used
def get_user_from_request(request: Request) -> dict:
    auth = request.headers.get("authorization")

    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")

    token = auth.split(" ")[1]

    res = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY
        }
    )

    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")

    return res.json()

# Fix: Moved save_artifact function before it's used
async def save_artifact(conversation_id: str, type_: str, content: str):
    try:
        existing = await load_artifact(conversation_id)

        if existing:
            supabase.table("artifacts").update({
                "content": content,
                "type": type_
            }).eq("id", existing["id"]).execute()
        else:
            supabase.table("artifacts").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "type": type_,
                "content": content,
                "created_at": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        logger.error(f"Failed to save artifact: {e}")

# Fix: Moved detect_artifact function before it's used
def detect_artifact(text: str):
    t = text.lower()

    if "<html" in t:
        return "html"
    if "```css" in t:
        return "css"
    if "```js" in t or "```javascript" in t:
        return "javascript"
    if "```python" in t:
        return "python"
    if "image:" in t or "draw" in t:
        return "image"
    if len(text) > 500:
        return "document"

    return None

# Fix: Moved load_history function before it's used
async def load_history(user_id: str, limit: int = 20):
    try:
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        if conv_response.data:
            conversation_id = conv_response.data[0]["id"]
            msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(limit).execute()
            rows = msg_response.data if msg_response.data else []
            return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
    return []

# Fix: Moved load_memory function before it's used
def load_memory(conversation_id: str, limit: int = 20):
    try:
        response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(limit).execute()
        rows = response.data if response.data else []
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")
    return []

# Fix: Moved extract_memory_from_prompt function before it's used
def extract_memory_from_prompt(prompt: str):
    p = prompt.lower()

    if "my name is" in p:
        name = prompt.split("my name is", 1)[1].strip().split()[0]
        return ("name", name)

    if "i live in" in p:
        location = prompt.split("i live in", 1)[1].strip()
        return ("location", location)

    if "i like" in p:
        pref = prompt.split("i like", 1)[1].strip()
        return ("preference", pref)

    return None

# Fix: Moved load_user_memory function before it's used
def load_user_memory(user_id: str):
    try:
        response = supabase.table("memories").select("key, value").eq("user_id", user_id).execute()
        rows = response.data if response.data else []
        return [{"key": row["key"], "value": row["value"]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load user memory: {e}")
    return []

# Fix: Moved universal_chat_stream function before it's used
async def universal_chat_stream(user_id: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = get_groq_headers()

    payload = {
        "model": CHAT_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
            {"role": "user", "content": prompt}
           ],
            "max_tokens": 1024
        
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield {"type": "chat_token", "text": delta}
                except Exception:
                    continue

# Fix: Moved analyze_prompt function before it's used
def analyze_prompt(prompt: str):
    p = prompt.lower()
    settings = {"model": "stable-diffusion-xl-v1","width":1024,"height":1024,"steps":30,"cfg_scale":7,"samples":1,"negative_prompt":"nsfw, nudity, watermark, lowres, text, logo"}
    if any(w in p for w in ("wallpaper","background","poster")):
        settings["width"], settings["height"] = 1920, 1080
    if any(w in p for w in ("landscape","city","wide","panorama")):
        settings["width"], settings["height"] = 1280, 720
    for token in p.split():
        if token.isnumeric():
            n = int(token)
            if 1 <= n <= 6:
                settings["samples"] = n
    return settings

# Fix: Moved image_stream_helper function before it's used
async def image_stream_helper(prompt: str, samples: int):
    try:
        result = await _generate_image_core(prompt, samples, "anonymous", return_base64=False)
        yield {
            "type": "images",
            "provider": result["provider"],
            "images": result["images"]
        }
    except HTTPException as e:
        yield {
            "type": "image_error",
            "error": e.detail
        }
    except Exception as e:
        logger.exception("Unhandled image stream error")
        yield {
            "type": "image_error",
            "error": "Unexpected image error"
        }

# Fix: Moved chat_stream_helper function before it's used
async def chat_stream_helper(user_id: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
    "model": CHAT_MODEL,
    "stream": True,
    "messages": [
        {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1024
}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            url,
            headers=get_groq_headers(),
            json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield {"type": "token", "text": delta}
                except Exception:
                    continue

# Fix: Moved enhance_prompt_with_groq function before it's used
async def enhance_prompt_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return prompt
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = get_groq_headers()
        system = "Rewrite the user's short prompt into a detailed, professional SDXL-style art prompt. Be concise but specific. Avoid explicit sexual or illegal content."
        body = {"model": CHAT_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}], "temperature":0.6,"max_tokens":300}
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code != 200:
                logger.warning("Groq enhance_prompt_with_groq failed: status=%s text=%s", r.status_code, (r.text[:100] + '...') if r.text else "")
                r.raise_for_status()
            jr = r.json()
            content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or prompt
    except Exception:
        logger.exception("Prompt enhancer failed")
    return prompt

# Fix: Moved generate_video_internal function before it's used
async def generate_video_internal(prompt: str, samples: int = 1, user_id: str = "anonymous") -> dict:
    """
    Generate video (stub). Stores video files in Supabase storage like images.
    Returns a list of signed URLs.
    """
    urls = []

    for i in range(samples):
        # For now, we create a placeholder video file
        placeholder_content = b"This is a placeholder video for prompt: " + prompt.encode('utf-8')
        filename = f"{user_id}/video-{int(time.time())}-{uuid.uuid4().hex[:8]}.mp4"

        # Upload to Supabase
        try:
            supabase.storage.from_("ai-videos").upload(
                path=filename,
                file=placeholder_content,
                file_options={"content-type": "video/mp4"}
            )
            # Get signed URL
            signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
            urls.append(signed["signedURL"])
        except Exception as e:
            logger.error(f"Video upload failed: {e}")

    return {"provider": "stub", "videos": urls}

# Fix: Moved detect_intent function before it's used
def detect_intent(prompt: str) -> str:
    if not prompt:
        return "chat"

    p = prompt.lower()

    # 🖼 Image generation
    if any(w in p for w in [
        "image of", "draw", "picture of", "generate image",
        "make me an image", "photo of", "art of"
    ]):
        return "image"

    # 🖼 Image → Image
    if any(w in p for w in [
        "edit this image", "change this image",
        "modify image", "img2img"
    ]):
        return "img2img"

    # 👁 Vision / analysis
    if any(w in p for w in [
        "analyze this image", "what is in this image",
        "describe this image", "vision"
    ]):
        return "vision"

    # 🎙 Speech → Text
    if any(w in p for w in [
        "transcribe", "speech to text", "stt"
    ]):
        return "stt"

    # 🔊 Text → Speech
    if any(w in p for w in [
        "say this", "speak", "tts", "read this", "read aloud"
    ]):
        return "tts"

    # 🎥 Video (future-ready)
    if any(w in p for w in [
        "video of", "make a video", "animation of", "clip of"
    ]):
        return "video"

    # 💻 Code
    if any(w in p for w in [
        "write code", "generate code", "python code",
        "javascript code", "fix this code"
    ]):
        return "code"

    # 🔍 Search
    if any(w in p for w in [
        "search", "look up", "find info", "who is", "what is"
    ]):
        return "search"

    return "chat"

# Fix: Moved tts_stream_helper function before it's used
async def tts_stream_helper(text: str):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "tts-1", 
        "voice": "alloy",
        "input": text
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload
        )
        r.raise_for_status()

    b64 = base64.b64encode(r.content).decode()
    yield {"type": "tts_done", "audio": b64}

# Fix: Moved _generate_image_core function before it's used
async def _generate_image_core(
    prompt: str,
    samples: int,
    user_id: str,
    return_base64: bool = False
):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    provider_used = "openai"
    urls = []

    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,  # DALL·E 3 supports only 1 image
        "size": "1024x1024",
        "response_format": "b64_json"
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # ---------- CALL OPENAI ----------
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/images/generations",
                json=payload,
                headers=headers
            )
            r.raise_for_status()
            result = r.json()

    except Exception:
        logger.exception("OpenAI image API call failed")
        raise HTTPException(500, "Image generation provider error")

    # ---------- VALIDATE ----------
    if not result or not result.get("data"):
        logger.error("OpenAI returned empty image response: %s", result)
        raise HTTPException(500, "Image generation failed")

    # ---------- PROCESS IMAGES ----------
    for img in result["data"]:
        try:
            b64 = img.get("b64_json")
            if not b64:
                continue

            image_bytes = base64.b64decode(b64)
            filename = f"{user_id}/{uuid.uuid4().hex}.png"

            upload = supabase.storage.from_("ai-images").upload(
                path=filename,
                file=image_bytes,
                file_options={
                    "content-type": "image/png",
                }
            )

            if isinstance(upload, dict) and upload.get("error"):
                raise RuntimeError(upload["error"])

            signed = supabase.storage.from_("ai-images").create_signed_url(
                filename, 60 * 60
            )

            if signed and signed.get("signedURL"):
                urls.append(signed["signedURL"])

        except Exception:
            logger.exception("Failed processing or uploading image")
            continue

    if not urls:
        raise HTTPException(500, "No images generated")

    cache_result(prompt, provider_used, {"images": urls})

    return {
        "provider": provider_used,
        "images": urls
    }

# Fix: Moved image_gen_internal function before it's used
async def image_gen_internal(prompt: str, samples: int = 1):
    """Helper for streaming /ask/universal."""
    result = await _generate_image_core(prompt, samples, "anonymous", return_base64=False)

# Fix: Moved stream_images function before it's used
async def stream_images(prompt: str, samples: int):
    try:
        async for chunk in image_stream_helper(prompt, samples):
            yield sse(chunk)
    except HTTPException as e:
        yield sse({"type": "image_error", "error": e.detail})

# Fix: Moved run_agents function before it's used
async def run_agents(prompt: str):
    async def research():
        return await duckduckgo_search(prompt)

    async def coding():
        return await run_code_safely(prompt)

    results = await asyncio.gather(
        research(),
        coding(),
        return_exceptions=True
    )
    return results

# Fix: Moved track_cost function before it's used
def track_cost(user_id: str, tokens: int, tool: Union[str, None] = None):
    try:
        supabase.table("usage").insert({
            "user_id": user_id,
            "tokens": tokens,
            "tool": tool,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Cost tracking failed: {e}")

# Fix: Moved auth function before it's used
async def auth(request: Request):
    # Simple auth function that extracts user_id from cookie
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return type('User', (), {'id': user_id})()

# Fix: Moved PERSONALITY_MAP before it's used
PERSONALITY_MAP = {
    "friendly": (
        "You are friendly, warm, and encouraging. "
        "Explain things clearly and be approachable."
    ),
    "professional": (
        "You are concise, formal, and professional. "
        "Give structured, direct answers."
    ),
    "playful": (
        "You are playful, witty, and creative, but still helpful."
    )
}

# Fix: Moved TOOLS before it's used
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Generate and execute code safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"}
                },
                "required": ["task"]
            }
        }
    }
]

# Tracks currently active SSE/streaming tasks per user
active_streams: Dict[str, asyncio.Task] = {}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }
    
@app.get("/")
async def root():
    return {"message": "Billy AI Backend is Running ✔"}
    
@app.post("/chat/stream")
async def chat_stream(req: Request, res: Response, tts: bool = False, samples: int = 1):
    """
    Unified streaming endpoint:
    - Image streaming (if prompt implies image)
    - Chat streaming (Groq)
    - Optional TTS
    Cookie-based identity + safe cancellation
    """
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "prompt required")

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)
    
# ---------- Chat endpoint ----------
@app.post("/chat")
async def chat_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt","")
    user_id = body.get("user_id", "anonymous")
    if not prompt:
        raise HTTPException(400,"prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400,"no groq key")
    payload = {"model":CHAT_MODEL,"messages":[{"role":"system","content":build_contextual_prompt(user_id, prompt)},{"role":"user","content":prompt}]}

    headers = get_groq_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if r.status_code != 200:
                logger.warning("Groq /chat returned status=%s text=%s", r.status_code, (r.text[:500] + '...') if r.text else "")
                r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            logger.exception("Groq HTTP error on /chat: %s", getattr(exc.response, "text", "no-response-text"))
            raise HTTPException(status_code=exc.response.status_code if exc.response is not None else 500, detail=f"Groq error: {exc.response.text[:400] if exc.response is not None else str(exc)}")
        except Exception:
            logger.exception("Groq /chat request failed")
            raise HTTPException(500, "groq_request_failed")

# =========================================================
# 🚀 UNIVERSAL MULTIMODAL ENDPOINT — /ask/universal
# =========================================================

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

@app.post("/ask/universal")
async def ask_universal(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    prompt = body.get("prompt", "").strip()
    user_id = body.get("user_id") or str(uuid.uuid4())
    role = body.get("role", "user")
    stream = bool(body.get("stream", False))

    if not prompt:
        raise HTTPException(400, "prompt required")

    conversation_id = get_or_create_conversation_id(
        supabase=supabase,
        user_id=user_id
    )

    history = await load_history(user_id)

    personality = "friendly"
    nickname = ""

    try:
        profile = (
            supabase.table("profiles")
            .select("nickname, personality")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if profile.data:
            personality = profile.data.get("personality", "friendly")
            nickname = profile.data.get("nickname", "")
    except Exception:
        pass

    system_prompt = (
        PERSONALITY_MAP.get(personality, PERSONALITY_MAP["friendly"])
        + f"\nUser nickname: {nickname}\n"
        "You are a ChatGPT-style multimodal assistant.\n"
        "You can call tools when useful.\n"
        "Maintain memory and context.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # ======================================================
    # STREAM MODE
    # ======================================================
    if stream:

       async def event_generator():
        assistant_reply = ""

        yield sse({"type": "starting"})
        yield ": heartbeat\n\n"

        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 1500
        }

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:

                    async for line in resp.aiter_lines():
                        if not line:
                            continue

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if data == "[DONE]":
                            break

                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]

                        # TOOL CALLS
                        if "tool_calls" in delta:
                            for call in delta["tool_calls"]:
                                name = call["function"]["name"]
                                args = json.loads(call["function"]["arguments"])

                                if not check_permission(role, name):
                                    yield sse({"type": "error", "error": "Permission denied"})
                                    continue

                                if name == "web_search":
                                    result = await duckduckgo_search(args["query"])
                                elif name == "run_code":
                                    result = await run_code_safely(args["task"])
                                else:
                                    continue

                                yield sse({
                                    "type": "tool",
                                    "tool": name,
                                    "result": result
                                })

                                messages.append({
                                    "role": "tool",
                                    "tool_name": name,
                                    "content": json.dumps(result)
                                })

                        # TEXT TOKENS
                        content = delta.get("content")
                        if content:
                            assistant_reply += content
                            yield sse({"type": "token", "text": content})

        except asyncio.CancelledError:
            logger.info("Stream cancelled by user")
            yield sse({"type": "cancelled"})

        finally:
            if assistant_reply.strip():
                supabase.table("messages").insert({
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": assistant_reply
                }).execute()

            yield sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

    # ======================================================
    # NON-STREAM MODE
    # ======================================================
    background_tasks.add_task(
        generate_ai_response,
        conversation_id,
        user_id,
        messages
    )

    return {
        "status": "processing",
        "conversation_id": conversation_id
    }
    
    
@app.post("/message/{message_id}/edit")
async def edit_message(
    message_id: str,
    req: Request,
    res: Response
):
    user_id = await get_or_create_user(req, res)
    body = await req.json()
    new_text = body.get("content")

    if not new_text:
        raise HTTPException(400, "content required")

    # Get message
    try:
        msg_response = supabase.table("messages").select("id, role, conversation_id, created_at").eq("id", message_id).execute()
        if not msg_response.data:
            raise HTTPException(404, "message not found")
        
        msg_row = msg_response.data[0]
        if msg_row["role"] != "user":
            raise HTTPException(403, "only user messages can be edited")

        conversation_id = msg_row["conversation_id"]
        edited_at = msg_row["created_at"]

        # Update message content
        supabase.table("messages").update({
            "content": new_text
        }).eq("id", message_id).execute()

        # 🔥 DELETE ALL ASSISTANT MESSAGES AFTER THIS MESSAGE
        supabase.table("messages").delete().eq("conversation_id", conversation_id).gt("created_at", edited_at).eq("role", "assistant").execute()

        return {
            "status": "edited",
            "conversation_id": conversation_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to edit message: {e}")
        raise HTTPException(500, "Failed to edit message")

@app.get("/stream")
async def stream_endpoint():
    async def event_generator():
        for i in range(1, 6):
            # Check if client disconnected
            yield sse({"message": f"This is chunk {i}"})
            await asyncio.sleep(1)
        yield sse({"message": "Done"})
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -----------------------------
# Stop endpoint
# -----------------------------
@app.post("/stop")
async def stop(user=Depends(auth)):
    task = active_streams.get(user.id)
    if task:
        task.cancel()
        del active_streams[user.id]
        return {"stopped": True}
    return {"stopped": False}
    
# -----------------------------
# Regenerate endpoint
# -----------------------------
@app.post("/regenerate")
async def regenerate(req: Request, res: Response, tts: bool = False, samples: int = 1):
    """
    Cancel current stream (if any) and re-run the prompt as a fresh stream.
    Cookie-based, streaming-safe.
    """
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(400, "prompt required")

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)

    # ✅ CANCEL EXISTING STREAM (IF ANY)
    old_task = active_streams.get(user_id)
    if old_task and not old_task.done():
        old_task.cancel()

    async def event_generator():
        # ✅ REGISTER NEW STREAM
        task = asyncio.current_task()
        active_streams[user_id] = task

        # Also register in database
        stream_id = str(uuid.uuid4())
        try:
            supabase.table("active_streams").insert({
                "user_id": user_id,
                "stream_id": stream_id,
                "started_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to register stream: {e}")

        try:
            # --- IMAGE (OPTIONAL) ---
            if any(w in prompt.lower() for w in ("image", "draw", "illustrate", "painting", "art", "picture")):
                try:
                    yield sse({"status": "image_start", "message": "Regenerating image"})

                    img_payload = {
                        "prompt": prompt,
                        "samples": samples,
                        "base64": False
                    }

                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "http://127.0.0.1:8000/image/stream",
                            json=img_payload
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if line.strip():
                                    yield line + "\n\n"

                    yield sse({"status": "image_done"})

                except Exception:
                    logger.exception("Image regenerate failed")
                    yield sse({"status": "image_error"})

            # --- CHAT ---
            payload = {
                "model": CHAT_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": safe_system_prompt(
    build_contextual_prompt(user_id, prompt)
)},
                    {"role": "user", "content": prompt}
                ]
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break

                        yield sse({
                            "status": "chat_progress",
                            "message": data
                        })

            # --- TTS (OPTIONAL) ---
            if tts:
                try:
                    tts_payload = {
                        "model": "tts-1",
                        "voice": "alloy",
                        "input": prompt
                    }

                    headers = {
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    }

                    audio_buffer = bytearray()

                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream(
                            "POST",
                            "https://api.openai.com/v1/audio/speech",
                            headers=headers,
                            json=tts_payload
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                if chunk:
                                    audio_buffer.extend(chunk)

                    yield sse({
                        "status": "tts_done",
                        "audio": base64.b64encode(audio_buffer).decode()
                    })

                except Exception:
                    logger.exception("TTS regenerate failed")
                    yield sse({"status": "tts_error"})

            yield sse({"status": "done"})

        except asyncio.CancelledError:
            logger.info(f"Regenerate cancelled for user {user_id}")
            yield sse({"status": "stopped"})
            raise

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)
            try:
                supabase.table("active_streams").delete().eq("user_id", user_id).execute()
            except Exception as e:
                logger.error(f"Failed to cleanup active stream: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

   
@app.post("/video")
async def generate_video(request: Request):
    """
    Generate a video from a prompt using Hugging Face and upload to Supabase.
    Returns signed URL(s) to the video(s).
    """
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    user_id = body.get("user_id", str(uuid.uuid4()))  # Generate a UUID if not provided
    samples = max(1, int(body.get("samples", 1)))

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not HF_API_KEY:
        raise HTTPException(500, "HF_API_KEY missing")

    video_urls = []

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            for _ in range(samples):
                payload = {"inputs": prompt}

                r = await client.post(
                    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-videos",
                    headers=headers,
                    json=payload
                )
                r.raise_for_status()
                # HF returns bytes, not base64
                video_bytes = r.content

                # Generate a unique filename
                filename = f"{user_id}/{int(time.time())}-video-{uuid.uuid4().hex[:10]}.mp4"

                # Upload to Supabase
                supabase.storage.from_("ai-videos").upload(
                    path=filename,
                    file=video_bytes,
                    file_options={"content-type": "video/mp4"}
                )

                # Create signed URL (1 hour)
                signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
                video_urls.append(signed["signedURL"])

    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {str(e)}")

    if not video_urls:
        raise HTTPException(500, "No video generated")

    return {"provider": "huggingface", "videos": video_urls}
    
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    user_id = body.get("user_id", str(uuid.uuid4()))  # Generate a UUID if not provided

    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1

    return_base64 = bool(body.get("base64", False))

    if not prompt:
        raise HTTPException(400, "prompt required")

    return await _generate_image_core(prompt, samples, user_id, return_base64)

@app.get("/test-stream")
async def test_stream(request: Request):
    async def event_generator():
        for i in range(1, 6):
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield sse({"message": f"This is chunk {i}"})
            await asyncio.sleep(1)
        yield sse({"message": "Done"})
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
@app.post("/image/stream")
async def image_stream(req: Request, res: Response):
    """
    Stream progress to the client while generating images.
    Uses cookies for user identity and supports safe cancellation.
    """
    body = await req.json()
    prompt = body.get("prompt", "")

    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1

    return_base64 = bool(body.get("base64", False))

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not OPENAI_API_KEY:
        raise HTTPException(400, "no OpenAI KEY")

    # ✅ COOKIE-BASED USER ID
    user_id = await get_or_create_user(req, res)

    async def event_generator():
        # ✅ REGISTER STREAM TASK (MUST BE HERE)
        task = asyncio.current_task()
        active_streams[user_id] = task

        # Also register in database
        stream_id = str(uuid.uuid4())
        try:
            supabase.table("active_streams").insert({
                "user_id": user_id,
                "stream_id": stream_id,
                "started_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to register stream: {e}")

        try:
            # --- initial message ---
            yield sse({"status": "starting", "message": "Preparing request"})
            await asyncio.sleep(0)

            payload = {
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,  # DALL·E 3 supports only 1
                "size": "1024x1024",
                "response_format": "b64_json"
            }

            yield sse({"status": "request", "message": "Sending to OpenAI"})

            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )

            if r.status_code != 200:
                text_snip = (await r.aread()).decode(errors="ignore")[:1000]
                yield sse({
                    "status": "error",
                    "message": "OpenAI error",
                    "detail": text_snip
                })
                return

            jr = r.json()
            urls = []

            data_list = jr.get("data", [])
            if not data_list:
                yield sse({"status": "warning", "message": "No data returned from provider"})

            for i, d in enumerate(data_list, start=1):
                yield sse({
                    "status": "progress",
                    "message": f"Processing {i}/{samples}"
                })
                await asyncio.sleep(0)

                b64 = d.get("b64_json")
                if not b64:
                    continue

                try:
                    image_bytes = base64.b64decode(b64)
                    filename = f"streaming/{unique_filename('png')}"
                    upload_image_to_supabase(image_bytes, filename)

                    signed = supabase.storage.from_("ai-images").create_signed_url(
                        filename, 60 * 60
                    )
                    urls.append(signed["signedURL"])

                except Exception as e:
                    logger.exception("Supabase upload failed in stream")
                    yield sse({
                        "status": "error",
                        "message": f"Storage failed: {str(e)}"
                    })

            yield sse({"status": "done", "images": urls})

        except asyncio.CancelledError:
            logger.info(f"Image stream cancelled for user {user_id}")
            yield sse({"status": "stopped"})
            raise

        except Exception as e:
            logger.exception("image_stream exception")
            yield sse({"status": "exception", "message": str(e)})

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)
            try:
                supabase.table("active_streams").delete().eq("user_id", user_id).execute()
            except Exception as e:
                logger.error(f"Failed to cleanup active stream: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
# ---------- Img2Img (DALL·E edits) ----------
@app.post("/img2img")
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = "", user_id: str = None):
    if user_id is None:
        user_id = str(uuid.uuid4())
        
    if not prompt:
        raise HTTPException(400, "prompt required")
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    if not OPENAI_API_KEY:
        raise HTTPException(400, "no OpenAI API key configured")

    urls = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"image": (file.filename, content)}
            data = {"prompt": prompt, "n": 1, "size": "1024x1024"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            r = await client.post("https://api.openai.com/v1/images/edits", headers=headers, files=files, data=data)
            r.raise_for_status()
            jr = r.json()
            for d in jr.get("data", []):
                b64 = d.get("b64_json")
                if b64:
                    fname = unique_filename("png")
                    # Note: Edit API still returns B64 or URL. If B64, we MUST upload to Supabase.
                    # If we use local_image_url for edits, we will hit the same 404 error on Railway.
                    # Let's upload to Supabase here too for consistency.
                    
                    image_bytes = base64.b64decode(b64)
                    supabase_fname = f"{user_id}/edits/{fname}"
                    upload_image_to_supabase(image_bytes, supabase_fname)
                    signed = supabase.storage.from_("ai-images").create_signed_url(supabase_fname, 60*60)
                    urls.append(signed["signedURL"])
    except Exception:
        logger.exception("img2img DALL-E edit failed")
        raise HTTPException(400, "img2img failed")

    return {"provider": "dalle3-edit", "images": urls}
    
# ---------- TTS ----------
@app.post("/tts")
async def text_to_speech(request: Request):
    """
    Convert text to speech using OpenAI TTS (tts-1).
    Accepts either JSON: {"text": "..."} or raw text/plain in the body.
    Returns audio/mpeg directly.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    # Try JSON first
    try:
        data = await request.json()
        text = data.get("text", None)
    except Exception:
        # Fallback: read raw text from body
        text = (await request.body()).decode("utf-8")

    if not text or not text.strip():
        raise HTTPException(400, "Missing 'text' in request")

    payload = {
        "model": "tts-1", 
        "voice": "alloy",  # default voice
        "input": text.strip(),
        "format": "mp3"
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload
            )
            r.raise_for_status()

            return Response(
                content=r.content,
                media_type="audio/mpeg"
            )

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {"error": f"OpenAI HTTP error: {e.response.status_code}", "detail": e.response.text},
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            {"error": "TTS request failed", "detail": str(e)},
            status_code=500
        )
        
@app.post("/tts/stream")
async def tts_stream(req: Request, res: Response):
    data = await req.json()
    text = data.get("text", "")

    if not text:
        raise HTTPException(400, "text required")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "tts-1",
        "voice": "alloy",
        "input": text
    }

    # ✅ COOKIE USER
    user_id = await get_or_create_user(req, res)

    async def audio_streamer():
        # ✅ REGISTER STREAM TASK
        task = asyncio.current_task()
        active_streams[user_id] = task

        # Also register in database
        stream_id = str(uuid.uuid4())
        try:
            supabase.table("active_streams").insert({
                "user_id": user_id,
                "stream_id": stream_id,
                "started_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to register stream: {e}")

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/audio/speech",
                    json=payload,
                    headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk

        except asyncio.CancelledError:
            logger.info(f"TTS stream cancelled for user {user_id}")
            raise

        except Exception as e:
            logger.exception("TTS streaming failed")
            # Audio streams cannot emit JSON errors safely mid-stream

        finally:
            # ✅ CLEANUP
            active_streams.pop(user_id, None)
            try:
                supabase.table("active_streams").delete().eq("user_id", user_id).execute()
            except Exception as e:
                logger.error(f"Failed to cleanup active stream: {e}")

    return StreamingResponse(
        audio_streamer(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ---------- Vision analyze ----------
@app.post("/vision/analyze")
async def vision_analyze(
    req: Request,
    res: Response,
    file: UploadFile = File(...)
):
    user_id = await get_or_create_user(req, res)
    content = await file.read()

    if not content:
        raise HTTPException(400, "empty file")

    # Load image
    img = Image.open(BytesIO(content)).convert("RGB")
    np_img = np.array(img)
    annotated = np_img.copy()

    # =========================
    # 1️⃣ YOLO OBJECT DETECTION
    # =========================
    obj_results = get_yolo_objects()(np_img, conf=0.25)
    detections = []

    for r in obj_results:
        for box in r.boxes:
            label = YOLO_OBJECTS.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    # =========================
    # 2️⃣ FACE DETECTION
    # =========================
    face_results = get_yolo_faces()(np_img)
    face_count = 0

    for r in face_results:
        for box in r.boxes:
            face_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(
                annotated,
                "face",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,0),
                2
            )

    # =========================
    # 3️⃣ DOMINANT COLORS
    # =========================
    hex_colors = []
    try:
        from sklearn.cluster import KMeans  # Added import inside function to avoid import error if sklearn is not installed
        pixels = np_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
        hex_colors = [
            '#%02x%02x%02x' % tuple(map(int, c))
            for c in kmeans.cluster_centers_
        ]
    except Exception:
        pass

    # =========================
    # 4️⃣ UPLOAD TO SUPABASE
    # =========================
    raw_path = f"{user_id}/raw/{uuid.uuid4().hex}.png"
    ann_path = f"{user_id}/annotated/{uuid.uuid4().hex}.png"

    _, ann_buf = cv2.imencode(".png", annotated)

    supabase.storage.from_("ai-images").upload(
        raw_path,
        content,
        {"content-type": "image/png"}
    )

    supabase.storage.from_("ai-images").upload(
        ann_path,
        ann_buf.tobytes(),
        {"content-type": "image/png"}
    )

    raw_url = supabase.storage.from_("ai-images").create_signed_url(raw_path, 3600)["signedURL"]
    ann_url = supabase.storage.from_("ai-images").create_signed_url(ann_path, 3600)["signedURL"]

    # =========================
    # 5️⃣ SAVE HISTORY
    # =========================
    analysis_id = str(uuid.uuid4())
    try:
        supabase.table("vision_history").insert({
            "id": analysis_id,
            "user_id": user_id,
            "image_path": raw_path,
            "annotated_path": ann_path,
            "detections": json.dumps(detections),
            "faces": face_count,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save vision analysis: {e}")

    return {
        "objects": detections,
        "faces_detected": face_count,
        "dominant_colors": hex_colors,
        "image_url": raw_url,
        "annotated_image_url": ann_url
    }

@app.get("/vision/history")
async def vision_history(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    try:
        response = supabase.table("vision_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        rows = response.data if response.data else []
        return rows
    except Exception as e:
        logger.error(f"Failed to get vision history: {e}")
        return []

    
# ---------- Code generation ----------
@app.post("/code")
async def code_gen(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    language = body.get("language", "python").lower()
    run_flag = bool(body.get("run", False))
    user_id = body.get("user_id", str(uuid.uuid4()))  # Generate a UUID if not provided

    if not prompt:
        raise HTTPException(400, "prompt required")

    # Generate code using Groq (unchanged)
    contextual_prompt = build_contextual_prompt(
        user_id,
        f"Write a complete {language} program:\n{prompt}"
    )

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": contextual_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=get_groq_headers(),
            json=payload
        )
        r.raise_for_status()
        code = r.json()["choices"][0]["message"]["content"]

    response = {
        "language": language,
        "generated_code": code
    }

    # ✅ Run via Judge0
    if run_flag:
        lang_id = JUDGE0_LANGUAGES.get(language, 71)
        execution = await run_code_judge0(code, lang_id)
        response["execution"] = execution

    return response

@app.get("/search")
async def duck_search(q: str = Query(..., min_length=1)):
    """
    Lightweight search endpoint backed by DuckDuckGo Instant Answer API.
    Example: /search?q=python+asyncio
    """
    try:
        return await duckduckgo_search(q)
    except httpx.HTTPStatusError as e:
        logger.exception("DuckDuckGo returned HTTP error")
        raise HTTPException(502, "duckduckgo_error")
    except Exception:
        logger.exception("DuckDuckGo search failed")
        raise HTTPException(500, "search_failed")

# ---------- STT ----------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/audio/transcriptions"

    # Whisper API requires multipart/form-data, NOT JSON
    files = {
        "file": (file.filename, content, file.content_type or "audio/mpeg"),
        "model": (None, "whisper-1"),
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, headers=headers, files=files)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"OpenAI STT error: {r.text}")

    data = r.json()
    return {"transcription": data.get("text", "")}

# ----------------------------------
# NEW CHAT
# ----------------------------------

@app.post("/chat/new")
async def new_chat(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)
    cid = str(uuid.uuid4())

    try:
        supabase.table("conversations").insert({
            "id": cid,
            "user_id": user_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to create new chat: {e}")

    return {"conversation_id": cid}

@app.post("/chat/{conversation_id}")
async def send_message(conversation_id: str, req: Request, res: Response):
    user_id = await get_or_create_user(req, res)
    body = await req.json()
    text = body.get("message")

    if not text:
        raise HTTPException(400, "message required")

    msg_id = str(uuid.uuid4())
    try:
        supabase.table("messages").insert({
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": "user",
            "content": text,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save user message: {e}")

    try:
        msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").execute()
        rows = msg_response.data if msg_response.data else []
        messages = [{"role": row["role"], "content": row["content"]} for row in rows]

        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": 1024
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]

        reply_id = str(uuid.uuid4())
        supabase.table("messages").insert({
            "id": reply_id,
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": reply,
            "created_at": datetime.now().isoformat()
        }).execute()

        return {"reply": reply}
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        raise HTTPException(500, "Failed to process message")

# ----------------------------------
# LIST CHATS
# ----------------------------------
@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    try:
        response = supabase.table("conversations").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        rows = response.data if response.data else []
        return rows
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        return []

# ----------------------------------
# SEARCH CHATS
# ----------------------------------
@app.get("/chats/search")
async def search_chats(q: str, req: Request, res: Response):
    user_id = await get_or_create_user(req, res)

    try:
        response = supabase.table("conversations").select("id, title").eq("user_id", user_id).ilike("title", f"%{q}%").order("updated_at", desc=True).execute()
        rows = response.data if response.data else []
        return rows
    except Exception as e:
        logger.error(f"Failed to search chats: {e}")
        return []

# ----------------------------------
# PIN / ARCHIVE
# ----------------------------------
@app.post("/chat/{id}/pin")
async def pin_chat(id: str):
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", id).execute()
    except Exception as e:
        logger.error(f"Failed to pin chat: {e}")
    return {"status": "pinned"}

@app.post("/chat/{id}/archive")
async def archive_chat(id: str):
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", id).execute()
    except Exception as e:
        logger.error(f"Failed to archive chat: {e}")
    return {"status": "archived"}

# ----------------------------------
# FOLDER
# ----------------------------------
@app.post("/chat/{id}/folder")
async def move_folder(id: str, folder: Optional[str] = None):
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", id).execute()
    except Exception as e:
        logger.error(f"Failed to move chat to folder: {e}")
    return {"status": "moved"}

# ----------------------------------
# SHARE CHAT
# ----------------------------------
@app.post("/chat/{id}/share")
async def share_chat(id: str):
    token = uuid.uuid4().hex

    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", id).execute()
    except Exception as e:
        logger.error(f"Failed to share chat: {e}")

    return {"share_url": f"/share/{token}"}

# ----------------------------------
# VIEW SHARED CHAT (READ ONLY)
# ----------------------------------
@app.get("/share/{token}")
async def view_shared_chat(token: str):
    # In a real implementation, you would store share tokens in the database
    # For now, we'll return a placeholder
    return {
        "title": "Shared Chat",
        "messages": []
    }

# Example FastAPI endpoint that fires AI response in the background
from fastapi.responses import StreamingResponse

@app.post("/chat/stream/{conversation_id}/{user_id}")
async def chat_stream_endpoint(conversation_id: str, user_id: str, messages: list):
    """
    Streams AI response tokens to the client, saving them in Supabase in real-time.
    """
    async def event_generator():
        async for token_sse in stream_llm(user_id, conversation_id, messages):
            yield token_sse  # only yield here, no return

    # Return StreamingResponse from the endpoint, not inside the generator
    return StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache"},
)
