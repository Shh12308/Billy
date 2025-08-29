# main.py — Billy AI (all-in-one, single file)
# 🎯 Features: LLM + RAG + Tools + Agent + ASR + TTS + Vision + Music + Redis cache/RL + Moderation + Streaming
# SECURITY NOTE: The python sandbox runs in a subprocess with timeout. For public use, run this API in Docker.

import os
import time
import uuid
import json
import math
import shlex
import tempfile
import hashlib
import subprocess
from threading import Thread
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, BackgroundTasks, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================
# Optional heavy deps (guarded)
# ===============================
try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
except Exception:
    AutoTokenizer = AutoModelForCausalLM = TextIteratorStreamer = BitsAndBytesConfig = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from supabase import create_client
except Exception:
    create_client = None

try:
    import redis as redis_lib
except Exception:
    redis_lib = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except Exception:
    Blip2Processor = Blip2ForConditionalGeneration = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from audiocraft.models import musicgen as musicgen_lib
except Exception:
    musicgen_lib = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from duckduckgo_search import ddg as ddg_func
except Exception:
    ddg_func = None

# Prometheus (optional)
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except Exception:
    generate_latest = CONTENT_TYPE_LATEST = None

# ===============================
# Environment / Config
# ===============================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
DEFAULT_MODEL = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
FRONTEND_API_KEY = os.getenv("FRONTEND_API_KEY", "changeme")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://zynara.xyz")
REDIS_URL = os.getenv("REDIS_URL")
ASR_MODEL_SIZE = os.getenv("ASR_MODEL", "small")
COQUI_TTS_MODEL = os.getenv("COQUI_TTS_MODEL")  # optional, else auto-pick
CDN_BASE_URL = os.getenv("CDN_BASE_URL")
DISABLE_MULTIMODAL = os.getenv("DISABLE_MULTIMODAL", "0") == "1"

# ===============================
# App + CORS
# ===============================
app = FastAPI(title="Billy AI — All-in-one", version="1.0.0")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Clients (OpenAI, Redis, Chroma, Supabase)
# ===============================
openai_client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_KEY and OpenAI) else None

redis_client = None
if redis_lib and REDIS_URL:
    try:
        redis_client = redis_lib.from_url(REDIS_URL)
        print("✅ Redis connected")
    except Exception as e:
        print("⚠️ Redis init failed:", e)

embedder = None
if SentenceTransformer is not None:
    try:
        embedder = SentenceTransformer(EMBED_MODEL)
        print("✅ Embedder loaded")
    except Exception as e:
        print("⚠️ Embedder init failed:", e)

chroma_client = None
chroma_collection = None
if chromadb is not None:
    try:
        chroma_client = chromadb.PersistentClient(path="./billy_rag_db")
        try:
            chroma_collection = chroma_client.get_collection("billy_rag")
        except Exception:
            chroma_collection = chroma_client.create_collection("billy_rag")
        print("✅ Chroma ready")
    except Exception as e:
        print("⚠️ Chroma init failed:", e)

supabase_client = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)

# ===============================
# Helpers (IDs, cosine, cache/RL, moderation)
# ===============================
def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _cosine(a: List[float], b: List[float]) -> float:
    import numpy as np
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a) or 1.0
    nb = np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (na * nb))

async def api_key_auth(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        if FRONTEND_API_KEY == "changeme":
            return True
        raise HTTPException(status_code=401, detail="Missing API key")
    if x_api_key != FRONTEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def rate_limit(key: str, limit: int = 60, window: int = 60) -> bool:
    if not redis_client:
        return True
    try:
        p = redis_client.pipeline()
        p.incr(key)
        p.expire(key, window)
        val, _ = p.execute()
        return int(val) <= limit
    except Exception:
        return True

def cache_get(key: str):
    if not redis_client:
        return None
    try:
        v = redis_client.get(key)
        return json.loads(v) if v else None
    except Exception:
        return None

def cache_set(key: str, value, ttl: int = 300):
    if not redis_client:
        return
    try:
        redis_client.set(key, json.dumps(value), ex=ttl)
    except Exception:
        pass

def is_safe_message(text: str) -> Tuple[bool, str]:
    if not text:
        return True, ""
    if openai_client is None:
        # very simple heuristic fallback
        banned = ["kill", "terror", "bomb", "nuke"]
        if any(b in text.lower() for b in banned):
            return False, "Blocked by local safety heuristic."
        return True, ""
    try:
        resp = openai_client.moderations.create(model="omni-moderation-latest", input=text)
        flagged = bool(resp.results[0].flagged)
        return (not flagged), ("Blocked by moderation." if flagged else "")
    except Exception:
        return True, ""

# ===============================
# RAG storage (Chroma/Supabase/memory)
# ===============================
memory_store: List[Dict[str, Any]] = []

def embed_text_local(text: str) -> List[float]:
    if not embedder:
        raise RuntimeError("Embedder not loaded.")
    return embedder.encode(text).tolist()

def store_knowledge(text: str, user_id: Optional[str] = None):
    if not text or not text.strip():
        return
    try:
        vec = embed_text_local(text)
    except Exception:
        return
    idx = _stable_id(text)
    if supabase_client:
        try:
            row = {"id": idx, "text": text, "embedding": vec, "source": "user", "created_at": int(time.time())}
            if user_id:
                row["user_id"] = user_id
            supabase_client.table("knowledge").upsert(row).execute()
            return
        except Exception:
            pass
    if chroma_collection:
        try:
            chroma_collection.add(documents=[text], embeddings=[vec], ids=[idx], metadatas=[{"user_id": user_id}])
            return
        except Exception:
            pass
    memory_store.append({"text": text, "embedding": vec, "user_id": user_id})

def retrieve_knowledge(query: str, k: int = 5) -> str:
    try:
        qvec = embed_text_local(query)
    except Exception:
        return ""
    if supabase_client:
        try:
            resp = supabase_client.table("knowledge").select("text,embedding").execute()
            data = resp.data or []
            scored = []
            for item in data:
                emb = item.get("embedding")
                if isinstance(emb, list):
                    scored.append((item["text"], _cosine(qvec, emb)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return " ".join([t for t, _ in scored[:k]])
        except Exception:
            pass
    if chroma_collection:
        try:
            res = chroma_collection.query(query_embeddings=[qvec], n_results=k)
            docs = res.get("documents", [])
            if docs and docs[0]:
                return " ".join(docs[0])
        except Exception:
            pass
    if memory_store:
        scored = []
        for item in memory_store:
            scored.append((item["text"], _cosine(qvec, item["embedding"])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return " ".join([t for t, _ in scored[:k]])
    return ""

def delete_memory_by_id(mem_id: str) -> bool:
    ok = False
    if supabase_client:
        try:
            supabase_client.table("knowledge").delete().eq("id", mem_id).execute()
            ok = True
        except Exception:
            pass
    if chroma_collection:
        try:
            chroma_collection.delete(ids=[mem_id])
            ok = True
        except Exception:
            pass
    global memory_store
    before = len(memory_store)
    memory_store = [m for m in memory_store if _stable_id(m.get("text","")) != mem_id]
    return ok or (len(memory_store) < before)

# ===============================
# Tools & Agent
# ===============================
class Tool:
    name: str
    description: str
    def run(self, args: str) -> Dict[str, Any]:
        raise NotImplementedError

TOOLS: Dict[str, Tool] = {}

def register_tool(tool: Tool):
    TOOLS[tool.name] = tool

def call_tool(name: str, args: str) -> Dict[str, Any]:
    tool = TOOLS.get(name)
    if not tool:
        return {"ok": False, "error": f"Tool '{name}' not found"}
    start = time.time()
    try:
        res = tool.run(args)
        return {"ok": True, "result": res, "runtime": time.time() - start}
    except Exception as e:
        return {"ok": False, "error": str(e), "runtime": time.time() - start}

class Calculator(Tool):
    name = "calculator"
    description = "Evaluate math expressions using Python's math (e.g., '2+2', 'sin(1)')."
    def run(self, args: str) -> Dict[str, Any]:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        expr = args.strip()
        if "__" in expr:
            raise ValueError("Invalid expression")
        val = eval(expr, {"__builtins__": {}}, allowed)
        return {"input": expr, "value": val}

register_tool(Calculator())

class PythonSandbox(Tool):
    name = "python_sandbox"
    description = "Run a short Python script in a subprocess (timeout 2s). For production, isolate via container."
    def run(self, args: str) -> Dict[str, Any]:
        code = args
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "script.py")
            with open(path, "w") as f:
                f.write(code)
            cmd = f"timeout 2 python3 {shlex.quote(path)}"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = proc.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"stdout": "", "stderr": "Execution timed out", "returncode": 124}
            return {"stdout": out.decode("utf-8", errors="ignore"), "stderr": err.decode("utf-8", errors="ignore"), "returncode": proc.returncode}

register_tool(PythonSandbox())

class WebSearchTool(Tool):
    name = "web_search"
    description = "DuckDuckGo search. Returns top snippets (no links)."
    def run(self, args: str) -> Dict[str, Any]:
        if not ddg_func:
            return {"error": "duckduckgo-search not installed"}
        q = args.strip()
        try:
            results = ddg_func(q, max_results=3)
        except TypeError:
            results = ddg_func(keywords=q, max_results=3)
        snippets = []
        for r in results or []:
            snippets.append(r.get("body") or r.get("snippet") or r.get("title") or "")
        return {"query": q, "snippets": [s for s in snippets if s]}

register_tool(WebSearchTool())

def agent_run(llm_func, system_prompt: str, user_prompt: str, chat_history: List[Tuple[str,str]] = None, max_steps: int = 4):
    chat_history = chat_history or []
    tools_info = "\n".join([f"{name}: {TOOLS[name].description}" for name in TOOLS])
    agent_hdr = (
        f"{system_prompt}\n\nAvailable tools:\n{tools_info}\n\n"
        "When you want to call a tool respond ONLY with a JSON object:\n"
        '{"action":"tool_name","args":"..."}\n'
        'When finished respond: {"action":"final","answer":"..."}\n'
    )
    context = agent_hdr + f"\nUser: {user_prompt}\n"
    for _ in range(max_steps):
        model_out = llm_func(context)
        try:
            first_line = model_out.strip().splitlines()[0]
            action_obj = json.loads(first_line)
        except Exception:
            return {"final": model_out}
        act = action_obj.get("action")
        if act == "final":
            return {"final": action_obj.get("answer", "")}
        args = action_obj.get("args", "")
        tool_res = call_tool(act, args)
        context += f"\nToolCall: {json.dumps({'tool': act, 'args': args})}\nToolResult: {json.dumps(tool_res)}\n"
    return {"final": "Max steps reached. Partial reasoning returned.", "context": context}

# ===============================
# Multimodal (ASR / TTS / Vision / Music)
# ===============================
ASR_MODEL = None
def init_asr():
    global ASR_MODEL
    if DISABLE_MULTIMODAL or WhisperModel is None:
        return None
    if ASR_MODEL is None:
        try:
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            ASR_MODEL = WhisperModel(ASR_MODEL_SIZE, device=device, compute_type=compute_type)
            print(f"✅ ASR model loaded: {ASR_MODEL_SIZE} on {device}")
        except Exception as e:
            print("⚠️ ASR init failed:", e)
            ASR_MODEL = None
    return ASR_MODEL

def transcribe_audio(path: str, language: Optional[str] = None, task: str = "transcribe"):
    if init_asr() is None:
        return {"text": "asr-disabled"}
    segments, info = ASR_MODEL.transcribe(path, language=language, task=task)
    text = " ".join(seg.text for seg in segments)
    return {"text": text, "duration": getattr(info, "duration", None)}

TTS_CLIENT = None
def init_tts():
    global TTS_CLIENT
    if DISABLE_MULTIMODAL or CoquiTTS is None:
        return None
    if TTS_CLIENT is None:
        try:
            TTS_CLIENT = CoquiTTS(model_name=COQUI_TTS_MODEL) if COQUI_TTS_MODEL else CoquiTTS()
            print("✅ Coqui TTS initialized")
        except Exception as e:
            print("⚠️ TTS init failed:", e)
            TTS_CLIENT = None
    return TTS_CLIENT

def synthesize_to_file(text: str, voice: Optional[str] = None, out_path: Optional[str] = None):
    out_path = out_path or f"/tmp/tts_{uuid.uuid4().hex}.mp3"
    if init_tts() is None:
        open(out_path, "wb").close()
        return {"path": out_path}
    try:
        # Some models require specific speaker names; None often works with single-speaker
        TTS_CLIENT.tts_to_file(text=text, speaker=voice, file_path=out_path)
    except Exception as e:
        print("⚠️ TTS synthesis failed:", e)
        open(out_path, "wb").close()
    return {"path": out_path}

BLIP_PROCESSOR = BLIP_MODEL = None
def init_vision():
    global BLIP_PROCESSOR, BLIP_MODEL
    if DISABLE_MULTIMODAL or (Blip2Processor is None or Blip2ForConditionalGeneration is None or Image is None):
        return None, None
    if BLIP_MODEL is None:
        try:
            model_name = "Salesforce/blip2-flan-t5-base"
            BLIP_PROCESSOR = Blip2Processor.from_pretrained(model_name)
            BLIP_MODEL = Blip2ForConditionalGeneration.from_pretrained(model_name)
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            BLIP_MODEL.to(device)
            print(f"✅ BLIP-2 loaded on {device}")
        except Exception as e:
            print("⚠️ Vision init failed:", e)
            BLIP_PROCESSOR = BLIP_MODEL = None
    return BLIP_PROCESSOR, BLIP_MODEL

def caption_image(path: str) -> str:
    proc, model = init_vision()
    if not proc or not model:
        return "A photo (caption placeholder)."
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    img = Image.open(path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    out_ids = model.generate(**inputs, max_new_tokens=64)
    return proc.decode(out_ids[0], skip_special_tokens=True)

def ocr_image(path: str) -> str:
    # Placeholder (integrate easyocr or pytesseract as needed)
    return "OCR placeholder text."

MUSIC_MODEL = None
def init_music():
    global MUSIC_MODEL
    if DISABLE_MULTIMODAL or musicgen_lib is None:
        return None
    if MUSIC_MODEL is None:
        try:
            MUSIC_MODEL = musicgen_lib.MusicGen.get_pretrained("melody")
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
            MUSIC_MODEL.to(device)
            print(f"✅ MusicGen loaded on {device}")
        except Exception as e:
            print("⚠️ Music init failed:", e)
            MUSIC_MODEL = None
    return MUSIC_MODEL

def generate_music(prompt: str, duration: int = 20) -> Dict[str, Any]:
    out = f"/tmp/music_{uuid.uuid4().hex}.wav"
    if init_music() is None:
        open(out, "wb").close()
        return {"path": out}
    try:
        wav = MUSIC_MODEL.generate([prompt], duration=duration)
        # audiocraft write helper changed over time; safest: torchaudio or soundfile
        try:
            import torchaudio
            torchaudio.save(out, wav[0].cpu(), 32000)
        except Exception:
            # fallback empty file
            open(out, "wb").close()
    except Exception as e:
        print("⚠️ Music generation failed:", e)
        open(out, "wb").close()
    return {"path": out}

# ===============================
# LLM loading & generation
# ===============================
MODEL = None
TOKENIZER = None
MODEL_DEVICE = "cpu"

def load_llm(model_id: str = DEFAULT_MODEL, use_bnb: bool = True):
    global MODEL, TOKENIZER, MODEL_DEVICE
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required. pip install transformers")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    kwargs = {}
    if torch is not None and torch.cuda.is_available():
        MODEL_DEVICE = "cuda"
        if BitsAndBytesConfig is not None and use_bnb:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            kwargs.update(dict(device_map="auto", quantization_config=bnb, token=HF_TOKEN))
        else:
            kwargs.update(dict(device_map="auto",
                               torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                               token=HF_TOKEN))
    else:
        MODEL_DEVICE = "cpu"
        kwargs.update(dict(torch_dtype=torch.float32, token=HF_TOKEN))
    try:
        MODEL = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        MODEL = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_TOKEN, **kwargs)
    print(f"✅ LLM loaded on {MODEL_DEVICE}")

def _get_eos_token_id():
    if TOKENIZER is None:
        return None
    eid = getattr(TOKENIZER, "eos_token_id", None)
    if isinstance(eid, list) and eid:
        return eid[0]
    return eid

def make_system_prompt(local_knowledge: str) -> str:
    base = ("You are Billy AI — a helpful, witty, and precise assistant. "
            "Be concise but thorough; use bullet points; cite assumptions; avoid hallucinations.")
    if local_knowledge:
        base += f"\nUseful context: {local_knowledge[:3000]}"
    return base

def build_prompt(user_prompt: str, chat_history: List[Tuple[str,str]]) -> str:
    context = retrieve_knowledge(user_prompt, k=5)
    system = make_system_prompt(context)
    hist = ""
    for u, a in (chat_history or []):
        if u:
            hist += f"\nUser: {u}\nAssistant: {a or ''}"
    return f"<s>[INST]{system}[/INST]</s>\n{hist}\n[INST]User: {user_prompt}\nAssistant:[/INST]"

def generate_text_sync(prompt_text: str, max_tokens: int = 600, temperature: float = 0.6, top_p: float = 0.9) -> str:
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("LLM not loaded")
    inputs = TOKENIZER(prompt_text, return_tensors="pt").to(MODEL_DEVICE)
    out_ids = MODEL.generate(
        **inputs,
        max_new_tokens=min(max_tokens, 2048),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=TOKENIZER.pad_token_id,
        eos_token_id=_get_eos_token_id(),
    )
    text = TOKENIZER.decode(out_ids[0], skip_special_tokens=True)
    if text.startswith(prompt_text):
        return text[len(prompt_text):].strip()
    return text.strip()

def stream_generate(prompt_text: str, max_tokens: int = 600, temperature: float = 0.6, top_p: float = 0.9):
    if MODEL is None or TOKENIZER is None:
        yield "ERROR: model not loaded"
        return
    inputs = TOKENIZER(prompt_text, return_tensors="pt").to(MODEL_DEVICE)
    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    def _gen():
        MODEL.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 2048),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=_get_eos_token_id(),
            streamer=streamer
        )
    Thread(target=_gen).start()
    for tok in streamer:
        yield tok

# ===============================
# Schemas
# ===============================
class GenerateRequest(BaseModel):
    prompt: str
    chat_history: Optional[List[Tuple[str,str]]] = []
    max_tokens: int = 600
    temperature: float = 0.6
    top_p: float = 0.9
    max_steps: int = 4  # for agent if triggered

class EmbedRequest(BaseModel):
    texts: List[str]

class RememberRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 3

class MusicRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    duration: Optional[int] = 20

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    format: Optional[str] = "mp3"

class AgentRequest(BaseModel):
    prompt: str
    chat_history: Optional[List[Tuple[str,str]]] = []
    max_steps: int = 4

class ForgetRequest(BaseModel):
    id: str

# ===============================
# Endpoints
# ===============================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", dependencies=[Depends(api_key_auth)])
def generate(req: GenerateRequest):
    # rate-limit
    rl_key = f"rl:{hashlib.sha1((req.prompt or '').encode()).hexdigest()}"
    if not rate_limit(rl_key, limit=120, window=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    safe, reason = is_safe_message(req.prompt)
    if not safe:
        return JSONResponse({"error": reason or "Unsafe prompt"}, status_code=400)

    # Agent trigger heuristic: allow "CALL_TOOL" or "use tool:" prefixes
    if req.prompt.strip().lower().startswith("use tool:") or "CALL_TOOL" in req.prompt:
        def _llm(p): return generate_text_sync(p, max_tokens=400, temperature=0.2, top_p=0.9)
        out = agent_run(_llm, make_system_prompt(retrieve_knowledge(req.prompt, k=5)), req.prompt, req.chat_history or [], max_steps=req.max_steps)
        return out

    prompt = build_prompt(req.prompt, req.chat_history or [])
    cache_key = f"resp:{hashlib.sha1(prompt.encode()).hexdigest()}"
    cached = cache_get(cache_key)
    if cached:
        return {"response": cached}

    out = generate_text_sync(prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)
    safe_out, _ = is_safe_message(out)
    if not safe_out:
        return JSONResponse({"error": "Response blocked by moderation."}, status_code=400)
    cache_set(cache_key, out, ttl=30)
    return {"response": out}

@app.post("/stream", dependencies=[Depends(api_key_auth)])
def stream(req: GenerateRequest):
    safe, reason = is_safe_message(req.prompt)
    if not safe:
        return StreamingResponse(iter([reason or "Unsafe prompt"]), media_type="text/plain")
    prompt = build_prompt(req.prompt, req.chat_history or [])
    def gen():
        for chunk in stream_generate(prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p):
            yield chunk
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/agent", dependencies=[Depends(api_key_auth)])
def agent_endpoint(req: AgentRequest):
    def _llm(p): return generate_text_sync(p, max_tokens=400, temperature=0.2, top_p=0.9)
    out = agent_run(_llm, make_system_prompt(retrieve_knowledge(req.prompt, k=5)), req.prompt, req.chat_history or [], max_steps=req.max_steps)
    return out

@app.post("/embed", dependencies=[Depends(api_key_auth)])
def embed(req: EmbedRequest):
    if not embedder:
        return JSONResponse({"error": "Embedder not loaded."}, status_code=500)
    vecs = [embed_text_local(t) for t in req.texts]
    # also store them for RAG convenience
    for t in req.texts:
        store_knowledge(t)
    return {"embeddings": vecs}

@app.post("/remember", dependencies=[Depends(api_key_auth)])
def remember(req: RememberRequest):
    store_knowledge(req.text, user_id=req.user_id if hasattr(req, "user_id") else None)
    return {"status": "stored"}

@app.post("/search", dependencies=[Depends(api_key_auth)])
def web_search(req: SearchRequest):
    # call web_search tool and ingest snippets into RAG
    ws = TOOLS.get("web_search")
    if not ws:
        return {"ingested": 0, "context_sample": ""}
    res = ws.run(req.query)
    count = 0
    for s in res.get("snippets", []):
        store_knowledge(s)
        count += 1
    ctx = retrieve_knowledge(req.query, k=req.max_results or 3)
    return {"ingested": count, "context_sample": ctx[:1000]}

@app.post("/music", dependencies=[Depends(api_key_auth)])
def music(req: MusicRequest, background_tasks: BackgroundTasks):
    try:
        tmp = generate_music(req.prompt, duration=req.duration or 20).get("path")
        if CDN_BASE_URL:
            url = f"{CDN_BASE_URL}/{os.path.basename(tmp)}"
        else:
            url = tmp
        return {"reply": f"Generated music for: {req.prompt}", "audioUrl": url}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/tts", dependencies=[Depends(api_key_auth)])
def tts(req: TTSRequest):
    try:
        out = synthesize_to_file(req.text, voice=req.voice)
        if CDN_BASE_URL:
            url = f"{CDN_BASE_URL}/tts/{os.path.basename(out['path'])}"
            return {"audioUrl": url}
        return FileResponse(out["path"], media_type="audio/mpeg", filename=os.path.basename(out["path"]))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/tts_stream", dependencies=[Depends(api_key_auth)])
def tts_stream(req: TTSRequest):
    try:
        out = synthesize_to_file(req.text, voice=req.voice)
        def iterfile():
            with open(out["path"], "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    yield chunk
        return StreamingResponse(iterfile(), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/asr", dependencies=[Depends(api_key_auth)])
async def asr(file: UploadFile = File(...)):
    try:
        tmp = f"/tmp/asr_{uuid.uuid4().hex}_{file.filename}"
        with open(tmp, "wb") as f:
            f.write(await file.read())
        res = transcribe_audio(tmp)
        return {"transcript": res.get("text", "")}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/vision", dependencies=[Depends(api_key_auth)])
async def vision(file: UploadFile = File(...), task: Optional[str] = "caption"):
    try:
        tmp = f"/tmp/vision_{uuid.uuid4().hex}.jpg"
        with open(tmp, "wb") as f:
            f.write(await file.read())
        if (task or "").lower() == "ocr":
            text = ocr_image(tmp)
            return {"text": text}
        caption = caption_image(tmp)
        return {"caption": caption}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/generate")
async def websocket_generate(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            prompt = data.get("prompt", "")
            chat_history = data.get("chat_history", [])
            max_tokens = int(data.get("max_tokens", 256))
            temperature = float(data.get("temperature", 0.6))
            top_p = float(data.get("top_p", 0.9))

            built = build_prompt(prompt, chat_history or [])
            inputs = TOKENIZER(built, return_tensors="pt").to(MODEL_DEVICE)
            streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)

            def run_gen():
                MODEL.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 2048),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=TOKENIZER.pad_token_id,
                    eos_token_id=_get_eos_token_id(),
                    streamer=streamer
                )
            Thread(target=run_gen).start()

            accumulated = ""
            for token in streamer:
                accumulated += token
                await ws.send_json({"delta": token})
            safe_out, _ = is_safe_message(accumulated)
            if not safe_out:
                await ws.send_json({"done": True, "final": "⚠️ Response blocked by moderation."})
            else:
                await ws.send_json({"done": True, "final": accumulated})
    except Exception:
        await ws.close()

@app.get("/admin/memory")
def admin_memory():
    return {"count": len(memory_store)}

@app.post("/forget", dependencies=[Depends(api_key_auth)])
def forget(req: ForgetRequest):
    try:
        ok = delete_memory_by_id(req.id)
        if ok:
            return {"status": "deleted"}
        return JSONResponse({"error": "Not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/metrics")
def metrics():
    if not generate_latest or not CONTENT_TYPE_LATEST:
        return JSONResponse({"error": "prometheus-client not installed"}, status_code=500)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ===============================
# Startup
# ===============================
@app.on_event("startup")
def on_startup():
    try:
        load_llm(DEFAULT_MODEL)
    except Exception as e:
        print("⚠️ LLM load failed:", e)
    try:
        if not DISABLE_MULTIMODAL:
            init_asr()
            init_tts()
            init_vision()
            init_music()
    except Exception:
        pass
    print("🚀 Billy AI startup complete")

# ===============================
# Run instructions
# ===============================
# uvicorn main:app --host 0.0.0.0 --port 8000
# Set FRONTEND_API_KEY in env. Optionally set HF_TOKEN, REDIS_URL, SUPABASE_URL/SUPABASE_KEY, CDN_BASE_URL.
# For CPU-only or fast boot without heavy models: export DISABLE_MULTIMODAL=1