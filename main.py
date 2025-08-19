import os
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from threading import Thread

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ===== Optional deps (loaded defensively) =====
# OpenAI moderation (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# DuckDuckGo (optional web search)
try:
    from duckduckgo_search import ddg as _ddg
    ddg = _ddg
except Exception:
    ddg = None

try:
    from duckduckgo_search import DDGS as _DDGS
    DDGS = _DDGS
except Exception:
    DDGS = None

# ChromaDB (optional persistent vector store)
try:
    import chromadb
except Exception:
    chromadb = None

# Supabase (optional persistence)
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None

# SentenceTransformers for embeddings
from sentence_transformers import SentenceTransformer

# Transformers for generation
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# ===============================
# 0) Environment / Config
# ===============================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ===============================
# 1) Optional clients
# ===============================
openai_client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_KEY and OpenAI) else None

supabase: Optional["Client"] = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        supabase = None
        print(f"⚠️ Supabase init failed: {e}")

# ===============================
# 2) Load LLM
# ===============================
print("🚀 Loading Billy AI model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# ensure pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def _gpu_bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

# Optional quantization (4-bit on GPU)
BITSANDBYTES_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except Exception:
    BITSANDBYTES_AVAILABLE = False

load_kwargs: Dict[str, Any] = {}
if torch.cuda.is_available():
    if BITSANDBYTES_AVAILABLE:
        print("⚙️ Using 4-bit quantization (bitsandbytes).")
        compute_dtype = torch.bfloat16 if _gpu_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs.update(dict(device_map="auto", quantization_config=bnb_config, token=HF_TOKEN))
    else:
        print("⚙️ No bitsandbytes: loading in half precision on GPU.")
        load_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.bfloat16 if _gpu_bf16_supported() else torch.float16,
            token=HF_TOKEN
        ))
else:
    print("⚠️ No GPU detected: CPU load (slow).")
    load_kwargs.update(dict(torch_dtype=torch.float32, token=HF_TOKEN))

# Some HF versions use `token` not `use_auth_token`; try both gracefully
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
except TypeError:
    # retry with legacy kw
    load_kwargs.pop("token", None)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **load_kwargs)
    except TypeError:
        # last resort
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)

MODEL_DEVICE = next(model.parameters()).device
print(f"✅ Model loaded on: {MODEL_DEVICE}")

# ===============================
# 3) Embeddings + Storage (RAG)
# ===============================
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    print("✅ Embedding model loaded.")
except Exception as e:
    raise RuntimeError(f"Embedding model load failed: {e}")

chroma_client = None
collection = None
if chromadb is not None:
    try:
        chroma_client = chromadb.PersistentClient(path="./billy_rag_db")
        try:
            # prefer get_or_create if available
            collection = chroma_client.get_collection("billy_rag")
        except Exception:
            collection = chroma_client.create_collection("billy_rag")
        print("✅ ChromaDB ready.")
    except Exception as e:
        print(f"⚠️ ChromaDB init failed: {e}; falling back to in-memory store.")

memory_store: List[Dict[str, Any]] = []

def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _cosine(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a) or 1.0
    nb = np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (na * nb))

def embed_text(text: str) -> List[float]:
    return embedder.encode(text).tolist()

def store_knowledge(text: str):
    if not text or not text.strip():
        return
    try:
        vec = embed_text(text)
    except Exception:
        return

    # Supabase persistent storage
    if supabase is not None:
        try:
            supabase.table("knowledge").upsert({
                "id": _stable_id(text),
                "text": text,
                "embedding": vec,
                "source": "web_or_local",
                "created_at": int(time.time())
            }).execute()
            return
        except Exception as e:
            print(f"⚠️ Supabase store failed: {e}")

    # Chroma persistent
    if collection is not None:
        try:
            collection.add(
                documents=[text],
                embeddings=[vec],
                ids=[_stable_id(text)],
                metadatas=[{"source": "web_or_local"}],
            )
            return
        except Exception as e:
            print(f"⚠️ Chroma store failed: {e}")

    # Fallback: in-memory
    memory_store.append({"text": text, "embedding": vec})

def retrieve_knowledge(query: str, k: int = 5) -> str:
    try:
        qvec = embed_text(query)
    except Exception:
        return ""

    # Supabase retrieval (brute force cosine)
    if supabase is not None:
        try:
            response = supabase.table("knowledge").select("text,embedding").execute()
            data = response.data or []
            scored: List[Tuple[str, float]] = []
            for item in data:
                emb = item.get("embedding")
                if isinstance(emb, list):
                    scored.append((item["text"], _cosine(qvec, emb)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return " ".join([t for t, _ in scored[:k]])
        except Exception as e:
            print(f"⚠️ Supabase retrieve failed: {e}")

    # Chroma
    if collection is not None:
        try:
            res = collection.query(query_embeddings=[qvec], n_results=k)
            docs = res.get("documents", [])
            if docs and docs[0]:
                return " ".join(docs[0])
        except Exception as e:
            print(f"⚠️ Chroma query failed: {e}")

    # In-memory
    if memory_store:
        scored: List[Tuple[str, float]] = []
        for item in memory_store:
            scored.append((item["text"], _cosine(qvec, item["embedding"])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return " ".join([t for t, _ in scored[:k]])

    return ""

# ===============================
# 4) Web Search (optional)
# ===============================
def search_web(query: str, max_results: int = 3) -> List[str]:
    # Try ddg simple
    try:
        if ddg is not None:
            try:
                results = ddg(query, max_results=max_results)
            except TypeError:
                results = ddg(keywords=query, max_results=max_results)
            snippets = []
            for r in results or []:
                if not r:
                    continue
                snippets.append(r.get("body") or r.get("snippet") or r.get("title") or "")
            return [s for s in snippets if s and s.strip()]
    except Exception:
        pass

    # Try DDGS session
    try:
        if DDGS is not None:
            with DDGS() as d:
                results = list(d.text(query, max_results=max_results))
            snippets = []
            for r in results or []:
                if not r:
                    continue
                snippets.append(r.get("body") or r.get("snippet") or r.get("title") or r.get("href") or "")
            return [s for s in snippets if s and s.strip()]
    except Exception:
        pass

    return []

def ingest_search(query: str, max_results: int = 3) -> int:
    snippets = search_web(query, max_results=max_results)
    for s in snippets:
        store_knowledge(s)
    return len(snippets)

# ===============================
# 5) Text Generation utils
# ===============================
def _get_eos_token_id():
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list) and eos_id:
        return eos_id[0]
    return eos_id

def make_system_prompt(local_knowledge: str) -> str:
    base = (
        "You are Billy AI — a helpful, witty, and precise assistant. "
        "Be concise but thorough; use bullet points for clarity; cite assumptions; avoid hallucinations."
    )
    if local_knowledge:
        base += f"\nUseful context: {local_knowledge[:3000]}"
    return base

def build_prompt(user_prompt: str, chat_history: List[Tuple[str, str]]) -> str:
    context = retrieve_knowledge(user_prompt, k=5)
    system = make_system_prompt(context)
    # Llama-style simple instruction block (robust across HF versions)
    history_str = ""
    for u, a in chat_history:
        if u:
            history_str += f"\nUser: {u}\nAssistant: {a or ''}"
    return (
        f"<s>[INST]{system}[/INST]</s>\n"
        f"{history_str}\n"
        f"[INST]User: {user_prompt}\nAssistant:[/INST]"
    )

def generate_text(prompt_text: str,
                  max_tokens: int = 600,
                  temperature: float = 0.6,
                  top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(MODEL_DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=min(max_tokens, 2048),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=_get_eos_token_id(),
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if text.startswith(prompt_text):
        return text[len(prompt_text):].strip()
    return text.strip()

# ===============================
# 6) Moderation helper (input + output)
# ===============================
def is_safe_message(text: str) -> Tuple[bool, str]:
    """
    Returns (safe, reason_if_blocked).
    If no OpenAI key/moderation, defaults to safe.
    """
    if not (openai_client and OPENAI_KEY):
        return True, ""
    try:
        resp = openai_client.moderations.create(model="omni-moderation-latest", input=text)
        flagged = bool(resp.results[0].flagged)
        return (not flagged), ("Blocked by moderation." if flagged else "")
    except Exception:
        # Fail open if moderation API fails
        return True, ""

# ===============================
# 7) FastAPI app + schemas
# ===============================
app = FastAPI(title="Billy AI API", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    chat_history: Optional[List[Tuple[str, str]]] = []
    max_tokens: int = 600
    temperature: float = 0.6
    top_p: float = 0.9

class EmbedRequest(BaseModel):
    texts: List[str]

class RememberRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    max_results: int = 3

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    safe, reason = is_safe_message(req.prompt)
    if not safe:
        return JSONResponse({"error": reason or "Unsafe prompt"}, status_code=400)

    prompt = build_prompt(req.prompt, req.chat_history or [])
    output = generate_text(prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)

    # Moderate output
    safe_out, _ = is_safe_message(output)
    if not safe_out:
        return JSONResponse({"error": "Response blocked by moderation."}, status_code=400)

    return {"response": output}

@app.post("/stream")
def stream(req: GenerateRequest):
    safe, reason = is_safe_message(req.prompt)
    if not safe:
        return StreamingResponse(iter([reason or "Unsafe prompt"]), media_type="text/plain")

    prompt = build_prompt(req.prompt, req.chat_history or [])
    inputs = tokenizer(prompt, return_tensors="pt").to(MODEL_DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate_tokens():
        # Launch generation in a background thread
        thread = Thread(target=model.generate, kwargs=dict(
            **inputs,
            max_new_tokens=min(req.max_tokens, 2048),
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=_get_eos_token_id(),
            streamer=streamer
        ))
        thread.start()

        response_accum = ""
        for token in streamer:
            response_accum += token
            yield token

        # Moderate full response
        safe_out, _ = is_safe_message(response_accum)
        if not safe_out:
            yield "\n\n⚠️ Response blocked by moderation."

    return StreamingResponse(generate_tokens(), media_type="text/plain")

@app.post("/embed")
def embed(req: EmbedRequest):
    vectors = [embed_text(t) for t in req.texts]
    return {"embeddings": vectors}

@app.post("/remember")
def remember(req: RememberRequest):
    store_knowledge(req.text)
    return {"status": "stored"}

@app.post("/search")
def web_search(req: SearchRequest):
    n = ingest_search(req.query, max_results=req.max_results)
    ctx = retrieve_knowledge(req.query, k=5)
    return {"ingested": n, "context_sample": ctx[:1000]}