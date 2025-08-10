import os
import sys
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional
from threading import Thread

import torch
import gradio as gr

# Supabase client
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Optional deps (web search + vector store)
ddg = None
DDGS = None
try:
    from duckduckgo_search import ddg as _ddg
    ddg = _ddg
except Exception:
    try:
        from duckduckgo_search import DDGS as _DDGS
        DDGS = _DDGS
    except Exception:
        ddg = None
        DDGS = None

try:
    import chromadb
except Exception:
    chromadb = None

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# Optional quantization (4-bit on GPU)
BITSANDBYTES_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except Exception:
    BITSANDBYTES_AVAILABLE = False

# ===============================
# 0) Setup Supabase client if available and configured
# ===============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        print(f"⚠️ Supabase init failed: {e}")

# ===============================
# 1) Model Setup (Llama-3.1-8B-Instruct)
# ===============================
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

print("🚀 Loading Billy AI model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
except TypeError:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def _gpu_bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False

def _model_device(m) -> torch.device:
    try:
        return next(m.parameters()).device
    except Exception:
        return torch.device("cpu")

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
        load_kwargs.update(dict(device_map="auto", quantization_config=bnb_config, use_auth_token=HF_TOKEN))
    else:
        print("⚙️ No bitsandbytes: loading in half precision on GPU.")
        load_kwargs.update(dict(
            device_map="auto",
            torch_dtype=torch.bfloat16 if _gpu_bf16_supported() else torch.float16,
            use_auth_token=HF_TOKEN
        ))
else:
    print("⚠️ No GPU detected: CPU load (slow). Consider a smaller model or enable GPU runtime.")
    load_kwargs.update(dict(torch_dtype=torch.float32, use_auth_token=HF_TOKEN))

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
except TypeError:
    load_kwargs.pop("use_auth_token", None)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **load_kwargs)

MODEL_DEVICE = _model_device(model)
print(f"✅ Model loaded on: {MODEL_DEVICE}")

# ===============================
# 2) Lightweight RAG (Embeddings + Optional Chroma + In-Memory + Supabase Fallback)
# ===============================
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
except Exception as e:
    raise RuntimeError(f"Embedding model load failed: {e}")

chroma_client = None
collection = None
if chromadb is not None:
    try:
        chroma_client = chromadb.PersistentClient(path="./billy_rag_db")
        try:
            collection = chroma_client.get_collection("billy_rag")
        except Exception:
            collection = chroma_client.create_collection("billy_rag")
        print("✅ ChromaDB ready.")
    except Exception as e:
        print(f"⚠️ ChromaDB init failed: {e}; falling back to in-memory store.")

memory_store: List[Dict[str, Any]] = []

def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def search_web(query: str, max_results: int = 3) -> List[str]:
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

def store_knowledge(text: str):
    if not text or not text.strip():
        return
    try:
        vec = embedder.encode(text).tolist()
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
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
            return
        except Exception as e:
            print(f"⚠️ Supabase store failed: {e}")

    # Chromadb persistent
    if collection is not None:
        try:
            collection.add(
                documents=[text],
                embeddings=[vec],
                ids=[_stable_id(text)],
                metadatas=[{"source": "web_or_local"}],
            )
            return
        except Exception:
            pass

    # Fallback: in-memory store
    memory_store.append({"text": text, "embedding": vec})

def _cosine(a: List[float], b: List[float]) -> float:
    s = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        s += x * y
        na += x * x
        nb += y * y
    na = na ** 0.5 or 1.0
    nb = nb ** 0.5 or 1.0
    return s / (na * nb)

def retrieve_knowledge(query: str, k: int = 5) -> str:
    try:
        qvec = embedder.encode(query).tolist()
    except Exception:
        return ""

    # Supabase persistent retrieval
    if supabase is not None:
        try:
            response = supabase.table("knowledge").select("id,text,embedding").execute()
            data = response.data or []
            scored = []
            for item in data:
                emb = item.get("embedding")
                if emb and isinstance(emb, list):
                    score = _cosine(qvec, emb)
                    scored.append((item["text"], score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return " ".join([t for t, _ in scored[:k]])
        except Exception as e:
            print(f"⚠️ Supabase retrieve failed: {e}")

    # Chromadb persistent retrieval
    if collection is not None:
        try:
            res = collection.query(query_embeddings=[qvec], n_results=k)
            docs = res.get("documents", [])
            if docs and docs[0]:
                return " ".join(docs[0])
        except Exception:
            pass

    # Fallback: in-memory cosine top-k
    if not memory_store:
        return ""

    scored: List[Tuple[str, float]] = []
    for item in memory_store:
        scored.append((item["text"], _cosine(qvec, item["embedding"])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return " ".join([t for t, _ in scored[:k]])

# ===============================
# 3) Generation Utilities
# ===============================

def build_messages(system_prompt: str, chat_history: List[Tuple[str, str]], user_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for u, a in chat_history or []:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def apply_chat_template_from_messages(messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_msg = ""
        user_msg = ""
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]
        sys_msg = (sys_msg or "").strip()
        user_msg = (user_msg or "").strip()
        prefix = f"{sys_msg}\n\n" if sys_msg else ""
        return f"{prefix}User: {user_msg}\nAssistant:"

def _get_eos_token_id():
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list) and eos_id:
        return eos_id[0]
    return eos_id

def generate_text(prompt_text: str,
                  max_tokens: int = 600,
                  temperature: float = 0.6,
                  top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}
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

def summarize_text(text: str) -> str:
    system = "You are Billy AI — a precise, helpful summarizer."
    user = f"Summarize the following text in simple, clear bullet points (max 6 bullets):\n\n{text}"
    messages = build_messages(system, [], user)
    return generate_text(apply_chat_template_from_messages(messages), max_tokens=220, temperature=0.3, top_p=0.9)

def translate_text(text: str, lang: str) -> str:
    system = "You are Billy AI — an expert translator."
    user = f"Translate the following text to {lang} while preserving meaning and tone:\n\n{text}"
    messages = build_messages(system, [], user)
    return generate_text(apply_chat_template_from_messages(messages), max_tokens=220, temperature=0.3, top_p=0.9)

def explain_code(code: str) -> str:
    system = "You are Billy AI — an expert software engineer and teacher."
    user = (
        "Explain the following code step by step for a mid-level developer. "
        "Include what it does, complexity, pitfalls, and an improved version if relevant.\n\n"
        f"{code}"
    )
    messages = build_messages(system, [], user)
    return generate_text(apply_chat_template_from_messages(messages), max_tokens=400, temperature=0.5, top_p=0.9)

# ===============================
# 4) Chat Orchestration
# ===============================

def make_system_prompt(local_knowledge: str) -> str:
    base = (
        "You are Billy AI — a helpful, witty, and precise assistant. "
        "You tend to outperform GPT-3.5 on reasoning, explanation, and coding tasks. "
        "Be concise but thorough; use bullet points for clarity; cite assumptions; avoid hallucinations."
    )
    if local_knowledge:
        base += f"\nUseful context: {local_knowledge[:3000]}"
    return base

def _ingest_search(query: str, max_results: int = 3) -> int:
    snippets = search_web(query, max_results=max_results)
    for s in snippets:
        store_knowledge(s)
    return len(snippets)

def _parse_translate_command(cmd: str) -> Tuple[Optional[str], Optional[str]]:
    rest = cmd[len("/translate"):].strip()
    if not rest:
        return None, None
    for sep in [":", "|"]:
        if sep in rest:
            lang, text = rest.split(sep, 1)
            return lang.strip(), text.strip()
    parts = rest.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None

def handle_message(message: str, chat_history: List[Tuple[str, str]]) -> str:
    msg = (message or "").strip()
    if not msg:
        return "Please send a non-empty message."

    low = msg.lower()
    if low.startswith("/summarize "):
        return summarize_text(msg[len("/summarize "):].strip() or "Nothing to summarize.")
    if low.startswith("/explain "):
        return explain_code(msg[len("/explain "):].strip())
    if low.startswith("/translate"):
        lang, txt = _parse_translate_command(msg)
        if not lang or not txt:
            return "Usage: /translate <lang>: <text>"
        return translate_text(txt, lang)
    if low.startswith("/search "):
        q = msg[len("/search "):].strip()
        if not q:
            return "Usage: /search <query>"
        n = _ingest_search(q, max_results=5)
        ctx = retrieve_knowledge(q, k=5)
        if n == 0 and not ctx:
            return "No results found or web search unavailable."
        return f"Ingested {n} snippet(s). Context now includes:\n\n{ctx[:1000]}"

    if low.startswith("/remember "):
        t = msg[len("/remember "):].strip()
        if not t:
            return "Usage: /remember <text>"
        store_knowledge(t)
        return "Saved to knowledge base."

    # Normal chat completion:
    local_knowledge = retrieve_knowledge(msg, k=5)
    system_prompt = make_system_prompt(local_knowledge)

    messages = build_messages(system_prompt, chat_history, msg)
    prompt = apply_chat_template_from_messages(messages)
    return generate_text(prompt, max_tokens=600, temperature=0.6, top_p=0.9)

# ===============================
# 5) Gradio UI
# ===============================

def respond(message, history):
    tuples: List[Tuple[str, str]] = []
    for turn in history or []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u = turn[0] if turn[0] is not None else ""
            a = turn[1] if turn[1] is not None else ""
            tuples.append((str(u), str(a)))
    try:
        return handle_message(message, tuples)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Billy AI") as demo:
    gr.Markdown("## Billy AI")
    gr.Markdown(
        "Commands: /summarize <text>, /explain <code>, /translate <lang>: <text>, /search <query>, /remember <text>"
    )
    chat = gr.ChatInterface(
        fn=respond,
        title="Billy AI",
        theme="soft",
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()

# ===============================
# 6) FastAPI streaming endpoint (async)
# ===============================

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/billy/stream")
async def billy_stream(prompt: str):
    """
    Streams Billy's reply token-by-token.
    Frontend should read this stream and append to chat window in real-time.
    """

    def generate():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        thread = Thread(target=model.generate, kwargs=dict(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            streamer=streamer
        ))
        thread.start()

        for token in streamer:
            yield token

    return StreamingResponse(generate(), media_type="text/plain")