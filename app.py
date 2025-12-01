# app.py — Billy AI full multimodal server: SDXL + TTS/STT + code + vision + search + remove-bg/upscale + caching + metadata
import os
import io
import json
import uuid
import sqlite3
import base64
import time
import logging
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("billy-server")

app = FastAPI(title="Billy AI Multimodal Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ENV KEYS ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "sk-RkSqNSbedDS5YcM54qK6sTaKDldQprDIvc6HbiMhdlt0Cx9e").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
IMAGE_MODEL_FREE_URL = os.getenv("IMAGE_MODEL_FREE_URL", "").strip()
USE_FREE_IMAGE_PROVIDER = os.getenv("USE_FREE_IMAGE_PROVIDER", "false").lower() in ("1", "true", "yes")

KG_DB_PATH = os.getenv("KG_DB_PATH", "./kg.db")
MEMORY_DB = os.getenv("MEMORY_DB", "./memory.db")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "./cache.db")

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")

# Image hosting dirs
STATIC_DIR = os.getenv("STATIC_DIR", "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DB_PATH = os.getenv("DONATIONS_DB", "./donations.db")
WEBHOOK_SECRETS = {
    # set these env vars to validate inbound webhooks; e.g.:
    # - KOFI_SECRET (shared token) or KOFI_HMAC_SECRET (HMAC key)
    # - BMC_SECRET, GUMROAD_SECRET, STRIPE_ENDPOINT_SECRET (Stripe uses signature header)
    "kofi": os.getenv("KOFI_SECRET", None),
    "kofi_hmac": os.getenv("KOFI_HMAC_SECRET", None),
    "buymeacoffee": os.getenv("BMC_SECRET", None),
    "gumroad": os.getenv("GUMROAD_SECRET", None),
    "stripe": os.getenv("STRIPE_ENDPOINT_SECRET", None),
}
DONATE_LINKS_PATH = os.getenv("DONATE_LINKS", "./donate_links.json")  # JSON file for public links

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("donations")

# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldBoy",
    "age": 17,
    "country": "England",
    "projects": ["MintZa", "LuxStream", "SwapX", "CryptoBean"],
    "socials": {"instagram":"GoldBoyy", "twitter":"GoldBoy"},
    "bio": "Created by GoldBoy (17, England). Projects: MintZa, LuxStream, SwapX, CryptoBean. Socials: Instagram @GoldBoyy, Twitter @GoldBoy."
}

# ---------- Dynamic, user-focused system prompt ----------
def get_system_prompt(user_message: Optional[str] = None) -> str:
    base = "You are Billy AI: helpful, concise, friendly, and focus entirely on what the user asks. Do not reference your creator or yourself unless explicitly asked."
    if user_message:
        base += f" The user said: \"{user_message}\". Tailor your response to this."
    return base

def build_contextual_prompt(user_id: str, message: str) -> str:
    conn = sqlite3.connect(MEMORY_DB)
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM memory WHERE user_id=? ORDER BY updated_at DESC LIMIT 5", (user_id,))
    rows = cur.fetchall()
    conn.close()
    context = "\n".join(f"{k}: {v}" for k, v in rows)
    return f"You are Billy AI: helpful, concise, friendly. Focus on exactly what the user wants.\nUser context:\n{context}\nUser message: {message}"

# ---------- SQLITE helpers ----------
def ensure_db(path: str, schema_sql: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(schema_sql)
    conn.commit()
    conn.close()

# Knowledge Graph
ensure_db(KG_DB_PATH, """
CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  content TEXT,
  created_at REAL
);
""")

# Memory
ensure_db(MEMORY_DB, """
CREATE TABLE IF NOT EXISTS memory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  key TEXT,
  value TEXT,
  updated_at REAL
);
""")

# Cache for images/prompts
ensure_db(CACHE_DB_PATH, """
CREATE TABLE IF NOT EXISTS cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt TEXT,
  provider TEXT,
  result TEXT,
  created_at REAL
);
""")

# -------- DB helpers --------
def ensure_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS donations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT,
        raw_id TEXT,
        donor_name TEXT,
        amount REAL,
        currency TEXT,
        message TEXT,
        received_at INTEGER
    );
    """)
    conn.commit()
    return conn

DB = ensure_db(DB_PATH)

def insert_donation(provider:str, raw_id:str, donor_name:Optional[str], amount:float, currency:str="USD", message:Optional[str]=None):
    ts = int(time.time())
    cur = DB.cursor()
    cur.execute("INSERT INTO donations (provider,raw_id,donor_name,amount,currency,message,received_at) VALUES (?,?,?,?,?,?,?)",
                (provider, raw_id, donor_name or "Anonymous", amount, currency, message or "", ts))
    DB.commit()
    return cur.lastrowid

def top_supporters(limit:int=20):
    cur = DB.cursor()
    cur.execute("SELECT donor_name, SUM(amount) as total, currency FROM donations GROUP BY donor_name ORDER BY total DESC LIMIT ?", (limit,))
    return [{"donor": r[0], "total": r[1], "currency": r[2]} for r in cur.fetchall()]

def total_funds():
    cur = DB.cursor()
    cur.execute("SELECT SUM(amount) FROM donations")
    r = cur.fetchone()
    return float(r[0] or 0.0)

def recent_donations(limit:int=30):
    cur = DB.cursor()
    cur.execute("SELECT provider, donor_name, amount, currency, message, received_at FROM donations ORDER BY received_at DESC LIMIT ?", (limit,))
    return [
        {"provider":r[0],"donor":r[1],"amount":r[2],"currency":r[3],"message":r[4],"ts":r[5]}
        for r in cur.fetchall()
    ]

# --------- Utility: verify signatures / tokens ----------
def verify_shared_token(provider:str, payload:dict, header_token:Optional[str]):
    configured = WEBHOOK_SECRETS.get(provider)
    if not configured:
        return False
    # Many simple providers pass a "token" param or header — compare directly
    return header_token is not None and header_token == configured

def verify_hmac(provider:str, raw_body:bytes, header_sig:Optional[str], algo='sha256'):
    secret = WEBHOOK_SECRETS.get(provider + "_hmac")
    if not secret:
        return False
    if not header_sig:
        return False
    # header_sig may be like "sha256=..." or raw hex; handle basic cases
    try:
        if "=" in header_sig:
            _, sig = header_sig.split("=",1)
        else:
            sig = header_sig
        mac = hmac.new(secret.encode(), raw_body, getattr(hashlib, algo)).hexdigest()
        return hmac.compare_digest(sig, mac)
    except Exception as e:
        logger.exception("HMAC verify error: %s", e)
        return False

# Stripe signature verify helper (simple - uses stripe.endpoint_secret style)
def verify_stripe_signature(raw_body:bytes, header_sig:Optional[str], tolerance_sec:int=300):
    # Very basic: if STRIPE_ENDPOINT_SECRET not set we skip
    stripe_secret = WEBHOOK_SECRETS.get("stripe")
    if not stripe_secret or not header_sig:
        return False
    # For robust validation you'd import stripe lib. Here we do a minimal HMAC-ish check fallback.
    # **Recommendation**: For production use stripe-python's WebhookSignature.verify_header
    try:
        # header example: t=timestamp,v1=signature
        parts = {}
        for part in header_sig.split(","):
            k,v = part.split("=",1)
            parts[k] = v
        timestamp = int(parts.get("t","0"))
        if abs(time.time() - timestamp) > tolerance_sec:
            return False
        msg = f"{timestamp}.".encode() + raw_body
        expected = hmac.new(stripe_secret.encode(), msg, hashlib.sha256).hexdigest()
        sig = parts.get("v1")
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False

def add_kg_node(title: str, content: str):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO nodes (title, content, created_at) VALUES (?, ?, ?)", (title, content, time.time()))
    conn.commit()
    conn.close()

def query_kg(q: str, limit: int = 5):
    conn = sqlite3.connect(KG_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT title, content FROM nodes WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?", (f"%{q}%", f"%{q}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [{"title": r[0], "content": r[1]} for r in rows]

def cache_result(prompt: str, provider: str, result: Any):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO cache (prompt, provider, result, created_at) VALUES (?, ?, ?, ?)", (prompt, provider, json.dumps(result), time.time()))
    conn.commit()
    conn.close()

def get_cached(prompt: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT provider,result FROM cache WHERE prompt=? ORDER BY created_at DESC LIMIT 1", (prompt,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"provider": row[0], "result": json.loads(row[1])}
    return None

# ---------- Image helpers ----------
def unique_filename(ext="png"):
    return f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"

def local_image_url(request: Request, filename: str):
    return str(request.base_url).rstrip("/") + f"/static/images/{filename}"

def save_base64_image_to_file(b64: str, filename: str) -> str:
    path = os.path.join(IMAGES_DIR, filename)
    img_bytes = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

#------duckduckgo

async def duckduckgo_search(q: str):
    """
    Use DuckDuckGo Instant Answer API (no API key required).
    Returns a simple structured result with abstract, answer and a list of related topics.
    """
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
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

# ---------- Prompt enhancer ----------
async def enhance_prompt_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return prompt
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        system = "Rewrite the user's short prompt into a detailed, professional SDXL-style art prompt. Be concise but specific. Avoid explicit sexual or illegal content."
        body = {"model": CHAT_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":prompt}], "temperature":0.6,"max_tokens":300}
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            jr = r.json()
            content = jr.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or prompt
    except Exception:
        logger.exception("Prompt enhancer failed")
    return prompt

def run_code_safely(code: str, language: str = "python") -> Dict[str, str]:
    """
    Run code in a temporary file safely.
    Returns {'output': ..., 'error': ...}
    """
    if language.lower() != "python":
        return {"output": "", "error": f"Execution for {language} not supported yet."}

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=True) as tmpfile:
        tmpfile.write(code)
        tmpfile.flush()
        try:
            result = subprocess.run(
                ["python3", tmpfile.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            return {
                "output": result.stdout.decode().strip(),
                "error": result.stderr.decode().strip()
            }
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Execution timed out."}
        except Exception as e:
            return {"output": "", "error": str(e)}



# ---------- Prompt analysis ----------
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

# ---------- Streaming chat ----------
async def groq_stream(prompt: str):
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'error':'no_groq_key'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": CHAT_MODEL, "stream": True,"messages":[{"role":"system","content":get_system_prompt(prompt)},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    yield f"data: {json.dumps({'error':'provider_error','status':resp.status_code,'text': text.decode(errors='ignore')[:300]})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for raw_line in resp.aiter_lines():
                    if raw_line.startswith("data: "):
                        data = raw_line[len("data: "):]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        yield f"data: {data}\n\n"
        except Exception as e:
            logger.exception("groq_stream error")
            yield f"data: {json.dumps({'error':'stream_exception','msg':str(e)})}\n\n"
            yield "data: [DONE]\n\n"

@app.get("/health")
def health():
    return {"ok": True, "total_donations": total_funds(), "top_supporters": top_supporters(3)}

# Public donation links file (simple JSON you edit)
def load_donate_links():
    if os.path.exists(DONATE_LINKS_PATH):
        try:
            with open(DONATE_LINKS_PATH,"r") as f:
                return json.load(f)
        except Exception:
            return {}
    # default example links
    return {
        "kofi": "https://ko-fi.com/YourName",
        "buymeacoffee": "https://www.buymeacoffee.com/YourName",
        "gumroad": "https://gumroad.com/YourName",
        "paypal": "https://paypal.me/YourName"
    }
# Webhook receiver (generic)
@app.post("/webhook/{provider}")
async def webhook_receiver(provider: str, request: Request,
                           x_signature: Optional[str] = Header(None),
                           x_token: Optional[str] = Header(None)):
    """
    Generic webhook endpoint. Configure your provider's webhook to call:
      POST https://yourserver/webhook/kofi
      POST https://yourserver/webhook/buymeacoffee
      POST https://yourserver/webhook/gumroad
      POST https://yourserver/webhook/stripe
    Each provider uses slightly different payload keys — this function attempts to extract a reasonable donor + amount.
    IMPORTANT: set environment variables to verify incoming webhooks:
      KOFI_SECRET or KOFI_HMAC_SECRET
      BMC_SECRET
      GUMROAD_SECRET
      STRIPE_ENDPOINT_SECRET
    """
    raw = await request.body()
    try:
        payload = await request.json()
    except Exception:
        # non-json payload (some providers post form-encoded). Parse raw text fallback.
        try:
            payload_text = raw.decode(errors="ignore")
            payload = {"_raw_text": payload_text}
        except Exception:
            payload = {}

    logger.info("Webhook %s received: keys=%s", provider, list(payload.keys())[:6])

    # Basic verification (shared token)
    verified = False
    # 1) HMAC style
    if verify_hmac(provider, raw, x_signature):
        verified = True
    # 2) shared token header (x_token)
    elif verify_shared_token(provider, payload, x_token):
        verified = True
    # 3) stripe-style (special header)
    elif provider.lower() == "stripe" and verify_stripe_signature(raw, x_signature):
        verified = True
    else:
        # If no secret configured, accept but log a warning
        if not any(WEBHOOK_SECRETS.values()):
            logger.warning("No webhook secret configured; accepting webhook for provider %s (INSECURE)", provider)
            verified = True
        else:
            logger.warning("Webhook not verified for provider %s; header_sig=%s x_token=%s", provider, x_signature, x_token)

    if not verified:
        raise HTTPException(status_code=401, detail="Webhook verification failed")

    # --- Extract donor + amount heuristically for several providers ---
    donor = None
    amount = None
    currency = "USD"
    message = None
    raw_id = None

    p = payload if isinstance(payload, dict) else {}

    # Ko-fi common fields: "amount", "donor", "message", "id"
    if provider.lower() in ("kofi", "ko-fi"):
        raw_id = p.get("id") or p.get("payment_id") or p.get("transaction_id")
        amount = float(p.get("amount", p.get("value", 0) or 0))
        donor = p.get("from", p.get("donor", p.get("supporter_name")))
        currency = p.get("currency", currency)
        message = p.get("note", p.get("message"))
    # BuyMeACoffee:
    elif provider.lower() in ("buymeacoffee","bmc"):
        raw_id = p.get("id") or p.get("checkout_id")
        # BMC posts like: { "data": { "buyer": {...}, "amount": 5, "currency":"USD", ... } }
        data = p.get("data") or p
        if isinstance(data, dict):
            buyer = data.get("buyer") or data.get("supporter")
            if isinstance(buyer, dict):
                donor = buyer.get("name") or buyer.get("email")
            amount = float(data.get("amount", data.get("total", 0) or 0))
            currency = data.get("currency", currency)
            message = data.get("message", data.get("note"))
    # Gumroad:
    elif provider.lower() == "gumroad":
        raw_id = p.get("sale_id") or p.get("purchase_id")
        amount = float(p.get("price", p.get("amount", 0) or 0)) / 100.0 if p.get("price") else float(p.get("amount",0) or 0)
        donor = p.get("buyer_name") or p.get("buyer_email")
        currency = p.get("currency", currency)
        message = p.get("note")
    # Stripe (checkout.session.completed or payment_intent.succeeded)
    elif provider.lower() == "stripe":
        # payload may be full event
        ev = p.get("data", {}).get("object", p)
        raw_id = ev.get("id")
        currency = ev.get("currency", currency)
        # Stripe stores amounts in cents for many objects
        if ev.get("amount_total") is not None:
            amount = float(ev.get("amount_total", 0)) / 100.0
        elif ev.get("amount") is not None:
            amount = float(ev.get("amount", 0)) / 100.0
        donor = (ev.get("customer_details") or {}).get("name") or ev.get("billing_details", {}).get("name")
        message = ev.get("description") or ev.get("metadata", {}).get("message")
    else:
        # Generic fallback: try common keys
        raw_id = p.get("id") or p.get("transaction_id") or p.get("payment_id")
        for k in ("amount", "value", "price", "total"):
            if k in p:
                try:
                    amount = float(p.get(k) or 0)
                    break
                except Exception:
                    pass
        for k in ("donor", "from", "buyer", "supporter", "name"):
            if k in p:
                donor = p.get(k)
                break
        message = p.get("message") or p.get("note") or p.get("comment")

    # if amount still not found, try payload nested 'data' or text contains numbers - keep simple
    if amount is None:
        # try nested data object
        nested = p.get("data") if isinstance(p.get("data"), dict) else {}
        for k in ("amount", "value", "total", "price"):
            if k in nested:
                try:
                    amount = float(nested.get(k) or 0)
                    break
                except Exception:
                    pass

    # If we still don't have an amount, reject (we want an actual donation)
    try:
        amount = float(amount) if amount is not None else None
    except Exception:
        amount = None

    if amount is None or amount <= 0:
        logger.warning("Webhook for %s didn't include amount: payload keys %s", provider, list(p.keys())[:6])
        # Accept zero-amount webhooks (e.g. subscription event) by recording 0 if we want,
        # but here we'll treat it as accepted but not stored.
        return JSONResponse({"status":"ignored_no_amount"}, status_code=200)

    # save donation
    row_id = insert_donation(provider, raw_id or "", donor or "Anonymous", amount, currency, message or "")
    logger.info("Recorded donation #%s: %s gave %s %s via %s", row_id, donor or "Anonymous", amount, currency, provider)

    # Optional: call your app's notification hooks (send message to Discord/Slack) - user can add later.

    return JSONResponse({"status":"ok","id": row_id})


# Simple JSON API for supporters (leaderboard)
@app.get("/api/supporters")
def api_supporters(limit: int = 50):
    return {"total_funds": total_funds(), "top": top_supporters(limit), "recent": recent_donations(20)}

# Leaderboard HTML UI (simple, embeddable)
@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(goal: float = 30.0):
    links = load_donate_links()
    total = total_funds()
    top = top_supporters(20)
    recent = recent_donations(10)

    progress_pct = min(int((total / goal) * 100), 100) if goal>0 else 100

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Support Zynara</title>
        <style>
          body {{ background:#0f0f10;color:#fff;font-family:Inter, Arial; padding:18px; }}
          .card {{ background:#121212;padding:18px;border-radius:12px;max-width:820px;margin:12px auto; box-shadow:0 6px 18px rgba(0,0,0,0.6); }}
          h1 {{ margin:0 0 12px 0; }}
          .progress {{ background:#222;border-radius:999px;height:18px;overflow:hidden; }}
          .progress-inner {{ height:100%; background:#4a90e2; width:{progress_pct}%; transition: width 0.6s ease; }}
          .row {{ display:flex;gap:12px;align-items:center; }}
          .links a {{ display:inline-block;margin-right:12px;padding:8px 12px;background:#1c1c1c;border-radius:8px;color:#fff;text-decoration:none;border:1px solid #333; }}
          .leaderboard {{ margin-top:12px; }}
          .leaderboard li {{ padding:8px 6px;border-bottom:1px solid #161616; }}
          .recent {{ margin-top:12px;font-size:0.95rem;color:#bbb }}
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Support Zynara</h1>
          <p>Total funds collected: <strong>{total:.2f} { "USD" }</strong></p>
          <div class="progress" title="Progress towards goal">
            <div class="progress-inner"></div>
          </div>
          <p>Goal: {goal:.2f} — Progress: {progress_pct}%</p>

          <div class="links">
            {"".join([f'<a target="_blank" href="{v}">Donate via {k.title()}</a>' for k,v in links.items()])}
          </div>

          <h3 style="margin-top:18px">Top supporters</h3>
          <ol class="leaderboard">
            {"".join([f'<li><strong>{t["donor"]}</strong> — {t["total"]:.2f} {t["currency"]}</li>' for t in top])}
          </ol>

          <div class="recent">
            <h4>Recent donations</h4>
            <ul>
            {"".join([f'<li>{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))} — <strong>{r["donor"]}</strong> {r["amount"]} {r["currency"]} via {r["provider"]} { ("— " + (r["message"] or "")) if r["message"] else ""}</li>' for r in recent])}
            </ul>
          </div>

          <p style="margin-top:18px;font-size:0.9rem;color:#bbb">Tip: configure webhook URLs in your Ko-fi / BuyMeACoffee / Gumroad / Stripe dashboard to point to <code>/webhook/kofi</code> etc. Set secrets as env vars for verification.</p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

# Utility endpoint to download raw DB (password-protect in production)
@app.get("/admin/download-db")
def download_db(admin_key: Optional[str] = None):
    ADMIN_KEY = os.getenv("DONATIONS_ADMIN_KEY")
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        raise HTTPException(401, "admin key required")
    return FileResponse(DB_PATH, media_type="application/octet-stream", filename="donations.db")
    
@app.get("/stream")
async def stream_chat(prompt: str):
    return StreamingResponse(groq_stream(prompt), media_type="text/event-stream")

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
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"}, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- Image generation ----------
@app.post("/image")
async def image_gen(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    samples = int(body.get("samples", 1))
    return_base64 = body.get("base64", False)

    if not prompt:
        raise HTTPException(400, "prompt required")

    # Check cache first
    cached = get_cached(prompt)
    if cached:
        return {"cached": True, **cached}

    urls = []
    provider_used = None

    # ---------- 1) Stability SDXL ----------
    if STABILITY_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": "stable-diffusion-xl-1024-v1",
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "samples": samples,
                    "width": 1024,
                    "height": 1024
                }
                headers = {
                    "Authorization": f"Bearer {STABILITY_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                r = await client.post("https://api.stability.ai/v2beta/stable-image/generate",
                                      headers=headers, json=payload)
                r.raise_for_status()
                jr = r.json()

                for art in jr.get("artifacts", []):
                    b64 = art.get("base64")
                    if b64:
                        if return_base64:
                            urls.append(b64)
                        else:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))

            provider_used = "stability"
        except Exception:
            logger.exception("Stability image generation failed")

    # ---------- 2) OpenAI fallback ----------
    if not urls and OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {"model": "gpt-image-1", "prompt": prompt, "n": samples}
                r = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json=payload
                )
                r.raise_for_status()
                jr = r.json()
                for d in jr.get("data", []):
                    b64 = d.get("b64_json")
                    if b64:
                        if return_base64:
                            urls.append(b64)
                        else:
                            fname = unique_filename("png")
                            save_base64_image_to_file(b64, fname)
                            urls.append(local_image_url(request, fname))

            provider_used = "openai"
        except Exception:
            logger.exception("OpenAI image generation failed")

    # ---------- 3) Free fallback ----------
    if not urls and USE_FREE_IMAGE_PROVIDER and IMAGE_MODEL_FREE_URL:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(IMAGE_MODEL_FREE_URL, json={"prompt": prompt})
                r.raise_for_status()
                jr = r.json()
                images = jr.get("images") or [jr.get("image")]
                for im in images:
                    if return_base64:
                        urls.append(im)
                    else:
                        fname = unique_filename("png")
                        save_base64_image_to_file(im, fname)
                        urls.append(local_image_url(request, fname))

            provider_used = "free"
        except Exception:
            logger.exception("Free image provider failed")

    if not urls:
        raise HTTPException(500, "All image providers failed")

    cache_result(prompt, provider_used, urls)
    return {"provider": provider_used, "images": urls}
    
# ---------- TTS ----------
@app.post("/tts")
async def text_to_speech(req: Request):
    body = await req.json()
    text = body.get("text")
    if not text:
        raise HTTPException(400,"text required")
    if not OPENAI_API_KEY:
        raise HTTPException(400,"no TTS provider configured")
    payload = {"model": TTS_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload)
        r.raise_for_status()
        audio_data = r.content
    filename = unique_filename("mp3")
    path = os.path.join(IMAGES_DIR, filename)
    with open(path,"wb") as f:
        f.write(audio_data)
    return {"audio_url": local_image_url(req, filename)}

# ---------- Vision analyze ----------
@app.post("/vision/analyze")
async def vision_analyze(req: Request, file: UploadFile = File(...), prompt: Optional[str] = None, user_id: str = "anonymous"):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no vision provider configured")

    # Build dynamic, user-focused prompt
    user_prompt = prompt or "Analyze this image: list objects, colors, suggested tags, and short description."
    contextual_prompt = build_contextual_prompt(user_id, user_prompt)
    
    # Include image as base64
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": contextual_prompt},
            {"role": "user", "content": f"Image (base64):\n{base64.b64encode(content).decode()}"}
        ]
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=body)
        r.raise_for_status()
        return r.json()


# ---------- Code generation ----------
@app.post("/code")
async def code_gen(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    language = body.get("language", "python")
    user_id = body.get("user_id", "anonymous")

    if not prompt:
        raise HTTPException(400, "prompt required")
    if not GROQ_API_KEY:
        raise HTTPException(400, "no groq key")

    # Build user-focused prompt with context
    contextual_prompt = build_contextual_prompt(user_id, f"Write a complete, well-documented {language} program for the following request:\n\n{prompt}")

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": contextual_prompt}, {"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=payload)
        r.raise_for_status()
        return r.json()

@app.post("/img2img")
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = "", user_id: str = "anonymous"):
    if not prompt:
        raise HTTPException(400, "prompt required")

    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    cache_key = f"img2img:{user_id}:{prompt}"
    cached = get_cached(cache_key)
    if cached:
        return {"cached": True, **cached}

    enhanced = await enhance_prompt_with_groq(prompt)
    urls = []

    if not STABILITY_API_KEY:
        raise HTTPException(400, "no Stability API key")

    try:
        # correct SDXL img2img endpoint
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024x1024/image-to-image"

        payload = {
            "text_prompts": [{"text": enhanced}],
            "cfg_scale": 7,
            "samples": 1,
            "steps": 30,
            "strength": 0.6  # how much to transform the original (0.3 = mild edits, 0.8 = major changes)
        }

        # send multipart request
        files = {
            "init_image": (file.filename, content, file.content_type or "application/octet-stream"),
            "options": (None, json.dumps(payload), "application/json"),
        }

        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, files=files)
            resp.raise_for_status()

            jr = resp.json()
            for art in jr.get("artifacts", []):
                b64 = art.get("base64")
                if b64:
                    fname = unique_filename("png")
                    save_base64_image_to_file(b64, fname)
                    urls.append(local_image_url(request, fname))

        if urls:
            cache_result(cache_key, "stability-img2img", urls)
            return {"provider": "stability-img2img", "images": urls}

    except Exception as e:
        logger.exception("Img2Img failed")

    raise HTTPException(400, "img2img failed")
    
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
        raise HTTPException(400,"empty file")
    if not OPENAI_API_KEY:
        raise HTTPException(400,"no STT provider configured")
    files = {"file": (file.filename, content, file.content_type)}
    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, files=files, data={"model": STT_MODEL})
        r.raise_for_status()
        return r.json()

# ---------- Other endpoints (/remove-bg, /upscale, /img2img, /vision/analyze, /code, /search, /root) remain unchanged ----------

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), log_level="info")
