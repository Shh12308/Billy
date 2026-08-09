import os
import re
import json
import uuid
import asyncio
import logging
import hashlib
import zipfile
import tempfile
import base64
import io
import time
import secrets
import shutil
import mimetypes
from io import BytesIO
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
import cv2
import numpy as np
from PIL import Image
from fastapi import (
    FastAPI, Request, Response, HTTPException, UploadFile, File,
    Form, Query, Header, Cookie, Depends
)
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# Try optional PDF support
try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

# Try optional moviepy (for watermarking)
try:
    from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
    _HAS_MOVIEPY = True
except ImportError:
    _HAS_MOVIEPY = False

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeloXAi")

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip() or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LOGO_URL = os.getenv("LOGO_URL", "https://heloxai.xyz/logo.png")

# Replicate (BIGGER MODELS - image, video, music)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_IMAGE_MODEL = os.getenv("REPLICATE_IMAGE_MODEL", "black-forest-labs/flux-1.1-pro")
REPLICATE_IMAGE_VERSION = os.getenv("REPLICATE_IMAGE_VERSION", "c8cd8cb5a5d4f9c4c0f9c4c0f9c4c0f9c4c0f9c4")  # override if needed
REPLICATE_VIDEO_MODEL = os.getenv("REPLICATE_VIDEO_MODEL", "wavespeed/wan-2.1-14b-i2v")  # bigger than fal wan
REPLICATE_MUSIC_MODEL = os.getenv("REPLICATE_MUSIC_MODEL", "meta/musicgen:b05b1dff1d8c6dc63d14b0cdb405cb92")
REPLICATE_MAX_WAIT = int(os.getenv("REPLICATE_MAX_WAIT", "600"))  # 10 min
REPLICATE_POLL_INTERVAL = float(os.getenv("REPLICATE_POLL_INTERVAL", "3.0"))

# File handling
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ZIP_SIZE = 100 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_EXTRACTED_SIZE = 200 * 1024 * 1024
MAX_TEXT_LENGTH = 380000
MAX_IMAGE_SIZE = 20 * 1024 * 1024

# Auth
SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

# Rate limiting
IP_RATE_LIMIT = 30
IP_RATE_WINDOW = 60
ENDPOINT_LIMITS = {
    "/ask/universal": {"limit": 20, "window": 60},
    "/tts":           {"limit": 10, "window": 60},
    "/stt":           {"limit": 10, "window": 60},
    "/analysis":      {"limit": 15, "window": 60},
    "/generate/image": {"limit": 8,  "window": 60},
    "/generate/video": {"limit": 4,  "window": 60},
    "/generate/music": {"limit": 6,  "window": 60},
}
_RATE_CLEANUP_INTERVAL = 300
_last_rate_cleanup = 0
_rate_store: Dict[str, List[float]] = {}

GROQ_MAX_RETRIES = 3

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

app = FastAPI(
    title="HeloXAi Unified",
    description="Multi-modal AI: GPT-4o chat + Replicate Flux/Wan/MusicGen + Groq Whisper + Tavily research",
    version="5.0.0"
)

# =========================
# MODEL CONFIGURATION (BIGGER MODELS)
# =========================
OPENAI_CHAT_MODEL = "gpt-4o"            # bigger than gpt-4o-mini
OPENAI_VISION_MODEL = "gpt-4o"          # bigger vision
OPENAI_TTS_MODEL = "tts-1-hd"           # higher-quality TTS
GROQ_STT_MODEL = "whisper-large-v3"     # fast, free STT
GROQ_MEMORY_MODEL = "llama-3.1-8b-instant"  # cheap memory consolidation
GROQ_FALLBACK_CHAT = "llama-3.3-70b-versatile"

MODEL_ROUTING = {
    "helox":    {"chat": OPENAI_CHAT_MODEL,        "vision": OPENAI_VISION_MODEL, "provider": "openai"},
    "chatgpt":  {"chat": "gpt-4o",                "vision": "gpt-4o",            "provider": "openai"},
    "chatz":    {"chat": OPENAI_CHAT_MODEL,        "vision": OPENAI_VISION_MODEL, "provider": "openai"},
    "groq":     {"chat": GROQ_FALLBACK_CHAT,       "vision": OPENAI_VISION_MODEL, "provider": "groq"},
}

# =========================
# CORS
# =========================
service_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SERVICE_URL") or "https://heloxai2.onrender.com"
frontend_url = os.getenv("FRONTEND_URL", service_url)
allowed_origins = list({
    frontend_url, service_url,
    "https://heloxai.xyz", "https://www.heloxai.xyz",
    "capacitor://localhost",
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# =========================
# DATABASE
# =========================
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
active_streams: Dict[str, asyncio.Task] = {}

_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300
_session_cache_last_cleanup = time.time()

_conv_creation_locks: Dict[str, asyncio.Lock] = {}

_new_user_lock = asyncio.Lock()
_pending_new_user_id: Optional[str] = None
_new_user_created_event = asyncio.Event()

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
}

def _get_conv_lock(conv_id: str) -> asyncio.Lock:
    if conv_id not in _conv_creation_locks:
        _conv_creation_locks[conv_id] = asyncio.Lock()
    return _conv_creation_locks[conv_id]


# =========================
# RATE LIMITING
# =========================
def _get_rate_key(client_ip: str, path: str) -> str:
    for ep in ENDPOINT_LIMITS:
        if path.startswith(ep):
            return f"{client_ip}:{ep}"
    return f"{client_ip}:__global__"

def _get_limits_for_key(key: str) -> Tuple[int, int]:
    for ep, cfg in ENDPOINT_LIMITS.items():
        if key.endswith(ep):
            return cfg["limit"], cfg["window"]
    return IP_RATE_LIMIT, IP_RATE_WINDOW

def _cleanup_rate_store(now: float):
    global _last_rate_cleanup
    if now - _last_rate_cleanup < _RATE_CLEANUP_INTERVAL:
        return
    _last_rate_cleanup = now
    stale = []
    for k, ts in _rate_store.items():
        _, w = _get_limits_for_key(k)
        f = [t for t in ts if now - t < w]
        if f:
            _rate_store[k] = f
        else:
            stale.append(k)
    for k in stale:
        del _rate_store[k]

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/" or request.method == "OPTIONS":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _cleanup_rate_store(now)
    key = _get_rate_key(client_ip, request.url.path)
    limit, window = _get_limits_for_key(key)
    if key not in _rate_store:
        _rate_store[key] = []
    _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
    cnt = len(_rate_store[key])
    if cnt >= limit:
        oldest = _rate_store[key][0]
        reset_at = int(oldest + window)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down.",
                     "limit": limit, "window": window, "reset_at": reset_at},
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
                "Retry-After": str(max(1, reset_at - int(now))),
            }
        )
    _rate_store[key].append(now)
    remaining = limit - cnt - 1
    reset_at = int(_rate_store[key][0] + window)
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Reset"] = str(reset_at)
    return response


# =========================
# FILE TYPE DEFINITIONS
# =========================
class FileCategory(Enum):
    CODE = "code"; DOCUMENT = "document"; DATA = "data"; IMAGE = "image"
    AUDIO = "audio"; VIDEO = "video"; ARCHIVE = "archive"
    CONFIG = "config"; BINARY = "binary"; UNKNOWN = "unknown"

CODE_EXTENSIONS = {
    '.py','.pyw','.pyx','.js','.jsx','.mjs','.ts','.tsx','.html','.htm','.css',
    '.scss','.sass','.less','.vue','.svelte','.astro','.java','.kt','.scala',
    '.groovy','.clj','.hs','.c','.h','.cpp','.hpp','.cc','.cs','.go','.rs',
    '.php','.rb','.swift','.dart','.sh','.bash','.zsh','.ps1','.lua','.pl',
    '.r','.sql','.json','.yaml','.yml','.toml','.ini','.cfg','.env','.xml',
    '.md','.rst','.tex','.dockerfile','.makefile','.cmake','.proto','.graphql',
    '.tf','.hcl','.sol','.move'
}
DOCUMENT_EXTENSIONS = {'.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.odt','.ods','.odp','.rtf','.txt','.log','.csv'}
DATA_EXTENSIONS = {'.csv','.tsv','.json','.xml','.yaml','.yml','.parquet','.arrow','.hdf5','.h5','.pickle','.pkl','.npy','.npz'}
IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.webp','.svg','.bmp','.ico','.tiff','.tif','.avif','.heic'}
AUDIO_EXTENSIONS = {'.mp3','.wav','.ogg','.flac','.aac','.m4a','.wma','.opus','.aiff','.ape'}
VIDEO_EXTENSIONS = {'.mp4','.webm','.avi','.mov','.mkv','.flv','.wmv','.m4v','.ogv','.3gp'}
ARCHIVE_EXTENSIONS = {'.zip','.tar','.gz','.tgz','.bz2','.xz','.7z','.rar','.zst','.lz4'}
CONFIG_EXTENSIONS = {'.json','.yaml','.yml','.toml','.ini','.cfg','.conf','.env','.properties','.xml','.editorconfig'}

def get_file_category(filename: str) -> FileCategory:
    if not filename: return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in CODE_EXTENSIONS: return FileCategory.CODE
    if ext in DOCUMENT_EXTENSIONS: return FileCategory.DOCUMENT
    if ext in DATA_EXTENSIONS: return FileCategory.DATA
    if ext in IMAGE_EXTENSIONS: return FileCategory.IMAGE
    if ext in AUDIO_EXTENSIONS: return FileCategory.AUDIO
    if ext in VIDEO_EXTENSIONS: return FileCategory.VIDEO
    if ext in ARCHIVE_EXTENSIONS: return FileCategory.ARCHIVE
    if ext in CONFIG_EXTENSIONS: return FileCategory.CONFIG
    return FileCategory.UNKNOWN

def get_file_language(filename: str) -> Optional[str]:
    m = {
        '.py':'python','.js':'javascript','.jsx':'javascript','.ts':'typescript','.tsx':'typescript',
        '.html':'html','.css':'css','.vue':'vue','.svelte':'svelte','.java':'java','.kt':'kotlin',
        '.c':'c','.cpp':'cpp','.cs':'csharp','.go':'go','.rs':'rust','.php':'php','.rb':'ruby',
        '.swift':'swift','.dart':'dart','.sh':'bash','.bash':'bash','.ps1':'powershell',
        '.lua':'lua','.pl':'perl','.r':'r','.sql':'sql','.json':'json','.xml':'xml',
        '.yaml':'yaml','.yml':'yaml','.toml':'toml','.md':'markdown','.tex':'latex',
        '.dockerfile':'dockerfile','.graphql':'graphql','.tf':'hcl','.sol':'solidity',
    }
    return m.get(Path(filename).suffix.lower())

def get_language_from_extension(filename: str) -> str:
    return get_file_language(filename) or "Unknown"

def is_binary_file(filename: str, content: bytes = None) -> bool:
    ext = Path(filename).suffix.lower()
    if ext in (IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | ARCHIVE_EXTENSIONS):
        return True
    if ext in {'.exe','.dll','.so','.dylib','.bin','.dat','.pyc','.pyo','.class','.o','.obj','.a','.lib',
               '.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.sqlite','.db','.sqlite3',
               '.woff','.woff2','.ttf','.otf','.eot','.pak','.bundle'}:
        return True
    if content and b'\x00' in content[:8192]:
        return True
    return False

def format_file_size(size_bytes: int) -> str:
    for unit in ['B','KB','MB','GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def _is_image_mime(mime: str) -> bool:
    return bool(mime) and mime.startswith("image/")


# =========================
# FILE EXTRACTOR
# =========================
class FileExtractionResult:
    def __init__(self, content: str, files: List[Dict[str, Any]] = None,
                 metadata: Dict[str, Any] = None, truncated: bool = False, original_size: int = 0):
        self.content = content
        self.files = files or []
        self.metadata = metadata or {}
        self.truncated = truncated
        self.original_size = original_size

def extract_text_with_fallback(content: bytes, max_length: int) -> Tuple[str, bool]:
    for enc in ['utf-8','utf-8-sig','latin-1','cp1252','iso-8859-1','ascii']:
        try:
            text = content.decode(enc, errors='strict' if enc != 'latin-1' else 'ignore')
            truncated = len(text) > max_length
            if truncated:
                text = text[:max_length] + "\n\n[... Content truncated ...]"
            return text, truncated
        except (UnicodeDecodeError, LookupError):
            continue
    text = content.decode('utf-8', errors='replace')
    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length] + "\n\n[... Content truncated ...]"
    return text, truncated

async def extract_pdf_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    if not _HAS_PYPDF:
        return FileExtractionResult(
            content=f"[PDF file: {filename} ({format_file_size(len(content))}) - PyPDF2 not installed]",
            metadata=metadata, original_size=len(content)
        )
    try:
        reader = PdfReader(BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append(f"--- Page {i+1} ---\n{page.extract_text() or ''}")
        full = "\n\n".join(pages)
        metadata["page_count"] = len(reader.pages)
        truncated = len(full) > max_length
        if truncated:
            full = full[:max_length] + "\n\n[... Content truncated ...]"
        return FileExtractionResult(content=full, metadata=metadata, truncated=truncated, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(
            content=f"[Error extracting PDF {filename}: {e}]",
            metadata={**metadata, "error": str(e)}, original_size=len(content)
        )

async def extract_zip_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    extracted_files, all_parts = [], []
    total_extracted = 0
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            if len(zf.namelist()) > MAX_ZIP_ENTRIES:
                return FileExtractionResult(
                    content=f"[ZIP: too many entries ({len(zf.namelist())})]",
                    metadata=metadata, original_size=len(content))
            for name in sorted(zf.namelist()):
                if name.endswith('/') or name.startswith('__MACOSX') or name.startswith('.'):
                    continue
                try:
                    info = zf.getinfo(name)
                    if info.file_size > MAX_FILE_SIZE:
                        extracted_files.append({"name": name, "status": "skipped"})
                        continue
                    if total_extracted + info.file_size > MAX_EXTRACTED_SIZE:
                        extracted_files.append({"name": name, "status": "skipped"})
                        continue
                    entry = zf.read(name)
                    total_extracted += len(entry)
                    if not is_binary_file(name, entry):
                        text, _ = extract_text_with_fallback(entry, max_length)
                        if text.strip():
                            all_parts.append(f"\n{'='*60}\nFile: {name}\n{'='*60}\n{text}")
                            extracted_files.append({"name": name, "size": len(entry), "status": "extracted"})
                        else:
                            extracted_files.append({"name": name, "status": "empty"})
                    else:
                        extracted_files.append({"name": name, "size": len(entry), "status": "binary"})
                except Exception as e:
                    extracted_files.append({"name": name, "status": "error", "error": str(e)})
        full = f"ZIP: {filename}\nEntries: {len(zf.namelist())}\n\n" + "".join(all_parts)
        metadata.update({"archive_type":"zip","entry_count":len(zf.namelist()),"files":extracted_files})
        truncated = len(full) > max_length
        if truncated:
            full = full[:max_length] + "\n\n[... truncated ...]"
        return FileExtractionResult(content=full, files=extracted_files, metadata=metadata,
                                    truncated=truncated, original_size=len(content))
    except zipfile.BadZipFile:
        return FileExtractionResult(content=f"[Bad ZIP: {filename}]", metadata=metadata, original_size=len(content))

async def extract_tar_content(content: bytes, filename: str, max_length: int, metadata: Dict) -> FileExtractionResult:
    import tarfile
    parts, files = [], []
    try:
        with tarfile.open(fileobj=BytesIO(content)) as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for m in members:
                if m.name.startswith('__MACOSX') or m.name.startswith('.'):
                    continue
                try:
                    f = tf.extractfile(m)
                    if not f: continue
                    c = f.read()
                    if not is_binary_file(m.name, c):
                        text, _ = extract_text_with_fallback(c, max_length)
                        if text.strip():
                            parts.append(f"\n{'='*60}\nFile: {m.name}\n{'='*60}\n{text}")
                            files.append({"name": m.name, "status": "extracted"})
                    else:
                        files.append({"name": m.name, "status": "binary"})
                except Exception as e:
                    files.append({"name": m.name, "status": "error", "error": str(e)})
        full = f"TAR: {filename}\n" + "".join(parts)
        metadata.update({"archive_type":"tar","files":files})
        truncated = len(full) > max_length
        if truncated:
            full = full[:max_length] + "\n\n[... truncated ...]"
        return FileExtractionResult(content=full, files=files, metadata=metadata,
                                    truncated=truncated, original_size=len(content))
    except Exception as e:
        return FileExtractionResult(content=f"[TAR error: {e}]", metadata=metadata, original_size=len(content))

async def extract_file_content(content: bytes, filename: str, max_length: int = MAX_TEXT_LENGTH) -> FileExtractionResult:
    original_size = len(content)
    category = get_file_category(filename)
    metadata = {
        "filename": filename, "category": category.value,
        "size": original_size, "size_formatted": format_file_size(original_size),
        "language": get_file_language(filename),
    }
    try:
        if category == FileCategory.ARCHIVE:
            ext = Path(filename).suffix.lower()
            if ext == '.zip':
                return await extract_zip_content(content, filename, max_length, metadata)
            if ext in ('.tar','.gz','.tgz','.bz2','.xz'):
                return await extract_tar_content(content, filename, max_length, metadata)
            return FileExtractionResult(content=f"[Unsupported archive: {filename}]", metadata=metadata, original_size=original_size)
        if category == FileCategory.IMAGE:
            return FileExtractionResult(content=f"[Image: {filename} - use image analysis]", metadata=metadata, original_size=original_size)
        if category in (FileCategory.AUDIO, FileCategory.VIDEO):
            return FileExtractionResult(content=f"[{category.value}: {filename}]", metadata=metadata, original_size=original_size)
        if filename.lower().endswith('.pdf'):
            return await extract_pdf_content(content, filename, max_length, metadata)
        if is_binary_file(filename, content):
            return FileExtractionResult(content=f"[Binary: {filename}]", metadata=metadata, original_size=original_size)
        text, truncated = extract_text_with_fallback(content, max_length)
        metadata["line_count"] = text.count('\n') + 1
        return FileExtractionResult(content=text, metadata=metadata, truncated=truncated, original_size=original_size)
    except Exception as e:
        return FileExtractionResult(content=f"[Error: {e}]", metadata={**metadata, "error": str(e)}, original_size=original_size)

async def extract_text_safe(content: bytes) -> str:
    for enc in ['utf-8','latin-1','cp1252']:
        try:
            return content.decode(enc, errors='ignore')[:MAX_TEXT_LENGTH]
        except Exception:
            continue
    return "[Binary or unreadable content]"


# =========================
# AUTH SYSTEM (PRODUCTION-GRADE)
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
FINGERPRINT_COOKIE = "HeloxAI_FP"
BACKUP_COOKIE = "HeloxAI_ID"
DEVICE_COOKIE = "HeloxAI_Dev"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    base = {
        "max_age": SESSION_DURATION if remember else 24*60*60,
        "httponly": True, "secure": True, "samesite": "none", "path": "/"
    }
    d = os.getenv("COOKIE_DOMAIN")
    if d: base["domain"] = d
    return base

def generate_device_fingerprint(request: Request) -> str:
    real_ip = (
        request.headers.get("x-forwarded-for","").split(",")[0].strip()
        or request.headers.get("x-real-ip","")
        or (request.client.host if request.client else "")
    )
    fp = "|".join([
        request.headers.get("user-agent",""),
        request.headers.get("accept-language",""),
        request.headers.get("accept-encoding",""),
        request.headers.get("sec-ch-ua-platform",""),
        request.headers.get("sec-ch-ua-mobile",""),
        real_ip,
    ])
    return hashlib.sha256(fp.encode()).hexdigest()[:32]

def generate_session_token() -> str:
    return secrets.token_urlsafe(64)

def set_session_cookies(response: Response, user_id: str, fingerprint: str,
                        session_token: str, remember: bool = True):
    s = get_cookie_settings(remember)
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24*60*60)
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **s)
    response.set_cookie(key=FINGERPRINT_COOKIE, value=fingerprint, **s)
    response.set_cookie(key=BACKUP_COOKIE, value=user_id, **s)
    response.set_cookie(key=DEVICE_COOKIE, value=f"{fingerprint}_{user_id[:8]}", **s)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=session_token, **s)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **s)

def clear_session_cookies(response: Response):
    domain = os.getenv("COOKIE_DOMAIN")
    for c in [PRIMARY_COOKIE, FINGERPRINT_COOKIE, BACKUP_COOKIE, DEVICE_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE]:
        kw = {"key": c, "path": "/", "secure": True, "samesite": "none"}
        if domain: kw["domain"] = domain
        response.delete_cookie(**kw)

def is_session_expired(expiry_str: str) -> bool:
    try:
        return time.time() > int(expiry_str)
    except Exception:
        return True

def should_refresh_session(expiry_str: str) -> bool:
    try:
        return (int(expiry_str) - time.time()) < REFRESH_THRESHOLD
    except Exception:
        return True

async def ensure_user_exists(user_id: str) -> bool:
    try:
        await _execute_supabase_with_retry(
            supabase.table("users").upsert(
                {"id": user_id, "created_at": datetime.now(timezone.utc).isoformat()},
                on_conflict="id"
            ),
            description="Ensure User"
        )
        return True
    except Exception as e:
        logger.error(f"ensure_user_exists failed: {e}")
        return False

async def validate_session_token(user_id: str, token: str) -> bool:
    try:
        if user_id in _session_cache and _session_cache[user_id].get("token") == token:
            if time.time() - _session_cache[user_id].get("time", 0) < _session_cache_ttl:
                return True
        result = await _execute_supabase_with_retry(
            supabase.table("user_sessions")
            .select("token, expires_at")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1),
            description="Validate Session"
        )
        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {"token": token, "time": time.time(),
                                        "expires_at": result.data[0].get("expires_at")}
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

async def create_user_session(user_id: str, fingerprint: str, remember: bool = True) -> Optional[str]:
    if not await ensure_user_exists(user_id):
        return None
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=SESSION_DURATION if remember else 24*60*60
    )
    try:
        await _execute_supabase_with_retry(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()), "user_id": user_id, "token": token,
                "fingerprint": fingerprint, "user_agent": "", "ip_address": "",
                "expires_at": expires_at.isoformat(), "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }),
            description="Create Session"
        )
        _session_cache[user_id] = {"token": token, "time": time.time(),
                                    "expires_at": expires_at.isoformat()}
        return token
    except Exception as e:
        logger.error(f"create_user_session failed: {e}")
        return None

async def cleanup_session_cache():
    global _session_cache_last_cleanup
    now = time.time()
    if now - _session_cache_last_cleanup < _session_cache_ttl:
        return
    _session_cache_last_cleanup = now
    expired = []
    for uid, data in _session_cache.items():
        ea = data.get("expires_at")
        if ea:
            try:
                if now > datetime.fromisoformat(ea).timestamp():
                    expired.append(uid)
            except Exception:
                expired.append(uid)
    for k in expired:
        _session_cache.pop(k, None)


# =========================
# SYSTEM PROMPTS
# =========================
BASE_SYSTEM_PROMPT = """You are HeloXAi, a powerful, multi-modal AI assistant.

**Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Coding:** Expert across all languages - writing, debugging, reviewing.
3. **Math:** Solve problems with step-by-step reasoning. Use LaTeX: $...$ inline, $$...$$ display.
4. **Research:** Real-time web search via Tavily. Cite sources as [1], [2] etc.
5. **Image Generation:** Flux 1.1 Pro (Replicate).
6. **Video Generation:** Wan 2.1 14B (Replicate).
7. **Music Generation:** MusicGen (Replicate).
8. **File Intelligence:** Read documents, code, archives, PDFs.

**Response Style:**
- Use Markdown (headers, bold, lists, tables, code blocks with language tags).
- Be concise but thorough. No walls of text.
- For code, always provide complete, runnable code — never placeholders.
- If you use web search, cite sources in a "Sources" section.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi"
"""

IMAGE_ANALYSIS_SYSTEM_PROMPT = """You are HeloXAi, an expert visual analyst.

Analyze the provided image thoroughly:
1. **Description:** What is shown (objects, scene, people, text).
2. **Details:** Colors, layout, style, composition, quality.
3. **Context:** Likely purpose, usage.
4. **Text:** Transcribe any readable text exactly.
5. **Issues:** Problems or anomalies.

If image contains code screenshots, read and explain. If diagrams/charts, describe data/trends.
Use Markdown formatting."""

CODE_ANALYSIS_SYSTEM_PROMPT = """You are HeloXAi, a senior software engineer and code reviewer.

1. **Overview:** What does this code do? Language and purpose?
2. **Architecture:** Structure and patterns.
3. **Quality Assessment:** Rate quality (1-10) with justification.
4. **Issues Found:**
   - Critical: Bugs, security vulnerabilities, crashes
   - Warnings: Bad practices, performance, maintainability
   - Suggestions: Improvements, modernizations, best practices
5. **Security Review:** Vulnerabilities (XSS, injection, auth).
6. **Performance:** Bottlenecks.
7. **Refactored Version:** Improved code with fixes applied.

Reference line numbers. Provide working improved code."""

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """You are HeloXAi, an expert document analyst.

1. **Summary:** 2-3 sentences.
2. **Key Points:** Bullet points of main ideas/facts.
3. **Structure:** Organization.
4. **Analysis:** Deep analysis of content, arguments, or data.
5. **Issues:** Errors, inconsistencies.
6. **Recommendations:** Next steps.

Use Markdown."""

FINANCE_SYSTEM_PROMPT = """You are HeloXAi, a financial analysis assistant.

1. **Always** search for current data before answering.
2. Provide specific numbers, percentages, dates.
3. Include relevant context.
4. **Disclaimer:** Always end with: "*Note: This is not financial advice. Do your own research.*"
5. Use tables for comparisons.
6. Cite sources."""

CREATOR_RESPONSE_INSTRUCTION = """IMPORTANT: The user is asking about your creator. Respond EXACTLY:
"I was constructed by GoldYLocks. You can find them on Twitter @HeloXAi"
No additional details."""

CREATOR_QUESTION_PATTERNS = [
    r'\b(who|whom)\b.*\b(made|created|built|developed|constructed|programmed|designed|founded|owns|runs)\b.*\b(you|this|helox|heloxai)\b',
    r'\b(your|the)\b.*(creator|developer|maker|builder|founder|owner|author)\b.*\b(is|are|who)\b',
    r'\bwho\s+is\s+behind\s+helox\b',
    r'\bwho\s+(made|created|built|developed|programmed|constructed|designed|owns|runs)\s+(you|helox)\b',
    r'\b(your|the)\s+(creator|developer|maker|builder|founder|owner)\b',
    r'\btell\s+me\s+about\s+your\s+(creator|developer|maker|builder|founder)\b',
    r'\bhow\s+were\s+you\s+(made|created|built|developed|born)\b',
]
COMPILED_CREATOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREATOR_QUESTION_PATTERNS]

def is_creator_question(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_CREATOR_PATTERNS)

def get_system_prompt(user_prompt: str, mode: Optional[str] = None) -> str:
    if mode == "finance":
        return FINANCE_SYSTEM_PROMPT
    if is_creator_question(user_prompt):
        return BASE_SYSTEM_PROMPT + "\n\n" + CREATOR_RESPONSE_INSTRUCTION
    return BASE_SYSTEM_PROMPT


# =========================
# ADVANCED INTENT DETECTION
# =========================
class IntentCategory(Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    MUSIC_GENERATION = "music_generation"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    DOCUMENT_CREATION = "document_creation"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"
    WEB_DEVELOPMENT = "web_development"
    API_DEVELOPMENT = "api_development"
    DATABASE = "database"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPLANATION = "explanation"
    CREATIVE_WRITING = "creative_writing"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    CONVERSATION = "conversation"

@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    sub_intents: List[IntentCategory] = field(default_factory=list)
    keywords_matched: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "sub_intents": [i.value for i in self.sub_intents],
            "keywords_matched": self.keywords_matched,
            "patterns_matched": self.patterns_matched
        }

class AdvancedIntentDetector:
    def __init__(self):
        self._compile_patterns()
        self._init_synonyms()
        self.negation_words = {
            "don't","dont","do not","doesn't","doesnt","does not",
            "didn't","didnt","did not","never","no","not","without",
            "skip","avoid","except","but not","ignore","rather than"
        }

    def _compile_patterns(self):
        self.patterns = {
            IntentCategory.VIDEO_GENERATION: [
                r'\b(generate|create|make|produce|render)\s+(a\s+)?(video|clip|movie|animation|motion\s+graphic)',
                r'\b(text\s+to\s+video|txt2vid)',
                r'\b(animate|animation)\s+(this|that|the|image|picture)',
                r'\b(video|clip|movie)\s+(of|showing|about|with)',
                r'\b(runway|pika|sora|kling|wan|wan2\.1)',
                r'\b(turn|convert)\s+(this|the|image)\s+(into|to)\s+(a\s+)?(video|animation)',
            ],
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render|paint|sketch|illustrate)\s+(a\s+|an\s+)?(image|picture|photo|drawing|illustration|artwork|painting|sketch|graphic|visual)',
                r'\b(image|picture|photo|drawing|illustration)\s+(of|showing|depicting|with|for|about)',
                r'\b(visualize|visualise)\s+(this|that|the|it)',
                r'\b(dall[eé]|midjourney|stable\s+diffusion|sdxl|flux|sd3)',
                r'\b(make\s+(me\s+)?(a\s+)?(visual|graphic|thumbnail|logo|icon|banner|poster))',
                r'\b(prompt\s+(for|to))\s+(generate|create|make)',
            ],
            IntentCategory.MUSIC_GENERATION: [
                r'\b(generate|create|make|compose|produce)\s+(a\s+|an\s+|some\s+)?(song|track|melody|beat|tune|instrumental|music)',
                r'\b(music|song|beat|melody|tune)\s+(for|about|like)',
                r'\b(write|compose)\s+(me\s+)?(a\s+)?(song|track|beat|melody)',
                r'\b(musicgen|suno|udio|bark)',
                r'\bmake\s+music\b',
                r'\bbackground\s+music\b',
            ],
            IntentCategory.AUDIO_GENERATION: [
                r'\b(generate|create|make|produce)\s+(a\s+)?(audio|sound|sound\s+effect|sfx|voice|speech|narration|voiceover)',
                r'\b(text\s+to\s+speech|tts|speech\s+to\s+text|stt)',
                r'\b(elevenlabs|bark)',
                r'\b(clone|replicate)\s+(a\s+)?voice',
            ],
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|generate|build|code|develop|implement)\s+(a\s+)?(\w+\s+)?(function|class|module|script|program|code|snippet|app|application|component)',
                r'\b(how\s+(to|can\s+i)\s+(write|create|implement|code|build))',
                r'\b(code\s+(for|that|this|to|which|example))',
                r'\b(convert\s+(this|to)\s+(code|python|javascript|java|c\+\+|rust|go|typescript))',
                r'\b(scaffold|boilerplate|template)\s+(for|a)',
                r'\b(wrapper|helper|utility)\s+(function|class|module)\s+(for|to)',
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|analyze|critique|evaluate|audit)\s+(this|my|the)\s+(code|function|class|script|implementation|pr)',
                r'\b(refactor|improve|optimize|clean\s+up)\s+(this|my|the)\s+(code|function|class)',
                r'\b(code\s+quality|technical\s+debt|code\s+smell)',
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve|troubleshoot|resolve)\s+(this|my|the|a)\s+(bug|error|issue|problem)',
                r'\b(why\s+(is|does|are|do)\s+(this|my|the|it)\s+(not\s+working|failing|breaking|erroring))',
                r'\b(error|exception|traceback|stack\s+trace)\s*[:\n]',
                r'\b(won\'t\s+work|doesn\'t\s+work|not\s+working|broken|failing)',
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|compute|solve|evaluate)\s+(this|the|a)\s*(equation|expression|formula|problem|integral|derivative)?',
                r'\b(math|mathematics|algebra|calculus|geometry|statistics|probability)\s*(problem|equation|question)?',
                r'\b(\d+[\.\d]*\s*[\+\-\*\/\^%\=]\s*[\.\d]*)',
                r'\b(integral|derivative|differentiat|integrat)\s*(of|the)?',
                r'\b(prove|proof)\s+(that|this|the)',
            ],
            IntentCategory.RESEARCH: [
                r'\b(research|find|search|look\s+up|investigate)\s+(about|on|for|into)',
                r'\b(academic|scholarly|peer[- ]?reviewed)\s*(source|paper|article|research|journal)?',
                r'\b(latest\s+news|current\s+events|what\s+is\s+happening)',
            ],
            IntentCategory.TRANSLATION: [
                r'\b(translate|translation)\s+(this|to|into|from)\s+(\w+)',
                r'\b(in|to|into)\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|urdu)',
                r'\b(how\s+(do\s+you|to)\s+say\s+.+\s+in\s+\w+)',
            ],
            IntentCategory.SUMMARIZATION: [
                r'\b(summarize|summary|summarise|tldr|tl;dr)\s+(this|the|it|that|for\s+me)',
                r'\b(key\s+(points|takeaways|highlights))\s*(from|of|in)?',
            ],
            IntentCategory.EXPLANATION: [
                r'\b(explain|explanation)\s+(to\s+me\s+)?',
                r'\b(what\s+(is|are|was|were|does|do|means|mean))\s+',
                r'\b(how\s+(does|do|did|can|would|should|to))\s+',
                r'\b(why\s+(is|does|do|are|did|can|would))\s+',
                r'\b(break\s+down|simplify|elaborate)\s+',
            ],
            IntentCategory.CREATIVE_WRITING: [
                r'\b(write|create|compose)\s+(a\s+)?(story|poem|poetry|novel|chapter|verse|lyrics|haiku|limerick)',
                r'\b(creative|fiction|fantasy|sci[- ]?fi|horror|romance|thriller|mystery)\s*(writing|story|tale)?',
                r'\b(narrative|plot|character|setting|dialogue)\s*(for|development|creation|arc)?',
            ],
            IntentCategory.WEB_DEVELOPMENT: [
                r'\b(create|build|develop|make)\s+(a\s+)?(website|web\s*page|web\s*app|landing\s+page|portfolio)',
                r'\b(html|css|react|vue|angular|next\.js|svelte|tailwind)\b',
                r'\b(frontend|backend|full[- ]stack)\s*(development|for|with|app)?',
            ],
            IntentCategory.API_DEVELOPMENT: [
                r'\b(create|build|develop|design|implement)\s+(a\s+)?(api|rest\s*api|graphql\s*api|endpoint|route)',
                r'\b(restful|rest|graphql|grpc|websocket)\s*(api|service|endpoint)?',
            ],
            IntentCategory.DATABASE: [
                r'\b(create|write|design)\s+(a\s+)?(database|schema|table|query|sql|migration)',
                r'\b(sql|mysql|postgres|mongodb|redis|sqlite)\s*(query|statement|command)?',
                r'\b(select|insert|update|delete)\s+(from|into|table)',
            ],
            IntentCategory.DATA_ANALYSIS: [
                r'\b(analyze|analysis|analyse)\s+(this|the|my|some)\s+(data|dataset|csv|excel|spreadsheet|json)',
                r'\b(statistics?|statistical)\s+(analysis|test|summary)',
                r'\b(correlation|regression|distribution|trend)\s+(analysis|of|in)',
                r'\b(clean|preprocess|prepare|wrangle)\s+(this|the)\s+(data|dataset)',
            ],
            IntentCategory.DATA_VISUALIZATION: [
                r'\b(create|make|generate|plot|chart|graph|visualize)\s+(a\s+)?(chart|graph|plot|visualization|diagram|dashboard)',
                r'\b(bar\s+chart|line\s+graph|scatter\s+plot|pie\s+chart|histogram|heatmap)',
                r'\b(matplotlib|seaborn|plotly|d3|chart\.js|ggplot|altair)',
            ],
            IntentCategory.DOCUMENT_CREATION: [
                r'\b(create|write|generate|draft|compose)\s+(a\s+)?(document|pdf|report|letter|email|memo|article|essay|paper|proposal|whitepaper)',
                r'\b(document|report|proposal|specification)\s+(for|about|on|regarding)',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))[\s!.?]*$',
                r'^(thank|thanks|thank\s+you|appreciate)[\s!.?]*$',
                r'^(how\s+are\s+you|how\s+is\s+it\s+going|what\s+is\s+up)[\s!.?]*$',
            ],
        }
        self.compiled_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.patterns.items()
        }

    def _init_synonyms(self):
        self.synonyms = {
            IntentCategory.IMAGE_GENERATION: ["image","picture","photo","drawing","illustration","artwork","painting","sketch","graphic","visual","render","logo","banner","flux","dalle","midjourney"],
            IntentCategory.VIDEO_GENERATION: ["video","clip","movie","film","animation","motion","gif","reel","runway","pika","sora","kling","wan"],
            IntentCategory.MUSIC_GENERATION: ["song","track","melody","beat","tune","instrumental","music","musicgen","suno","udio","composition"],
            IntentCategory.AUDIO_GENERATION: ["audio","sound","sfx","voice","speech","narration","voiceover","tts","elevenlabs","bark"],
            IntentCategory.CODE_GENERATION: ["code","script","function","class","module","program","app","snippet","implementation","algorithm","library","package"],
            IntentCategory.CODE_REVIEW: ["review","refactor","improve","optimize","clean up","best practice","code quality"],
            IntentCategory.CODE_DEBUG: ["bug","error","issue","problem","debug","fix","crash","exception","broken","incorrect"],
            IntentCategory.MATHEMATICAL: ["calculate","compute","solve","math","equation","formula","integral","derivative","proof","algebra","calculus","geometry","statistics"],
            IntentCategory.RESEARCH: ["research","find","search","investigate","study","academic","scholarly","citation","news","current","weather","stock","price"],
            IntentCategory.TRANSLATION: ["translate","translation","localize","i18n","multilingual"],
            IntentCategory.SUMMARIZATION: ["summarize","summary","tldr","brief","overview","key points","takeaways","gist"],
            IntentCategory.EXPLANATION: ["explain","explanation","what is","how does","why","understand","elaborate","simplify","break down"],
            IntentCategory.CREATIVE_WRITING: ["story","poem","poetry","novel","fiction","creative","narrative","lyrics","haiku"],
            IntentCategory.WEB_DEVELOPMENT: ["website","webpage","web app","landing page","frontend","backend","html","css","react","vue","angular","next.js","svelte","tailwind"],
            IntentCategory.API_DEVELOPMENT: ["api","rest api","graphql","endpoint","route","restful","swagger","openapi"],
            IntentCategory.DATABASE: ["database","schema","table","sql","query","migration","mysql","postgres","mongodb","redis","sqlite","orm","crud"],
            IntentCategory.DATA_ANALYSIS: ["data","dataset","csv","excel","spreadsheet","analytics","statistics","insights","metrics","analysis"],
            IntentCategory.DATA_VISUALIZATION: ["chart","graph","plot","visualization","diagram","dashboard","histogram","heatmap","matplotlib","plotly"],
            IntentCategory.DOCUMENT_CREATION: ["document","pdf","report","letter","email","memo","article","essay","paper","proposal","whitepaper","manual"],
            IntentCategory.CONVERSATION: ["hello","hi","hey","thanks","thank you","bye","goodbye"],
        }

    def _has_negation(self, text: str, pos: int) -> bool:
        before = text[:pos].lower().split()[-6:]
        return any(n in " ".join(before) for n in self.negation_words)

    def _calculate_confidence(self, kws: List[str], pats: List[str], text_len: int) -> float:
        if not kws and not pats:
            return 0.0
        pc = min(len(pats) * 0.35, 0.65)
        kc = min(len(kws) * 0.12, 0.25)
        bonus = 0.1 if (kws and pats) else 0.0
        length = max(0.5, 1.0 - (text_len / 1500) * 0.4)
        return min((pc + kc + bonus) * length, 1.0)

    def _are_related(self, a: IntentCategory, b: IntentCategory) -> bool:
        groups = [
            {IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG},
            {IntentCategory.DATA_ANALYSIS, IntentCategory.DATA_VISUALIZATION},
            {IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION, IntentCategory.AUDIO_GENERATION, IntentCategory.MUSIC_GENERATION},
            {IntentCategory.WEB_DEVELOPMENT, IntentCategory.API_DEVELOPMENT, IntentCategory.DATABASE},
            {IntentCategory.DOCUMENT_CREATION, IntentCategory.RESEARCH},
            {IntentCategory.EXPLANATION, IntentCategory.SUMMARIZATION},
        ]
        for g in groups:
            if a in g and b in g:
                return True
        return False

    def detect(self, text: str, threshold: float = 0.25) -> Optional[IntentResult]:
        text_lower = text.lower()
        results = []
        # Check VIDEO first (so "make a video" doesn't fall into image)
        priority_order = [
            IntentCategory.VIDEO_GENERATION,
            IntentCategory.MUSIC_GENERATION,
            IntentCategory.AUDIO_GENERATION,
            IntentCategory.IMAGE_GENERATION,
        ]
        # Priority intents — slightly boosted
        for intent in priority_order:
            pats = [p.pattern for p in self.compiled_patterns[intent] if p.search(text)]
            kws = []
            for syn in self.synonyms.get(intent, []):
                p = text_lower.find(syn)
                if p >= 0 and not self._has_negation(text, p):
                    kws.append(syn)
            if pats or kws:
                conf = self._calculate_confidence(kws, pats, len(text)) + 0.1
                results.append(IntentResult(intent, min(conf, 0.99), [], kws, pats))
        # Other intents
        for intent, compiled in self.compiled_patterns.items():
            if intent in priority_order:
                continue
            pats = [p.pattern for p in compiled if p.search(text)]
            kws = []
            for syn in self.synonyms.get(intent, []):
                p = text_lower.find(syn)
                if p >= 0 and not self._has_negation(text, p):
                    kws.append(syn)
            if pats or kws:
                conf = self._calculate_confidence(kws, pats, len(text))
                if conf >= threshold:
                    results.append(IntentResult(intent, conf, [], kws, pats))
        results.sort(key=lambda x: x.confidence, reverse=True)
        if not results:
            return IntentResult(IntentCategory.CONVERSATION, 0.5)
        primary = results[0]
        for r in results[1:]:
            if self._are_related(primary.intent, r.intent):
                primary.sub_intents.append(r.intent)
        return primary

    def get_action_type(self, text: str) -> str:
        i = self.detect(text)
        if not i: return "general"
        return {
            IntentCategory.IMAGE_GENERATION: "image",
            IntentCategory.VIDEO_GENERATION: "video",
            IntentCategory.MUSIC_GENERATION: "music",
            IntentCategory.AUDIO_GENERATION: "audio",
            IntentCategory.CODE_GENERATION: "code",
            IntentCategory.CODE_REVIEW: "code",
            IntentCategory.CODE_DEBUG: "code",
            IntentCategory.MATHEMATICAL: "math",
            IntentCategory.RESEARCH: "research",
            IntentCategory.TRANSLATION: "translation",
            IntentCategory.WEB_DEVELOPMENT: "web",
            IntentCategory.API_DEVELOPMENT: "api",
            IntentCategory.DATABASE: "database",
            IntentCategory.DATA_ANALYSIS: "data",
            IntentCategory.DATA_VISUALIZATION: "data",
            IntentCategory.DOCUMENT_CREATION: "document",
            IntentCategory.SUMMARIZATION: "summary",
            IntentCategory.EXPLANATION: "explanation",
            IntentCategory.CREATIVE_WRITING: "creative",
            IntentCategory.CONVERSATION: "conversation",
        }.get(i.intent, "general")

    def get_code_system_prompt(self, text: str) -> str:
        base = get_system_prompt(text)
        i = self.detect(text)
        if not i: return base + "\n\nYou are also a helpful coding assistant."
        sub = {
            IntentCategory.CODE_DEBUG: "\n\nYou are also an expert debugger. Identify root cause, explain WHY, provide exact fix, suggest prevention.",
            IntentCategory.CODE_REVIEW: "\n\nYou are also a senior code reviewer. Cover: quality, bugs, edge cases, performance, best practices, security.",
            IntentCategory.CODE_GENERATION: "\n\nYou are also an expert software engineer. Write clean, production-ready code with error handling and comments.",
            IntentCategory.WEB_DEVELOPMENT: "\n\nYou are also a full-stack web developer. Use modern best practices, responsive, accessible, reusable components.",
            IntentCategory.API_DEVELOPMENT: "\n\nYou are also an API expert. RESTful, error handling, validation, security, documentation.",
            IntentCategory.DATABASE: "\n\nYou are also a database expert. Efficient schemas, optimized queries, indexes, integrity, best practices.",
        }
        return base + sub.get(i.intent, "\n\nYou are also a helpful coding assistant.")

_detector = AdvancedIntentDetector()

def detect_intent(prompt: str) -> Optional[IntentResult]:
    return _detector.detect(prompt)


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True
    image_size: str = "1024x1024"
    image_quality: str = "medium"
    model: Optional[str] = "helox"
    mode: Optional[str] = "general"

class RegenerateRequest(BaseModel):
    conversation_id: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

async def _execute_supabase_with_retry(query_builder, description="Supabase Op"):
    max_retries = 3
    last = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(query_builder.execute)
        except Exception as e:
            last = e
            err = str(e)
            if "502" in err or "Bad Gateway" in err or "Expecting value" in err:
                logger.warning(f"{description} transient (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
            else:
                logger.error(f"{description} failed: {e}")
                break
    if last:
        raise last

async def cleanup_session_cache_safe():
    try:
        await cleanup_session_cache()
    except Exception:
        pass

async def get_user(req: Request, res: Response, remember: Optional[bool] = None) -> Dict[str, Any]:
    global _pending_new_user_id, _new_user_created_event
    await cleanup_session_cache_safe()

    primary_id = req.cookies.get(PRIMARY_COOKIE)
    backup_id = req.cookies.get(BACKUP_COOKIE)
    device_cookie = req.cookies.get(DEVICE_COOKIE)
    stored_fp = req.cookies.get(FINGERPRINT_COOKIE)
    session_token = req.cookies.get(SESSION_TOKEN_COOKIE)
    session_expiry = req.cookies.get(SESSION_EXPIRY_COOKIE)
    current_fp = generate_device_fingerprint(req)

    if remember is None:
        remember = not is_session_expired(session_expiry or "0")

    user_obj = {
        "id": None, "email": None, "memory": "",
        "fingerprint": current_fp, "session_valid": False, "session_token": None,
        "is_premium": False, "is_lifetime": False, "plan": "free"
    }

    user_id = None
    if primary_id and session_token:
        if is_session_expired(session_expiry or "0"):
            clear_session_cookies(res)
        elif await validate_session_token(primary_id, session_token):
            user_id = primary_id
            user_obj["session_valid"] = True
            user_obj["session_token"] = session_token
            if should_refresh_session(session_expiry or "0"):
                new_token = await create_user_session(user_id, current_fp, remember)
                if new_token:
                    user_obj["session_token"] = new_token

    if not user_id and backup_id:
        user_id = backup_id

    if not user_id and device_cookie:
        try:
            fp_part = device_cookie.split("_")[0] if "_" in device_cookie else device_cookie
            r = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", fp_part).limit(1),
                description="FP Lookup"
            )
            if r.data:
                user_id = r.data[0]["id"]
        except Exception as e:
            logger.error(f"FP lookup failed: {e}")

    if not user_id and stored_fp:
        try:
            r = await _execute_supabase_with_retry(
                supabase.table("users").select("id").eq("fingerprint", stored_fp).limit(1),
                description="Stored FP Lookup"
            )
            if r.data:
                user_id = r.data[0]["id"]
        except Exception as e:
            logger.error(f"Stored FP lookup failed: {e}")

    if user_id:
        try:
            r = await _execute_supabase_with_retry(
                supabase.table("users").select("*").eq("id", user_id).limit(1),
                description="User by ID"
            )
            if r.data:
                u = r.data[0]
                user_obj.update({
                    "id": u["id"], "email": u.get("email"),
                    "memory": u.get("memory", ""),
                    "is_premium": u.get("is_premium", False),
                    "is_lifetime": u.get("is_lifetime", False),
                    "plan": u.get("plan", "free"),
                })
                if u.get("fingerprint") != current_fp:
                    try:
                        await _execute_supabase_with_retry(
                            supabase.table("users").update({"fingerprint": current_fp}).eq("id", user_id),
                            description="Update FP"
                        )
                    except Exception:
                        pass
                if not user_obj["session_valid"]:
                    new_token = await create_user_session(user_id, current_fp, remember)
                    if new_token:
                        user_obj["session_token"] = new_token
                        user_obj["session_valid"] = True
                if user_obj["session_token"]:
                    set_session_cookies(res, user_id, current_fp, user_obj["session_token"], remember)
                return user_obj
        except Exception as e:
            logger.error(f"User fetch failed: {e}")

    # Create new user (with lock to avoid races)
    async with _new_user_lock:
        if _pending_new_user_id is not None and _pending_new_user_id not in ("creating", "__failed__"):
            cand = _pending_new_user_id
            try:
                r = await _execute_supabase_with_retry(
                    supabase.table("user_sessions").select("token").eq("user_id", cand)
                    .eq("is_valid", True).order("created_at", desc=True).limit(1),
                    description="Reuse Pending User"
                )
                if r.data:
                    tok = r.data[0]["token"]
                    set_session_cookies(res, cand, current_fp, tok, remember)
                    _session_cache[cand] = {"token": tok, "time": time.time()}
                    _pending_new_user_id = None
                    return {**user_obj, "id": cand, "session_valid": True, "session_token": tok,
                            "fingerprint": current_fp}
            except Exception:
                pass

        _pending_new_user_id = "creating"
        try:
            new_id = str(uuid.uuid4())
            try:
                await _execute_supabase_with_retry(
                    supabase.table("users").upsert({
                        "id": new_id, "email": f"anon+{new_id[:8]}@local",
                        "memory": "", "fingerprint": current_fp,
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }, on_conflict="id"),
                    description="Create Anon User"
                )
            except Exception as e:
                logger.error(f"Anon user creation failed: {e}")

            new_token = await create_user_session(new_id, current_fp, remember)
            if not new_token:
                _pending_new_user_id = "__failed__"
                _new_user_created_event.set()
                raise HTTPException(500, "Failed to create user session")

            set_session_cookies(res, new_id, current_fp, new_token, remember)
            _pending_new_user_id = new_id
            _new_user_created_event.set()

            return {**user_obj, "id": new_id, "session_valid": True,
                    "session_token": new_token, "fingerprint": current_fp}
        except HTTPException:
            _pending_new_user_id = "__failed__"
            _new_user_created_event.set()
            raise
        except Exception as e:
            _pending_new_user_id = "__failed__"
            _new_user_created_event.set()
            logger.error(f"get_user unexpected: {e}")
            raise HTTPException(500, "Failed to create user session")

async def get_user_with_auth(req: Request, res: Response, remember: bool = True) -> Dict[str, Any]:
    auth_header = req.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
        if SUPABASE_ANON_KEY and token == SUPABASE_ANON_KEY:
            return await get_user(req, res, remember)
        try:
            user = await asyncio.to_thread(supabase.auth.get_user, token)
            if user and user.user:
                user_id = user.user.id
                await ensure_user_exists(user_id)
                return {"id": user_id, "session_valid": True, "memory": "",
                        "fingerprint": generate_device_fingerprint(req),
                        "session_token": None, "is_premium": False, "is_lifetime": False, "plan": "free"}
        except Exception as e:
            logger.debug(f"Auth header failed: {e}")
    return await get_user(req, res, remember)

async def save_message(user_id: str, conv_id: str, role: str, content: str):
    await _execute_supabase_with_retry(
        supabase.table("messages").insert({
            "id": str(uuid.uuid4()), "conversation_id": conv_id,
            "role": role, "content": content,
            "created_at": datetime.now(timezone.utc).isoformat()
        }),
        description="Save Message"
    )

def estimate_tokens(text: str) -> int:
    return len(text) // 4

async def get_history(conv_id: str, limit: int = 50):
    res = await _execute_supabase_with_retry(
        supabase.table("messages").select("role, content")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False).limit(limit),
        description="Get History"
    )
    raw = res.data or []
    MAX_TOK = 4000
    cur = 0
    final = []
    for m in reversed(raw):
        t = estimate_tokens(m.get("content", ""))
        if cur + t > MAX_TOK:
            break
        final.append(m)
        cur += t
    final.reverse()
    return [{"role": m["role"], "content": m["content"]} for m in final]

async def get_or_create_conversation(user_id: str, proposed_id: Optional[str], title: str) -> str:
    lock_key = proposed_id or "__new__"
    lock = _get_conv_lock(lock_key)
    async with lock:
        if proposed_id:
            for _ in range(3):
                check = await _execute_supabase_with_retry(
                    supabase.table("conversations").select("id").eq("id", proposed_id).eq("user_id", user_id).limit(1),
                    description="Conv Check"
                )
                if check.data:
                    _conv_creation_locks.pop(lock_key, None)
                    return proposed_id
                if _ < 2:
                    await asyncio.sleep(0.2)
        new_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": new_id, "user_id": user_id, "title": title[:50],
                "created_at": now, "updated_at": now
            }),
            description="Create Conv"
        )
        _conv_creation_locks.pop(lock_key, None)
        return new_id


# =========================
# API HEADERS
# =========================
def get_groq_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def get_groq_headers_multipart() -> Dict[str, str]:
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}

def get_openai_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def get_replicate_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json",
            "Prefer": "wait"}

def _parse_retry_after(error_body: str) -> float:
    m = re.search(r'try again in ([\d\.]+)s', error_body)
    if m:
        return float(m.group(1)) + 0.5
    return 5.0


# =========================
# TAVILY WEB SEARCH
# =========================
async def perform_web_search_formatted(query: str) -> Tuple[str, str]:
    if not TAVILY_API_KEY:
        return "[Search API Key missing]", ""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_API_KEY, "query": query,
                "search_depth": "basic", "max_results": 5,
                "include_answer": True, "include_images": True,
                "include_raw_content": False,
            })
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            raw_images = data.get("images", [])
            if not results:
                return "[No search results found]", ""
            context = ""
            if data.get("answer"):
                context += f"Answer: {data['answer']}\n"
            for i, r in enumerate(results):
                context += f"[{i+1}] {r['title']}: {r['content']}\nURL: {r['url']}\n\n"
            domain_images = {}
            for img in raw_images[:20]:
                try:
                    d = urlparse(img).hostname or ""
                    if d and d not in domain_images:
                        domain_images[d] = img
                except Exception:
                    pass
            html = '<div class="search-sources-bar">\n<i class="fa-solid fa-globe"></i> Sources:\n'
            for r in results[:5]:
                d = urlparse(r["url"]).hostname or ""
                html += (f'<a href="{r["url"]}" class="source-chip" target="_blank" rel="noopener">'
                         f'<img class="source-chip-img" src="https://www.google.com/s2/favicons?domain={d}&sz=32" '
                         f'alt="" onerror="this.style.display=\'none\'">{d}</a>\n')
            html += '</div>\n\n'
            for i, r in enumerate(results[:4]):
                d = urlparse(r["url"]).hostname or ""
                thumb = domain_images.get(d, "")
                fav = f"https://www.google.com/s2/favicons?domain={d}&sz=32"
                if thumb:
                    html += (f'<a href="{r["url"]}" class="search-card" target="_blank" rel="noopener">'
                             f'<img class="search-thumb" src="{thumb}" alt="" loading="lazy" '
                             f'onerror="this.src=\'{fav}\';this.style.width=\'32px\';this.style.height=\'32px\';this.style.borderRadius=\'6px\';">'
                             f'<div class="search-info"><div class="search-title">{r["title"]}</div>'
                             f'<div class="search-link"><img class="search-link-favicon" src="{fav}" '
                             f'alt="" onerror="this.style.display=\'none\'">{d}</div>'
                             f'<div class="search-snippet">{r.get("content", "")[:300]}</div></div></a>\n\n')
                else:
                    html += (f'<a href="{r["url"]}" class="search-card compact" target="_blank" rel="noopener">'
                             f'<div class="search-info"><div class="search-title">{r["title"]}</div>'
                             f'<div class="search-link"><img class="search-link-favicon" src="{fav}" '
                             f'alt="" onerror="this.style.display=\'none\'">{r["url"][:80]}</div>'
                             f'<div class="search-snippet">{r.get("content", "")[:300]}</div></div></a>\n\n')
            return context, html
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "[Search failed]", ""


# =========================
# GROQ CHAT (chat + memory consolidation)
# =========================
async def stream_groq_chat(messages: list, model: str = None, max_tokens: int = 4096):
    use_model = model or GROQ_FALLBACK_CHAT
    attempt = 0
    while attempt < GROQ_MAX_RETRIES:
        attempt += 1
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json={"model": use_model, "messages": messages, "stream": True, "max_tokens": max_tokens}
                ) as resp:
                    if resp.status_code == 429:
                        body = (await resp.aread()).decode()
                        delay = _parse_retry_after(body)
                        logger.warning(f"Groq 429 attempt {attempt}/{GROQ_MAX_RETRIES}, retry in {delay:.1f}s")
                        await asyncio.sleep(delay); continue
                    if resp.status_code == 413 and use_model == "openai/gpt-oss-120b":
                        use_model = "llama-3.1-8b-instant"
                        max_tokens = min(max_tokens, 1500)
                        await asyncio.sleep(2); continue
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise Exception(f"Groq Error {resp.status_code}: {body.decode()}")
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]": return
                            try:
                                chunk = json.loads(payload)
                                d = chunk["choices"][0]["delta"].get("content")
                                if d: yield d
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                    return
            except httpx.RemoteProtocolError:
                if attempt < GROQ_MAX_RETRIES:
                    await asyncio.sleep(2.0); continue
                raise
    raise Exception(f"Groq rate limit exceeded after {GROQ_MAX_RETRIES} retries.")

async def groq_chat_sync(messages: list, model: str = None, max_tokens: int = 4096) -> str:
    use_model = model or GROQ_FALLBACK_CHAT
    attempt = 0
    while attempt < GROQ_MAX_RETRIES:
        attempt += 1
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": use_model, "messages": messages, "max_tokens": max_tokens}
            )
            if r.status_code == 429:
                await asyncio.sleep(_parse_retry_after(r.text)); continue
            if r.status_code == 413 and use_model == "openai/gpt-oss-120b":
                use_model = "llama-3.1-8b-instant"
                max_tokens = min(max_tokens, 1500); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    raise Exception("Groq retries exhausted")


# =========================
# OPENAI CHAT (GPT-4o)
# =========================
async def stream_openai_chat(messages: list, model: str = OPENAI_CHAT_MODEL, max_tokens: int = 4096):
    if not OPENAI_API_KEY:
        yield "[OpenAI API not configured]"; return
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST", "https://api.openai.com/v1/chat/completions",
                headers=get_openai_headers(),
                json={"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise Exception(f"OpenAI Error {resp.status_code}: {body.decode()}")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload == "[DONE]": return
                        try:
                            chunk = json.loads(payload)
                            d = chunk["choices"][0]["delta"].get("content")
                            if d: yield d
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
        except httpx.RemoteProtocolError:
            raise

async def openai_chat_sync(messages: list, model: str = OPENAI_CHAT_MODEL, max_tokens: int = 4096) -> str:
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API not configured")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=get_openai_headers(),
            json={"model": model, "messages": messages, "max_tokens": max_tokens}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# =========================
# MEMORY UPDATE (LLM CONSOLIDATION)
# =========================
async def _background_update_user_memory(user_id: str, old_memory: str, user_prompt: str, assistant_response: str):
    try:
        await update_user_memory(user_id, old_memory, user_prompt, assistant_response)
    except Exception as e:
        logger.error(f"Background memory update failed: {e}")

async def update_user_memory(user_id: str, old_memory: str, user_prompt: str, assistant_response: str):
    if not GROQ_API_KEY:
        return
    memory_prompt = """You are a memory management AI. Update the user's long-term memory based on the latest interaction.

Rules:
1. Retain permanent user facts (Name, Job, Preferences).
2. Update current context/topic.
3. Be concise (max 250 words).
4. Discard conversational filler.
5. Maintain continuity.
6. Return ONLY the new memory string."""
    user_msg = f"""Current Memory:
{old_memory if old_memory else "[Empty]"}

Latest Interaction:
User: {user_prompt}
Assistant: {assistant_response}

Updated Memory:"""
    messages = [{"role": "system", "content": memory_prompt}, {"role": "user", "content": user_msg}]
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json={"model": GROQ_MEMORY_MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.1}
                )
                if r.status_code == 429:
                    await asyncio.sleep(5 * (attempt + 1)); continue
                r.raise_for_status()
                new_mem = r.json()["choices"][0]["message"]["content"].strip()
                await _execute_supabase_with_retry(
                    supabase.table("users").update({"memory": new_mem}).eq("id", user_id),
                    description="Update Memory"
                )
                if user_id in _session_cache:
                    _session_cache[user_id]["memory"] = new_mem
                return
        except Exception as e:
            logger.error(f"Memory update attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2)


# =========================
# REPLICATE: IMAGE, VIDEO, MUSIC (BIGGER MODELS)
# =========================
async def _replicate_run_model(model_ref: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Submit to Replicate and poll until complete. Returns the output dict."""
    if not REPLICATE_API_TOKEN:
        raise Exception("REPLICATE_API_TOKEN not configured")
    async with httpx.AsyncClient(timeout=60) as client:
        # Create prediction
        # model_ref can be "owner/model" or "owner/model:version"
        url = f"https://api.replicate.com/v1/models/{model_ref.split(':')[0]}/predictions"
        body = {"input": input_data}
        if ":" in model_ref:
            body["version"] = model_ref.split(":")[1]
        r = await client.post(url, headers=get_replicate_headers(), json=body)
        if r.status_code not in (200, 201):
            raise Exception(f"Replicate create failed {r.status_code}: {r.text}")
        prediction = r.json()
        # If sync response, return immediately
        if prediction.get("status") == "succeeded":
            return prediction
        # Prefer sync wait header
        get_url = prediction.get("urls", {}).get("get") or prediction["id"]
        elapsed = 0.0
        while elapsed < REPLICATE_MAX_WAIT:
            await asyncio.sleep(REPLICATE_POLL_INTERVAL)
            elapsed += REPLICATE_POLL_INTERVAL
            poll = await client.get(get_url, headers=get_replicate_headers())
            if poll.status_code != 200:
                continue
            data = poll.json()
            status = data.get("status")
            if status == "succeeded":
                return data
            if status == "failed":
                err = data.get("error") or "Replicate prediction failed"
                raise Exception(f"Replicate: {err}")
            if status == "canceled":
                raise Exception("Replicate prediction was canceled")
        raise Exception(f"Replicate timeout after {int(REPLICATE_MAX_WAIT)}s")

async def generate_image_replicate(prompt: str, size: str = "1024x1024") -> str:
    """Generate image with Flux 1.1 Pro (or whatever REPLICATE_IMAGE_MODEL points to).
    Returns a public URL. We download + reupload to Supabase for persistence."""
    w, h = (1024, 1024)
    try:
        parts = size.lower().split("x")
        w, h = int(parts[0]), int(parts[1])
    except Exception:
        pass
    input_data = {
        "prompt": prompt,
        "aspect_ratio": f"{w}:{h}" if w == h else f"{w}:{h}",
        "output_format": "png",
        "output_quality": 100,
        "safety_tolerance": 2,
    }
    result = await _replicate_run_model(REPLICATE_IMAGE_MODEL, input_data)
    output = result.get("output")
    if isinstance(output, list) and output:
        output = output[0]
    if not isinstance(output, str) or not output:
        raise Exception(f"Replicate returned no image URL: {result}")
    # Download + persist to Supabase
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(output)
        r.raise_for_status()
        img_bytes = r.content
    filename = f"flux_{uuid.uuid4().hex[:8]}.png"
    return await upload_bytes_to_storage(img_bytes, filename, "image/png", bucket="ai-media")

async def generate_video_replicate(prompt: str, image_url: Optional[str] = None) -> str:
    """Generate video with Wan 2.1 14B (text-to-video or image-to-video). Returns video URL."""
    input_data = {"prompt": prompt}
    if image_url:
        input_data["image"] = image_url
    result = await _replicate_run_model(REPLICATE_VIDEO_MODEL, input_data)
    output = result.get("output")
    if isinstance(output, list) and output:
        output = output[0]
    if not isinstance(output, str) or not output:
        raise Exception(f"Replicate returned no video URL: {result}")
    # Download + persist
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(output)
        r.raise_for_status()
        vid_bytes = r.content
    filename = f"wan_{uuid.uuid4().hex[:8]}.mp4"
    return await upload_bytes_to_storage(vid_bytes, filename, "video/mp4", bucket="ai-media")

async def generate_music_replicate(prompt: str, duration: int = 30) -> str:
    """Generate music with Meta MusicGen. Returns audio URL."""
    input_data = {
        "prompt": prompt,
        "model": "thepxlster/musicgen-stereo-large" if False else "large",
        "duration": min(max(duration, 5), 120),
        "output_format": "mp3",
        "normalization_strategy": "peak",
    }
    result = await _replicate_run_model(REPLICATE_MUSIC_MODEL, input_data)
    output = result.get("output")
    if isinstance(output, str) and output.startswith("data:"):
        # base64 data URL
        b64 = output.split(",", 1)[1]
        audio_bytes = base64.b64decode(b64)
    elif isinstance(output, str):
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(output)
            r.raise_for_status()
            audio_bytes = r.content
    elif isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str) and first.startswith("data:"):
            audio_bytes = base64.b64decode(first.split(",", 1)[1])
        else:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(first)
                r.raise_for_status()
                audio_bytes = r.content
    else:
        raise Exception(f"MusicGen returned no audio: {result}")
    filename = f"musicgen_{uuid.uuid4().hex[:8]}.mp3"
    return await upload_bytes_to_storage(audio_bytes, filename, "audio/mpeg", bucket="ai-media")

async def upload_bytes_to_storage(file_bytes: bytes, filename: str, content_type: str,
                                  bucket: str = "ai-media") -> str:
    try:
        path = f"public/{filename}"
        await asyncio.to_thread(
            lambda: supabase.storage.from_(bucket).upload(path, file_bytes, {"content-type": content_type})
        )
        url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
        logger.info(f"Uploaded {filename} -> {url}")
        return url
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        b64 = base64.b64encode(file_bytes).decode()
        return f"data:{content_type};base64,{b64}"


# =========================
# PROGRESSIVE IMAGE STREAMING
# =========================
def generate_progressive_frames(image_b64: str, steps: int = 8) -> list:
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        frames = []
        curve = [3, 8, 15, 28, 45, 65, 82, 95][:steps]
        for i, q in enumerate(curve):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, progressive=True, optimize=True)
            fb64 = base64.b64encode(buf.getvalue()).decode()
            frames.append({"progress": int((i + 1) / len(curve) * 100), "data": fb64})
            buf.close()
        return frames
    except Exception as e:
        logger.error(f"Progressive frame gen failed: {e}")
        return [{"progress": 100, "data": image_b64}]


# =========================
# VIDEO FRAME EXTRACTION (for analysis)
# =========================
def get_video_duration(video_bytes: bytes) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(video_bytes)
        path = tf.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps == 0:
            return 0.0
        return fc / fps
    finally:
        if os.path.exists(path):
            os.remove(path)

def extract_video_frames(video_bytes: bytes, max_frames: int = 4) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(video_bytes)
        path = tf.name
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            return []
        indices = [int(i * total / max_frames) for i in range(max_frames)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buf = cv2.imencode('.jpg', rgb)
                frames.append(base64.b64encode(buf).decode())
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return []
    finally:
        if os.path.exists(path):
            os.remove(path)


# =========================
# ANALYSIS MESSAGE BUILDERS
# =========================
def _build_image_analysis_messages(image_b64: str, mime_type: str, user_prompt: Optional[str]) -> list:
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        {"type": "text", "text": user_prompt or (
            "Analyze this image in detail. Describe what you see, read any text, "
            "explain any code or diagrams, and provide a comprehensive analysis."
        )}
    ]
    return [
        {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

def _build_code_analysis_messages(code_text: str, filename: str, language: str, user_prompt: Optional[str]) -> list:
    instruction = user_prompt or f"Analyze this {language} code from the file `{filename}`."
    user_content = f"""{instruction}

```{language.lower()}
{code_text}
```

Provide a thorough code review: bugs, security, performance, style, and an improved version."""
    return [
        {"role": "system", "content": CODE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

def _build_document_analysis_messages(doc_text: str, filename: str, user_prompt: Optional[str]) -> list:
    instruction = user_prompt or f"Analyze the content from the file `{filename}`."
    user_content = f"""{instruction}

--- FILE CONTENT START ---
{doc_text}
--- FILE CONTENT END ---

Provide a thorough analysis: summary, key points, structure, issues, recommendations."""
    return [
        {"role": "system", "content": DOCUMENT_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


# =========================
# STREAMING GENERATORS
# =========================
async def _stream_image_generation(prompt: str, size: str, quality: str, result: dict):
    yield sse({"type": "image_generating"})
    yield sse({"type": "status", "message": "Generating with Flux 1.1 Pro (Replicate)..."})
    try:
        image_url = await generate_image_replicate(prompt, size=size)
        # Download for progressive frames
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(image_url)
            r.raise_for_status()
            image_b64 = base64.b64encode(r.content).decode()
        frames = generate_progressive_frames(image_b64, steps=8)
        for frame in frames:
            yield sse({"type": "image_progress", "progress": frame["progress"], "data": frame["data"]})
            await asyncio.sleep(0.08)
        is_png = not image_b64.startswith('/9j/')
        mime = "data:image/png;base64," if is_png else "data:image/jpeg;base64,"
        markdown_image = f"![Generated Image]({image_url})"
        yield sse({"type": "image_generated", "url": image_url, "data": image_b64,
                   "size": size, "quality": quality})
        result["text"] = markdown_image
    except Exception as e:
        logger.error(f"Image gen stream error: {e}")
        yield sse({"type": "image_error", "error": str(e)})
        result["text"] = f"[Image generation failed: {e}]"

async def _stream_video_generation(prompt: str, result: dict):
    if not REPLICATE_API_TOKEN:
        msg = "Video generation not configured (REPLICATE_API_TOKEN missing)."
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {msg}*"})
        result["text"] = f"[{msg}]"
        return
    yield sse({"type": "status", "message": "Submitting to Wan 2.1 14B (Replicate)..."})
    try:
        video_url = await generate_video_replicate(prompt)
        yield sse({"type": "status", "message": "Video ready!"})
        yield sse({"type": "video", "url": video_url})
        markdown_video = (
            f'\n\n<video controls autoplay muted loop playsinline '
            f'style="max-width:100%;border-radius:12px;display:block;margin:8px 0;">'
            f'<source src="{video_url}" type="video/mp4">'
            f'Your browser does not support the video tag.</video>\n'
        )
        result["text"] = markdown_video
    except Exception as e:
        logger.error(f"Video gen stream error: {e}")
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {e}*"})
        result["text"] = f"[Video generation failed: {e}]"

async def _stream_music_generation(prompt: str, duration: int, result: dict):
    if not REPLICATE_API_TOKEN:
        msg = "Music generation not configured (REPLICATE_API_TOKEN missing)."
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {msg}*"})
        result["text"] = f"[{msg}]"
        return
    yield sse({"type": "status", "message": f"Composing with MusicGen (duration: {duration}s)..."})
    try:
        audio_url = await generate_music_replicate(prompt, duration=duration)
        yield sse({"type": "status", "message": "Music ready!"})
        yield sse({"type": "music", "url": audio_url, "prompt": prompt, "duration": duration})
        markdown_audio = (
            f'\n\n<audio controls style="width:100%;margin:8px 0;">'
            f'<source src="{audio_url}" type="audio/mpeg">'
            f'Your browser does not support the audio element.</audio>\n'
        )
        result["text"] = markdown_audio
    except Exception as e:
        logger.error(f"Music gen stream error: {e}")
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {e}*"})
        result["text"] = f"[Music generation failed: {e}]"

async def _stream_chat_response(prompt: str, conversation_id: str, model_key: str,
                                mode: Optional[str], user_id: str, user_memory: str,
                                result: dict):
    model_config = MODEL_ROUTING.get(model_key, MODEL_ROUTING["helox"])
    chat_model = model_config["chat"]
    provider = model_config["provider"]

    should_search = False
    intent = detect_intent(prompt)
    if intent and intent.intent == IntentCategory.RESEARCH:
        should_search = True
    elif mode in ("research", "finance", "web"):
        should_search = True
    else:
        time_kw = ['today','now','current','latest','recent','2024','2025',
                   'price','stock','news','weather','score','update','happening']
        if any(k in prompt.lower() for k in time_kw):
            should_search = True

    search_context, search_html = "", ""
    if should_search and TAVILY_API_KEY:
        try:
            search_context, search_html = await perform_web_search_formatted(prompt)
            if search_html:
                yield sse({"type": "search_results", "html": search_html})
        except Exception as e:
            logger.error(f"Search failed: {e}")

    system_prompt = get_system_prompt(prompt, mode)

    # Intent-specific tuning
    if intent:
        if intent.intent == IntentCategory.MATHEMATICAL:
            system_prompt += "\n\nYou are a mathematical expert. Think step-by-step. Use LaTeX for math."
        elif intent.intent == IntentCategory.TRANSLATION:
            system_prompt += "\n\nYou are a professional translator. Provide accurate, context-aware translations."
        elif intent.intent in (IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG):
            system_prompt = _detector.get_code_system_prompt(prompt)

    if user_memory:
        system_prompt += f"\n\nUser Context: {user_memory}"

    messages = [{"role": "system", "content": system_prompt}]
    try:
        history = await get_history(conversation_id, limit=10)
        messages.extend(history)
    except Exception as e:
        logger.warning(f"History load failed: {e}")

    if (search_context and search_context not in
            ("[Search API Key missing]", "[No search results found]", "[Search failed]")):
        user_content = f"""Using these search results as context:

{search_context}

User question: {prompt}

Provide a comprehensive answer based on the search results above. Cite sources as [1], [2] etc."""
    else:
        user_content = prompt
    messages.append({"role": "user", "content": user_content})

    full_response = ""
    stream_fn = stream_groq_chat if provider == "groq" else stream_openai_chat
    try:
        async for delta in stream_fn(messages, model=chat_model):
            full_response += delta
            yield sse({"type": "text_delta", "content": delta})
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        if not full_response:
            full_response = f"[Error: {e}]"
            yield sse({"type": "text_delta", "content": f"\n\n*Error: {e}*"})
    result["text"] = full_response


# =========================
# ENDPOINTS — ROOT
# =========================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "running",
        "service": "HeloXAi Unified",
        "version": "5.0.0",
        "models": {
            "chat": OPENAI_CHAT_MODEL,
            "vision": OPENAI_VISION_MODEL,
            "tts": OPENAI_TTS_MODEL,
            "stt": GROQ_STT_MODEL,
            "image": f"replicate/{REPLICATE_IMAGE_MODEL}",
            "video": f"replicate/{REPLICATE_VIDEO_MODEL}",
            "music": f"replicate/{REPLICATE_MUSIC_MODEL}",
            "memory": GROQ_MEMORY_MODEL,
        },
        "features": [
            "chat", "code", "math", "web_search", "tts", "stt",
            "image_generation_flux", "video_generation_wan",
            "music_generation_musicgen", "audio_generation",
            "image_analysis", "code_analysis", "document_analysis", "archive_analysis",
            "finance", "model_routing", "mode_routing",
            "progressive_image_streaming", "progressive_video_streaming",
            "intelligent_memory", "creator_detection", "rate_limiting",
            "advanced_file_extraction", "production_auth"
        ],
        "endpoints": {
            "chat": "POST /ask/universal",
            "new_chat": "POST /newchat",
            "analysis": "POST /analysis",
            "delete_chat": "DELETE /chats/{chat_id}",
            "list_chats": "GET /chats",
            "messages": "GET /chat/{conversation_id}/messages",
            "user_plan": "GET /user/plan",
            "user_info": "GET /user/info",
            "tts": "POST /tts",
            "tts_voices": "GET /tts/voices",
            "stt": "POST /stt",
            "stop": "POST /stop",
            "regenerate": "POST /regenerate",
            "logout": "POST /session/logout",
            "validate_session": "POST /session/validate",
            "refresh_session": "POST /session/refresh",
            "analyze_intent": "POST /analyze-intent"
        }
    }

@app.options("/{full_path:path}")
async def preflight(full_path: str):
    return Response(status_code=200)

@app.get("/robots.txt")
def robots():
    return PlainTextResponse("User-agent: *\nDisallow:")


# =========================
# USER ENDPOINTS
# =========================
@app.get("/user/plan")
async def get_user_plan(req: Request):
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = auth.replace("Bearer ", "")
    try:
        user = await asyncio.to_thread(supabase.auth.get_user, token)
        if not user or not user.user:
            raise HTTPException(401, "Invalid token")
        uid = user.user.id
        r = await _execute_supabase_with_retry(
            supabase.table("users").select("plan, is_premium, is_lifetime").eq("id", uid).limit(1),
            description="User Plan"
        )
        if r.data and r.data[0]:
            u = r.data[0]
            plan = "free"
            if u.get("is_lifetime"): plan = "lifetime"
            elif u.get("is_premium"): plan = u.get("plan", "ultimate_monthly") or "ultimate_monthly"
            else: plan = u.get("plan", "free") or "free"
            return {"plan": plan, "is_premium": bool(u.get("is_premium", False)),
                    "is_lifetime": bool(u.get("is_lifetime", False))}
        return {"plan": "free", "is_premium": False, "is_lifetime": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan endpoint error: {e}")
        return {"plan": "free", "is_premium": False, "is_lifetime": False}

@app.get("/user/info")
async def get_user_info(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    return {
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_identified": True,
        "session_valid": user.get("session_valid", False),
        "is_authenticated": bool(user.get("email") and not user.get("email", "").startswith("anon+"))
    }

@app.post("/user/merge")
async def merge_user(req: Request, res: Response):
    body = await req.json()
    target_id = body.get("target_user_id")
    user = await get_user_with_auth(req, res)
    if not target_id or target_id == user["id"]:
        return {"status": "no_merge_needed"}
    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations").update({"user_id": target_id}).eq("user_id", user["id"]),
            description="Merge Conv"
        )
        await _execute_supabase_with_retry(
            supabase.table("messages").update({"user_id": target_id}).eq("user_id", user["id"]),
            description="Merge Msgs"
        )
        fp = user.get("fingerprint", "")
        tok = await create_user_session(target_id, fp, True)
        if tok:
            set_session_cookies(res, target_id, fp, tok, True)
        return {"status": "merged", "new_user_id": target_id}
    except Exception as e:
        logger.error(f"User merge failed: {e}")
        raise HTTPException(500, "Failed to merge user data")


# =========================
# SESSION MANAGEMENT
# =========================
@app.post("/session/validate")
async def validate_session_ep(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    return {
        "valid": user.get("session_valid", False),
        "user_id": user["id"],
        "fingerprint": user.get("fingerprint", "")[:8] + "...",
        "is_authenticated": bool(user.get("email") and not user.get("email", "").startswith("anon+"))
    }

@app.post("/session/refresh")
async def refresh_session(req: Request, res: Response):
    body = await req.json() if req.headers.get("content-type") == "application/json" else {}
    remember = body.get("remember", True)
    user = await get_user_with_auth(req, res, remember=remember)
    new_token = await create_user_session(user["id"], user.get("fingerprint", ""), remember)
    if new_token:
        set_session_cookies(res, user["id"], user.get("fingerprint", ""), new_token, remember)
    return {"status": "refreshed", "user_id": user["id"],
            "expires_in": SESSION_DURATION if remember else 24*60*60}

@app.post("/session/logout")
async def logout(req: Request, res: Response):
    uid = req.cookies.get(PRIMARY_COOKIE)
    token = req.cookies.get(SESSION_TOKEN_COOKIE)
    if uid and token:
        try:
            await _execute_supabase_with_retry(
                supabase.table("user_sessions").update({"is_valid": False})
                .eq("user_id", uid).eq("token", token),
                description="Invalidate Session"
            )
        except Exception as e:
            logger.error(f"Logout invalidate failed: {e}")
    if uid in _session_cache:
        _session_cache.pop(uid, None)
    clear_session_cookies(res)
    return {"logged_out": True}


# =========================
# MAIN CHAT ENDPOINT
# =========================
@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")

    conversation_id = body.get("conversation_id")
    remember = body.get("remember", True)
    image_size = body.get("image_size", "1024x1024")
    image_quality = body.get("image_quality", "medium")
    model_key = body.get("model", "helox") or "helox"
    mode = body.get("mode", "general")
    music_duration = body.get("music_duration", body.get("duration", 30))

    user = await get_user_with_auth(req, res, remember)
    user_id = user["id"]
    user_memory = user.get("memory", "")

    title = prompt[:50] if len(prompt) > 10 else prompt
    conv_id = await get_or_create_conversation(user_id, conversation_id, title)
    await save_message(user_id, conv_id, "user", prompt)

    intent = detect_intent(prompt)
    logger.info(f"[CHAT] intent={intent.intent.value if intent else 'none'} conf={intent.confidence if intent else 0:.2f}")

    # MUSIC GENERATION
    if intent and intent.intent == IntentCategory.MUSIC_GENERATION:
        async def music_stream():
            result = {}
            async for ev in _stream_music_generation(prompt, int(music_duration), result):
                yield ev
            tail = f"\n\nHere's the music I composed based on: *\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            for ch in tail:
                yield sse({"type": "text_delta", "content": ch})
            full = result.get("text", "") + tail
            try:
                await save_message(user_id, conv_id, "assistant", full)
            except Exception as e:
                logger.error(f"Save music msg failed: {e}")
            asyncio.create_task(_background_update_user_memory(user_id, user_memory, prompt, full))
            yield sse({"type": "done", "conversation_id": conv_id})
        return StreamingResponse(music_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)

    # VIDEO GENERATION
    if intent and intent.intent == IntentCategory.VIDEO_GENERATION:
        async def video_stream():
            result = {}
            async for ev in _stream_video_generation(prompt, result):
                yield ev
            tail = f"\n\nHere's the video I generated based on: *\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            for ch in tail:
                yield sse({"type": "text_delta", "content": ch})
            full = result.get("text", "") + tail
            try:
                await save_message(user_id, conv_id, "assistant", full)
            except Exception as e:
                logger.error(f"Save video msg failed: {e}")
            asyncio.create_task(_background_update_user_memory(user_id, user_memory, prompt, full))
            yield sse({"type": "done", "conversation_id": conv_id})
        return StreamingResponse(video_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)

    # IMAGE GENERATION
    if intent and intent.intent == IntentCategory.IMAGE_GENERATION:
        async def image_stream():
            result = {}
            async for ev in _stream_image_generation(prompt, image_size, image_quality, result):
                yield ev
            tail = f"\n\nHere's the image I generated based on: *\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            for ch in tail:
                yield sse({"type": "text_delta", "content": ch})
            full = result.get("text", "") + tail
            try:
                await save_message(user_id, conv_id, "assistant", full)
            except Exception as e:
                logger.error(f"Save image msg failed: {e}")
            asyncio.create_task(_background_update_user_memory(user_id, user_memory, prompt, full))
            yield sse({"type": "done", "conversation_id": conv_id})
        return StreamingResponse(image_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)

    # TEXT CHAT (with optional search)
    async def chat_stream():
        result = {}
        async for ev in _stream_chat_response(prompt, conv_id, model_key, mode, user_id, user_memory, result):
            yield ev
        full = result.get("text", "")
        try:
            await save_message(user_id, conv_id, "assistant", full)
        except Exception as e:
            logger.error(f"Save chat msg failed: {e}")
        asyncio.create_task(_background_update_user_memory(user_id, user_memory, prompt, full))
        yield sse({"type": "done", "conversation_id": conv_id})
    return StreamingResponse(chat_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)


# =========================
# NEW CHAT
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        body = await req.json()
    except Exception:
        body = {}
    title = body.get("title", "New Chat")[:50]
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await _execute_supabase_with_retry(
        supabase.table("conversations").insert({
            "id": new_id, "user_id": user["id"], "title": title,
            "created_at": now, "updated_at": now
        }),
        description="New Chat"
    )
    return JSONResponse({"id": new_id, "title": title})


# =========================
# CHAT MANAGEMENT
# =========================
@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    r = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True).limit(100),
        description="List Chats"
    )
    return JSONResponse({"chats": r.data or []})

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", chat_id).eq("user_id", user["id"]).limit(1),
            description="Chat Ownership"
        )
        if not check.data:
            raise HTTPException(404, "Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ownership check failed: {e}")
        raise HTTPException(500, "Failed to verify chat")
    try:
        await _execute_supabase_with_retry(
            supabase.table("messages").delete().eq("conversation_id", chat_id),
            description="Delete Messages"
        )
        await _execute_supabase_with_retry(
            supabase.table("conversations").delete().eq("id", chat_id),
            description="Delete Conv"
        )
        return JSONResponse({"deleted": True})
    except Exception as e:
        logger.error(f"Delete chat failed: {e}")
        raise HTTPException(500, "Failed to delete chat")

@app.get("/chat/{conversation_id}/messages")
async def get_messages(conversation_id: str, req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations").select("id").eq("id", conversation_id)
            .eq("user_id", user["id"]).limit(1),
            description="Msg Ownership"
        )
        if not check.data:
            raise HTTPException(404, "Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Msg ownership check failed: {e}")
        raise HTTPException(500, "Failed to verify chat")
    r = await _execute_supabase_with_retry(
        supabase.table("messages").select("id, role, content, created_at")
        .eq("conversation_id", conversation_id).order("created_at", desc=False),
        description="Get Messages"
    )
    return JSONResponse({"messages": r.data or []})


# =========================
# STOP & REGENERATE
# =========================
@app.post("/stop")
async def stop_generation(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    task = active_streams.get(user["id"])
    if task and not task.done():
        task.cancel()
        active_streams.pop(user["id"], None)
        return {"status": "stopped"}
    return {"status": "no_active_stream"}

@app.post("/regenerate")
async def regenerate(req: Request, res: Response):
    body = await req.json()
    conv_id = body.get("conversation_id")
    user = await get_user_with_auth(req, res)
    user_id = user["id"]
    if not conv_id:
        raise HTTPException(400, "conversation_id required")
    msgs = await _execute_supabase_with_retry(
        supabase.table("messages").select("*").eq("conversation_id", conv_id)
        .order("created_at", desc=True).limit(10),
        description="Regen History"
    )
    if not msgs.data:
        raise HTTPException(404, "No messages found")
    last_user = None
    for m in msgs.data:
        if m["role"] == "user":
            last_user = m; break
    if not last_user:
        raise HTTPException(400, "No user message to regenerate")
    await _execute_supabase_with_retry(
        supabase.table("messages").delete().gt("created_at", last_user["created_at"])
        .eq("role", "assistant").eq("conversation_id", conv_id),
        description="Delete Old Assistant"
    )
    last_prompt = last_user.get("content", "")

    async def gen():
        task = asyncio.current_task()
        active_streams[user_id] = task
        try:
            history = await get_history(conv_id)
            base_system = get_system_prompt(last_prompt)
            intent = detect_intent(last_prompt)
            if intent:
                if intent.intent == IntentCategory.MATHEMATICAL:
                    base_system += "\n\nYou are a mathematical expert. Think step-by-step."
                elif intent.intent == IntentCategory.TRANSLATION:
                    base_system += "\n\nYou are a professional translator."
                elif intent.intent in (IntentCategory.CODE_GENERATION, IntentCategory.CODE_REVIEW, IntentCategory.CODE_DEBUG):
                    base_system = _detector.get_code_system_prompt(last_prompt)
            if user.get("memory"):
                base_system += f"\n\nUser Context: {user['memory']}"
            full_history = [{"role": "system", "content": base_system}] + history
            full = ""
            async for tok in stream_openai_chat(full_history, model=OPENAI_CHAT_MODEL):
                if task and task.cancelled(): break
                full += tok
                yield sse({"type": "text_delta", "content": tok})
            try:
                await save_message(user_id, conv_id, "assistant", full)
            except Exception as e:
                logger.error(f"Save regen msg failed: {e}")
            yield sse({"type": "done"})
            asyncio.create_task(_background_update_user_memory(user_id, user.get("memory", ""), last_prompt, full))
        except Exception as e:
            logger.error(f"Regen stream error: {e}")
            yield sse({"type": "text_delta", "content": f"\n\n*Error: {e}*"})
            yield sse({"type": "done"})
        finally:
            active_streams.pop(user_id, None)
    return StreamingResponse(gen(), media_type="text/event-stream", headers=STREAM_HEADERS)


# =========================
# ANALYSIS ENDPOINT
# =========================
@app.post("/analysis")
async def analyze_file(
    req: Request, res: Response,
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    remember: bool = Form(True),
    analysis_type: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
    image_mime: Optional[str] = Form("image/png"),
):
    user = await get_user_with_auth(req, res, remember)

    image_data_b64 = None
    image_mime_type = image_mime or "image/png"
    file_text_content = None
    file_filename = "unknown"
    file_category = FileCategory.UNKNOWN

    if image_base64:
        clean = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        image_data_b64 = clean.strip()
        file_category = FileCategory.IMAGE
    elif file and file.filename:
        file_filename = file.filename
        content_bytes = b""
        while chunk := await file.read(1024 * 1024):
            content_bytes += chunk
            if len(content_bytes) > MAX_FILE_SIZE:
                raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE//(1024*1024)}MB)")
        if not content_bytes:
            raise HTTPException(400, "Empty file uploaded")
        if analysis_type and analysis_type != "auto":
            try:
                file_category = FileCategory(analysis_type)
            except ValueError:
                file_category = get_file_category(file_filename)
        else:
            file_category = get_file_category(file_filename)
        if file.content_type and _is_image_mime(file.content_type):
            file_category = FileCategory.IMAGE
        if file_category == FileCategory.IMAGE:
            image_data_b64 = base64.b64encode(content_bytes).decode()
            image_mime_type = file.content_type or "image/png"
        else:
            extracted = await extract_file_content(content_bytes, file_filename)
            file_text_content = extracted.content
            if not file_text_content.strip() or file_text_content.strip() == "[Binary or unreadable content]":
                raise HTTPException(400, f"Could not extract text from {file_filename}")
    else:
        raise HTTPException(400, "Either 'file' or 'image_base64' must be provided")

    conv_id = await get_or_create_conversation(
        user["id"], conversation_id,
        f"Analysis: {file_filename}" if file_filename else "Image Analysis"
    )
    user_msg = prompt or f"[Uploaded {file_filename} for analysis]"
    await save_message(user["id"], conv_id, "user", user_msg)

    if file_category == FileCategory.IMAGE:
        analysis_messages = _build_image_analysis_messages(image_data_b64, image_mime_type, prompt)
    else:
        language = get_language_from_extension(file_filename)
        if file_category == FileCategory.CODE:
            analysis_messages = _build_code_analysis_messages(file_text_content, file_filename, language, prompt)
        else:
            analysis_messages = _build_document_analysis_messages(file_text_content, file_filename, prompt)

    if stream:
        async def analysis_stream():
            full = ""
            try:
                async for delta in stream_openai_chat(analysis_messages, model=OPENAI_VISION_MODEL):
                    full += delta
                    yield sse({"type": "text_delta", "content": delta})
            except Exception as e:
                logger.error(f"Analysis stream error: {e}")
                if not full:
                    full = f"[Analysis error: {e}]"
                    yield sse({"type": "text_delta", "content": f"*Error: {e}*"})
            try:
                await save_message(user["id"], conv_id, "assistant", full)
            except Exception as e:
                logger.error(f"Save analysis msg failed: {e}")
            yield sse({"type": "done", "conversation_id": conv_id})
        return StreamingResponse(analysis_stream(), media_type="text/event-stream", headers=STREAM_HEADERS)
    else:
        try:
            text = await openai_chat_sync(analysis_messages, model=OPENAI_VISION_MODEL)
        except Exception as e:
            text = f"[Analysis error: {e}]"
        await save_message(user["id"], conv_id, "assistant", text)
        return JSONResponse({"response": text, "conversation_id": conv_id})


# =========================
# MULTI-FILE ANALYSIS
# =========================
@app.post("/analysis/multi")
async def analyze_multi_files(
    req: Request, res: Response,
    files: List[UploadFile] = File(...),
    prompt: str = Form(""),
    stream: bool = Form(True),
):
    user = await get_user_with_auth(req, res)
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 files allowed")

    visual_items = []
    text_items = []
    metadata_list = []

    for f in files:
        content = await f.read()
        filename = f.filename or "unknown"
        ctype = f.content_type or ""
        if not content:
            continue
        if ctype.startswith("video/") or filename.lower().endswith(('.mp4', '.mov', '.webm', '.avi')):
            try:
                dur = get_video_duration(content)
                if dur > 60:
                    raise HTTPException(400, f"Video too long ({dur:.1f}s). Max 60s.")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Video processing: {e}")
                raise HTTPException(400, "Could not process video file")
            for fb64 in extract_video_frames(content):
                visual_items.append({'type': 'video', 'b64': fb64})
        elif ctype.startswith("image/") or get_file_category(filename) == FileCategory.IMAGE:
            visual_items.append({'type': 'image', 'b64': base64.b64encode(content).decode()})
        else:
            cat = get_file_category(filename)
            max_allowed = MAX_ZIP_SIZE if cat == FileCategory.ARCHIVE else MAX_FILE_SIZE
            if len(content) > max_allowed:
                raise HTTPException(400, f"File {filename} too large")
            extracted = await extract_file_content(content, filename)
            text_items.append(f"--- FILE: {filename} ---\n{extracted.content}")
            metadata_list.append(extracted.metadata)

    if visual_items:
        content_parts = [{"type": "text", "text": prompt or "Analyze these visual items in detail."}]
        for item in visual_items:
            mime = "image/jpeg" if item['type'] == 'video' else "image/png"
            content_parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{mime};base64,{item['b64']}"}})
        payload_messages = [{"role": "user", "content": content_parts}]
        if stream:
            async def vstream():
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        r = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=get_openai_headers(),
                            json={"model": OPENAI_VISION_MODEL, "messages": payload_messages}
                        )
                        r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"]
                    yield sse({"type": "text_delta", "content": text})
                    yield sse({"type": "done"})
                except Exception as e:
                    logger.error(f"Visual analysis error: {e}")
                    yield sse({"type": "text_delta", "content": f"*Error: {e}*"})
                    yield sse({"type": "done"})
            return StreamingResponse(vstream(), media_type="text/event-stream", headers=STREAM_HEADERS)
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=get_openai_headers(),
                json={"model": OPENAI_VISION_MODEL, "messages": payload_messages}
            )
            r.raise_for_status()
        return {"analysis": r.json()["choices"][0]["message"]["content"]}

    if text_items:
        combined = "\n\n".join(text_items)
        if prompt:
            combined = f"USER INSTRUCTION: {prompt}\n\n" + combined
        messages = [
            {"role": "system", "content": get_system_prompt(prompt) + "\n\nYou analyze files and code. Be structured and clear."},
            {"role": "user", "content": combined}
        ]
        if stream:
            async def tstream():
                try:
                    async for tok in stream_openai_chat(messages, model=OPENAI_CHAT_MODEL):
                        yield sse({"type": "text_delta", "content": tok})
                    yield sse({"type": "done"})
                except Exception as e:
                    logger.error(f"Text analysis error: {e}")
                    yield sse({"type": "text_delta", "content": f"*Error: {e}*"})
                    yield sse({"type": "done"})
            return StreamingResponse(tstream(), media_type="text/event-stream", headers=STREAM_HEADERS)
        return {"analysis": await openai_chat_sync(messages, model=OPENAI_CHAT_MODEL)}

    raise HTTPException(400, "No valid files provided")


# =========================
# TTS
# =========================
@app.post("/tts")
async def text_to_speech(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    if not OPENAI_API_KEY:
        raise HTTPException(500, "TTS not configured")
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    text = body.get("text", "").strip()
    voice = body.get("voice", "alloy")
    if not text:
        raise HTTPException(400, "Text is required")
    if len(text) > 4096:
        raise HTTPException(400, "Text too long (max 4096 chars)")
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        voice = "alloy"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={"model": OPENAI_TTS_MODEL, "input": text, "voice": voice, "response_format": "mp3"}
            )
            r.raise_for_status()
            return StreamingResponse(
                io.BytesIO(r.content), media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"TTS error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"TTS API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(500, "TTS generation failed")

@app.get("/tts/voices")
async def list_tts_voices():
    return JSONResponse({"voices": [
        {"id": "alloy", "name": "Alloy", "description": "Balanced and neutral"},
        {"id": "echo", "name": "Echo", "description": "Clear and focused"},
        {"id": "fable", "name": "Fable", "description": "Warm and expressive"},
        {"id": "onyx", "name": "Onyx", "description": "Deep and authoritative"},
        {"id": "nova", "name": "Nova", "description": "Bright and energetic"},
        {"id": "shimmer", "name": "Shimmer", "description": "Soft and gentle"},
    ]})


# =========================
# STT (Groq Whisper Large v3)
# =========================
@app.post("/stt")
async def speech_to_text(req: Request, res: Response, file: UploadFile = File(...)):
    user = await get_user_with_auth(req, res)
    if not GROQ_API_KEY:
        raise HTTPException(500, "STT not configured")
    audio_bytes = b""
    while chunk := await file.read(1024 * 1024):
        audio_bytes += chunk
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(413, "Audio file too large")
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=get_groq_headers_multipart(),
                files={"file": (file.filename or "audio.wav", audio_bytes)},
                data={"model": GROQ_STT_MODEL, "response_format": "json", "language": "en"}
            )
            r.raise_for_status()
            data = r.json()
            return JSONResponse({"text": data.get("text", ""), "language": data.get("language", "en")})
    except httpx.HTTPStatusError as e:
        logger.error(f"STT error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"STT API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(500, "Speech transcription failed")


# =========================
# INTENT ANALYSIS
# =========================
@app.post("/analyze-intent")
async def analyze_intent_endpoint(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        raise HTTPException(400, "Prompt required")
    intent = detect_intent(prompt)
    return {
        "intent": intent.to_dict() if intent else None,
        "action_type": _detector.get_action_type(prompt),
        "confidence": intent.confidence if intent else 0.0
    }


# =========================
# FILE TYPES INFO
# =========================
@app.get("/file-types")
async def get_supported_file_types():
    return {
        "code": sorted(list(CODE_EXTENSIONS)),
        "document": sorted(list(DOCUMENT_EXTENSIONS)),
        "data": sorted(list(DATA_EXTENSIONS)),
        "image": sorted(list(IMAGE_EXTENSIONS)),
        "audio": sorted(list(AUDIO_EXTENSIONS)),
        "video": sorted(list(VIDEO_EXTENSIONS)),
        "archive": sorted(list(ARCHIVE_EXTENSIONS)),
        "limits": {
            "max_file_size": format_file_size(MAX_FILE_SIZE),
            "max_zip_size": format_file_size(MAX_ZIP_SIZE),
            "max_zip_entries": MAX_ZIP_ENTRIES,
            "max_extracted_size": format_file_size(MAX_EXTRACTED_SIZE),
            "max_text_length": format_file_size(MAX_TEXT_LENGTH)
        }
    }


# =========================
# STARTUP
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
