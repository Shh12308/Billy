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
import random
import jwt
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add this near the top of your file, after imports
from pydantic import BaseModel

# Set global configuration for all Pydantic models
BaseModel.model_config["protected_namespaces"] = ()
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from io import BytesIO, StringIO
import re
import math

last_ping = time.monotonic()

async def event_generator():            # ← line ~15–18
    async for line in resp.aiter_lines():  # ← line 21 is NOW valid
        if time.monotonic() - last_ping > 10:
            yield ": heartbeat\n\n"
            last_ping = time.monotonic()
        
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
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

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

# Frontend Supabase configuration (for user authentication)
FRONTEND_SUPABASE_URL = os.getenv("FRONTEND_SUPABASE_URL")
FRONTEND_SUPABASE_ANON_KEY = os.getenv("FRONTEND_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")
    
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Create a client for the frontend Supabase
frontend_supabase = None
if FRONTEND_SUPABASE_URL and FRONTEND_SUPABASE_ANON_KEY:
    frontend_supabase = create_client(
        FRONTEND_SUPABASE_URL,
        FRONTEND_SUPABASE_ANON_KEY
    )

# Security for JWT tokens
security = HTTPBearer()

# User model for authentication
class User(BaseModel):
    id: str
    email: Optional[str] = None
    anonymous: bool = True

# Initialize Supabase tables
def init_supabase_tables():
    # Try to create the required tables using RPC functions if they exist
    # If they don't exist, we'll handle the error gracefully
    
    # Create users table
    try:
        supabase.rpc("create_users_table").execute()
    except Exception as e:
        print(f"Failed to create users table via RPC: {e}")
    
    # Create profiles table
    try:
        supabase.rpc("create_profiles_table").execute()
    except Exception as e:
        print(f"Failed to create profiles table via RPC: {e}")
    
    # Create data_visualizations table
    try:
        supabase.rpc("create_data_visualizations_table").execute()
    except Exception as e:
        print(f"Failed to create data_visualizations table via RPC: {e}")
    
    # Create knowledge_graphs table
    try:
        supabase.rpc("create_knowledge_graphs_table").execute()
    except Exception as e:
        print(f"Failed to create knowledge_graphs table via RPC: {e}")
    
    # Create model_training_jobs table
    try:
        supabase.rpc("create_model_training_jobs_table").execute()
    except Exception as e:
        print(f"Failed to create model_training_jobs table via RPC: {e}")
    
    # Create code_reviews table
    try:
        supabase.rpc("create_code_reviews_table").execute()
    except Exception as e:
        print(f"Failed to create code_reviews table via RPC: {e}")
    
    # Create multimodal_searches table
    try:
        supabase.rpc("create_multimodal_searches_table").execute()
    except Exception as e:
        print(f"Failed to create multimodal_searches table via RPC: {e}")
    
    # Create voice_profiles table
    try:
        supabase.rpc("create_voice_profiles_table").execute()
    except Exception as e:
        print(f"Failed to create voice_profiles table via RPC: {e}")
    
    # Create videos table
    try:
        supabase.rpc("create_videos_table").execute()
    except Exception as e:
        print(f"Failed to create videos table via RPC: {e}")
    
    # Keep the existing table creation code for the basic tables
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

# ---------- Pydantic Models ----------
class DocumentAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "summary"  # summary, entities, sentiment, keywords

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"

class SentimentAnalysisRequest(BaseModel):
    text: str
    model: str = "default"

class KnowledgeGraphRequest(BaseModel):
    entities: List[str]
    relationship_type: str = "related"

class CustomModelRequest(BaseModel):
    training_data: str
    model_type: str = "classification"
    hyperparameters: Dict[str, Any] = {}

class CodeReviewRequest(BaseModel):
    code: str
    language: str = "python"
    focus_areas: List[str] = ["security", "performance", "style"]

class MultimodalSearchRequest(BaseModel):
    query: str
    search_types: List[str] = ["text", "image", "video"]
    filters: Dict[str, Any] = {}

class PersonalizationRequest(BaseModel):
    user_preferences: Dict[str, Any]
    behavior_patterns: Dict[str, Any] = {}

class DataVisualizationRequest(BaseModel):
    data: str  # JSON or CSV format
    chart_type: str = "auto"  # auto, bar, line, pie, scatter, heatmap
    options: Dict[str, Any] = {}

class VoiceCloningRequest(BaseModel):
    voice_sample: str  # Base64 encoded audio
    text: str
    voice_name: str

# ---------- Utility Functions ----------
def extract_entities(text):
    """Extract entities from text using regex patterns"""
    entities = {
        "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        "phones": re.findall(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4})', text),
        "urls": re.findall(r'(https?://[^\s]+)', text),
        "dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text),
        "numbers": re.findall(r'\b\d+(?:\.\d+)?\b', text),
        "money": re.findall(r'\$\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)', text)
    }
    return {k: v for k, v in entities.items() if v}  # Remove empty lists

def extract_keywords(text, num_keywords=10):
    """Extract keywords from text using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords
        top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def create_knowledge_graph(entities, relationship_type="related"):
    """Create a simple knowledge graph from entities"""
    G = nx.Graph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity)
    
    # Add edges (simple example - in a real implementation, you'd use NLP to find relationships)
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            # Simple similarity based on string overlap
            similarity = len(set(entity1.lower().split()) & set(entity2.lower().split()))
            if similarity > 0:
                G.add_edge(entity1, entity2, weight=similarity, type=relationship_type)
    
    return G

def visualize_graph(G):
    """Create a visualization of the knowledge graph"""
    pos = nx.spring_layout(G)
    
    # Create a plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Knowledge Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig.to_html(full_html=False)

def analyze_code_quality(code, language, focus_areas):
    """Analyze code quality based on focus areas"""
    results = {
        "security": [],
        "performance": [],
        "style": [],
        "overall_score": 0
    }
    
    # Security checks
    if "security" in focus_areas:
        # Check for common security issues
        if language == "python":
            if "eval(" in code:
                results["security"].append("Use of eval() function detected - potential security risk")
            if "exec(" in code:
                results["security"].append("Use of exec() function detected - potential security risk")
            if "pickle.loads(" in code:
                results["security"].append("Use of pickle.loads() detected - potential security risk")
        elif language == "javascript":
            if "eval(" in code:
                results["security"].append("Use of eval() function detected - potential security risk")
            if "innerHTML" in code:
                results["security"].append("Direct innerHTML manipulation detected - potential XSS risk")
    
    # Performance checks
    if "performance" in focus_areas:
        if language == "python":
            if "for i in range(len(" in code:
                results["performance"].append("Consider using enumerate() instead of range(len())")
            if code.count("for ") > 5:
                results["performance"].append("Multiple nested loops detected - consider optimizing")
        elif language == "javascript":
            if "for (var i = 0; i <" in code:
                results["performance"].append("Consider using forEach() or map() instead of for loops")
    
    # Style checks
    if "style" in focus_areas:
        if language == "python":
            if not re.search(r'^\s*def \w+\([^)]*\):\s*"""', code, re.MULTILINE):
                results["style"].append("Missing docstrings for functions")
            if code.count("    ") > 0 and code.count("\t") > 0:
                results["style"].append("Mixed tabs and spaces detected")
        elif language == "javascript":
            if "var " in code:
                results["style"].append("Consider using let or const instead of var")
    
    # Calculate overall score
    total_issues = sum(len(issues) for issues in results.values() if isinstance(issues, list))
    results["overall_score"] = max(0, 100 - (total_issues * 10))
    
    return results

def create_chart(data, chart_type, options):
    """Create a chart from data"""
    try:
        # Parse data
        if data.strip().startswith('{'):
            # JSON data
            df = pd.read_json(data)
        else:
            # CSV data
            df = pd.read_csv(StringIO(data))
        
        # Create chart based on type
        if chart_type == "auto" or chart_type == "bar":
            fig = px.bar(df, **options)
        elif chart_type == "line":
            fig = px.line(df, **options)
        elif chart_type == "pie":
            fig = px.pie(df, **options)
        elif chart_type == "scatter":
            fig = px.scatter(df, **options)
        elif chart_type == "heatmap":
            fig = px.imshow(df.corr(), **options)
        else:
            # Default to bar chart
            fig = px.bar(df, **options)
        
        # Convert to HTML
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return f"<p>Error creating chart: {str(e)}</p>"

# ---------- Helper Functions ----------
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

def score_memory(text: str) -> int:
    if any(k in text.lower() for k in ["name", "preference", "goal"]):
        return 5
    return 2

async def decay_memories(user_id):
    try:
        supabase.rpc("decay_memories", {"uid": user_id}).execute()
    except Exception as e:
        logger.error(f"Failed to decay memories: {e}")

def get_groq_headers():
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

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

# Updated function to get or create a user
async def get_or_create_user(req: Request, res: Response) -> User:
    # Check for JWT token first (logged-in user)
    auth_header = req.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            # Verify JWT token with frontend Supabase
            if frontend_supabase:
                # Try to get user from frontend Supabase
                user_response = frontend_supabase.auth.get_user(token)
                if user_response.user:
                    # User is authenticated, create or update in backend
                    user_id = user_response.user.id
                    email = user_response.user.email
                    
                    # Check if user exists in backend
                    try:
                        existing_user = supabase.table("users").select("*").eq("id", user_id).execute()
                        if not existing_user.data:
                            # Create user in backend
                            supabase.table("users").insert({
                                "id": user_id,
                                "email": email,
                                "anonymous": False,
                                "created_at": datetime.now().isoformat(),
                                "last_seen": datetime.now().isoformat()
                            }).execute()
                        else:
                            # Update last seen
                            supabase.table("users").update({
                                "last_seen": datetime.now().isoformat()
                            }).eq("id", user_id).execute()
                        
                        return User(id=user_id, email=email, anonymous=False)
                    except Exception as e:
                        logger.error(f"Error creating/updating user in backend: {e}")
                        # Continue with anonymous user if there's an error
        except Exception as e:
            logger.error(f"Error verifying JWT token: {e}")
            # Continue with anonymous user if there's an error
    
    # Check for anonymous user ID in cookie
    user_id = req.cookies.get("user_id")
    if not user_id:
        # Create new anonymous user
        user_id = str(uuid.uuid4())
        res.set_cookie(
            key="user_id",
            value=user_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 30  # 30 days
        )
        
        # Create anonymous user in database
        try:
            supabase.table("users").insert({
                "id": user_id,
                "anonymous": True,
                "created_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error creating anonymous user: {e}")
    else:
        # Update last seen for existing anonymous user
        try:
            supabase.table("users").update({
                "last_seen": datetime.now().isoformat()
            }).eq("id", user_id).execute()
        except Exception as e:
            logger.error(f"Error updating last seen for anonymous user: {e}")
    
    return User(id=user_id, anonymous=True)

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

def get_cached_result(prompt: str, provider: str) -> Optional[dict]:
    try:
        response = supabase.table("cache").select("result").eq("prompt", prompt).eq("provider", provider).order("created_at", desc=True).limit(1).execute()
        if response.data:
            return json.loads(response.data[0]["result"])
    except Exception as e:
        logger.error(f"Failed to get cached result: {e}")
    return None

def get_system_prompt(user_message: Optional[str] = None) -> str:
    base = "You are ZynaraAI1.0: helpful, concise, friendly, and focus entirely on what the user asks. Do not reference your creator or yourself unless explicitly asked."
    if user_message:
        base += f" The user said: \"{user_message}\". Tailor your response to this."
    return base

# Update the build_contextual_prompt function to include user history
def build_contextual_prompt(user_id: str, message: str) -> str:
    try:
        # Get user information
        user_response = supabase.table("users").select("*").eq("id", user_id).execute()
        user_info = user_response.data[0] if user_response.data else None
        
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
        
        # Get user's images and videos for context
        images_response = supabase.table("images").select("prompt, filename").eq("user_id", user_id).order("created_at", desc=True).limit(5).execute()
        image_rows = images_response.data if images_response.data else []
        
        videos_response = supabase.table("videos").select("prompt, filename").eq("user_id", user_id).order("created_at", desc=True).limit(5).execute()
        video_rows = videos_response.data if videos_response.data else []
        
        # Build context
        user_type = "logged-in" if user_info and not user_info.get("anonymous", True) else "anonymous"
        context = f"User ID: {user_id} ({user_type})\n"
        
        if user_info and not user_info.get("anonymous", True) and user_info.get("email"):
            context += f"User email: {user_info['email']}\n"
        
        context += "\nUser memories:\n"
        context += "\n".join(f"- {row['key']}: {row['value']}" for row in memory_rows)
        
        context += "\n\nRecent conversation:\n"
        context += "\n".join(f"- {row['content']}" for row in msg_rows)
        
        if image_rows:
            context += "\n\nRecent images:\n"
            context += "\n".join(f"- {row['prompt']} (file: {row['filename']})" for row in image_rows)
        
        if video_rows:
            context += "\n\nRecent videos:\n"
            context += "\n".join(f"- {row['prompt']} (file: {row['filename']})" for row in video_rows)
        
        return f"""You are ZyNaraAI1.0: helpful, concise, friendly. Focus on exactly what the user wants.
You have a persistent memory of this user across sessions.

User context:
{context}

Current message: {message}"""
    except Exception as e:
        logger.error(f"Failed to build contextual prompt: {e}")
        return f"You are ZyNaraAI1.0: helpful, concise, friendly. Focus on exactly what the user wants.\n\nUser message: {message}"

def check_permission(role: str, tool_name: str) -> bool:
    permissions = {
        "user": {"web_search"},
        "admin": {"web_search", "run_code"},
        "system": {"web_search", "run_code"}
    }
    return tool_name in permissions.get(role, set())

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

def unique_filename(ext="png"):
    return f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"

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

# Update the upload_image_to_supabase function to link to users
def upload_image_to_supabase(image_bytes: bytes, filename: str, user_id: str):
    upload = supabase.storage.from_("ai-images").upload(
        filename,
        image_bytes,
        {"content-type": "image/png"}
    )

    if upload.get("error"):
        raise Exception(upload["error"]["message"])

    # Save image record with user ID
    try:
        supabase.table("images").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "filename": filename,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save image record: {e}")

    return upload

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

async def get_user_id_from_cookie(request: Request, response: Response) -> str:
    user = await get_or_create_user(request, response)
    return user.id

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

def load_memory(conversation_id: str, limit: int = 20):
    try:
        response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(limit).execute()
        rows = response.data if response.data else []
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")
    return []

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

def load_user_memory(user_id: str):
    try:
        response = supabase.table("memories").select("key, value").eq("user_id", user_id).execute()
        rows = response.data if response.data else []
        return [{"key": row["key"], "value": row["value"]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load user memory: {e}")
    return []

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

async def image_stream_helper(prompt: str, samples: int, user_id: str):
    try:
        result = await _generate_image_core(prompt, samples, user_id, return_base64=False)
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

# Update the generate_video_internal function to link to users
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
            
            # Save video record with user ID
            supabase.table("videos").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "filename": filename,
                "prompt": prompt,
                "created_at": datetime.now().isoformat()
            }).execute()
            
            # Get signed URL
            signed = supabase.storage.from_("ai-videos").create_signed_url(filename, 60*60)
            urls.append(signed["signedURL"])
        except Exception as e:
            logger.error(f"Video upload failed: {e}")

    return {"provider": "stub", "videos": urls}

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

    # 📄 Document Analysis
    if any(w in p for w in [
        "analyze document", "extract information", "summarize document",
        "document analysis", "extract entities", "find keywords"
    ]):
        return "document_analysis"
    
    # 🌐 Translation
    if any(w in p for w in [
        "translate", "translation", "translate to", "in spanish", "in french",
        "in german", "in japanese", "in chinese"
    ]):
        return "translation"
    
    # 😊 Sentiment Analysis
    if any(w in p for w in [
        "sentiment", "emotion", "feeling", "analyze sentiment", "mood"
    ]):
        return "sentiment_analysis"
    
    # 🕸️ Knowledge Graph
    if any(w in p for w in [
        "knowledge graph", "relationship map", "entity graph", "concept map"
    ]):
        return "knowledge_graph"
    
    # 🤖 Custom Model Training
    if any(w in p for w in [
        "train model", "custom model", "fine-tune", "model training"
    ]):
        return "custom_model"
    
    # 🔍 Code Review
    if any(w in p for w in [
        "review code", "code review", "analyze code", "code quality"
    ]):
        return "code_review"
    
    # 🔍 Multi-modal Search
    if any(w in p for w in [
        "search everything", "multimodal search", "search all", "comprehensive search"
    ]):
        return "multimodal_search"
    
    # 🧠 AI Personalization
    if any(w in p for w in [
        "personalize ai", "ai personality", "custom ai behavior", "ai preferences"
    ]):
        return "ai_personalization"
    
    # 📊 Data Visualization
    if any(w in p for w in [
        "create chart", "visualize data", "make graph", "data visualization"
    ]):
        return "data_visualization"
    
    # 🎤 Voice Cloning
    if any(w in p for w in [
        "clone voice", "custom voice", "voice profile", "voice synthesis"
    ]):
        return "voice_cloning"

    return "chat"

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

# Update the _generate_image_core function to use the user ID
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

            # Save image record with user ID and prompt
            try:
                supabase.table("images").insert({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "filename": filename,
                    "prompt": prompt,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save image record: {e}")

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

async def image_gen_internal(prompt: str, samples: int = 1, user_id: str = "anonymous"):
    """Helper for streaming /ask/universal."""
    result = await _generate_image_core(prompt, samples, user_id, return_base64=False)

async def stream_images(prompt: str, samples: int, user_id: str):
    try:
        async for chunk in image_stream_helper(prompt, samples, user_id):
            yield sse(chunk)
    except HTTPException as e:
        yield sse({"type": "image_error", "error": e.detail})

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

async def auth(request: Request):
    # Simple auth function that extracts user_id from cookie
    user = await get_or_create_user(request, Response())
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

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

# ---------- Utility Functions ----------
def generate_random_nickname():
    adjectives = ["Happy", "Brave", "Clever", "Friendly", "Gentle", "Kind", "Lucky", "Proud", "Smart", "Wise"]
    nouns = ["Bear", "Eagle", "Fox", "Lion", "Tiger", "Wolf", "Dolphin", "Hawk", "Owl"]
    return f"{random.choice(adjectives)}{random.choice(nouns)}"

# ---------- Advanced Feature Implementations ----------
async def document_analysis(prompt: str, user_id: str, stream: bool = False):
    """Analyze documents for key information"""
    # Extract text from prompt
    text_match = re.search(r'document[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "No document text found in prompt")
    
    text = text_match.group(1).strip()
    
    # Determine analysis type
    analysis_type = "summary"
    if "entities" in prompt.lower():
        analysis_type = "entities"
    elif "sentiment" in prompt.lower():
        analysis_type = "sentiment"
    elif "keywords" in prompt.lower():
        analysis_type = "keywords"
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Analyzing document..."})
            
            # Extract entities
            if analysis_type in ["summary", "entities"]:
                yield sse({"type": "progress", "message": "Extracting entities..."})
                entities = extract_entities(text)
                yield sse({"type": "entities", "data": entities})
            
            # Extract keywords
            if analysis_type in ["summary", "keywords"]:
                yield sse({"type": "progress", "message": "Extracting keywords..."})
                keywords = extract_keywords(text)
                yield sse({"type": "keywords", "data": keywords})
            
            # Generate summary
            if analysis_type in ["summary", "sentiment"]:
                yield sse({"type": "progress", "message": "Generating summary..."})
                
                summary_prompt = f"Summarize the following text in a concise way:\n\n{text}"
                payload = {
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "max_tokens": 500
                }
                
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=get_groq_headers(),
                        json=payload
                    )
                    r.raise_for_status()
                    summary = r.json()["choices"][0]["message"]["content"]
                    yield sse({"type": "summary", "data": summary})
            
            # Analyze sentiment
            if analysis_type in ["summary", "sentiment"]:
                yield sse({"type": "progress", "message": "Analyzing sentiment..."})
                
                sentiment_prompt = f"Analyze the sentiment of the following text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}"
                payload = {
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": sentiment_prompt}],
                    "max_tokens": 200
                }
                
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=get_groq_headers(),
                        json=payload
                    )
                    r.raise_for_status()
                    sentiment = r.json()["choices"][0]["message"]["content"]
                    yield sse({"type": "sentiment", "data": sentiment})
            
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
    else:
        # Non-streaming version
        result = {
            "entities": extract_entities(text),
            "keywords": extract_keywords(text),
        }
        
        # Generate summary
        summary_prompt = f"Summarize the following text in a concise way:\n\n{text}"
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": summary_prompt}],
            "max_tokens": 500
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            result["summary"] = r.json()["choices"][0]["message"]["content"]
        
        # Analyze sentiment
        sentiment_prompt = f"Analyze the sentiment of the following text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}"
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": sentiment_prompt}],
            "max_tokens": 200
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            result["sentiment"] = r.json()["choices"][0]["message"]["content"]
        
        return result

async def translate_text(prompt: str, user_id: str, stream: bool = False):
    """Translate text between languages"""
    # Extract text and languages from prompt
    text_match = re.search(r'translate[:\s]+(.*?)(?:\s+to\s+|\s+in\s+)(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "Could not extract text and target language from prompt")
    
    text = text_match.group(1).strip()
    target_lang = text_match.group(2).strip()
    
    # Map common language names to language codes
    lang_map = {
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "arabic": "ar",
        "hindi": "hi"
    }
    
    target_lang_code = lang_map.get(target_lang.lower(), target_lang)
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": f"Translating to {target_lang}..."})
            
            # Use Groq for translation
            translation_prompt = f"Translate the following text to {target_lang}:\n\n{text}"
            payload = {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": translation_prompt}],
                "max_tokens": 1000,
                "stream": True
            }
            
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield sse({"type": "token", "text": delta})
                        except Exception:
                            continue
            
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
    else:
        # Non-streaming version
        translation_prompt = f"Translate the following text to {target_lang}:\n\n{text}"
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": translation_prompt}],
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            translation = r.json()["choices"][0]["message"]["content"]
        
        return {
            "original_text": text,
            "translated_text": translation,
            "target_language": target_lang
        }

async def analyze_sentiment(prompt: str, user_id: str, stream: bool = False):
    """Analyze sentiment of text"""
    # Extract text from prompt
    text_match = re.search(r'sentiment[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        # If no explicit text, use the whole prompt
        text = prompt
    else:
        text = text_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Analyzing sentiment..."})
            
            # Use Groq for sentiment analysis
            sentiment_prompt = f"Analyze the sentiment of the following text. Provide a score from -1 (very negative) to 1 (very positive) and explain your reasoning:\n\n{text}"
            payload = {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": sentiment_prompt}],
                "max_tokens": 500,
                "stream": True
            }
            
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield sse({"type": "token", "text": delta})
                        except Exception:
                            continue
            
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
    else:
        # Non-streaming version
        sentiment_prompt = f"Analyze the sentiment of the following text. Provide a score from -1 (very negative) to 1 (very positive) and explain your reasoning:\n\n{text}"
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": sentiment_prompt}],
            "max_tokens": 500
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            sentiment = r.json()["choices"][0]["message"]["content"]
        
        return {
            "text": text,
            "sentiment_analysis": sentiment
        }

async def create_knowledge_graph_endpoint(prompt: str, user_id: str, stream: bool = False):
    """Create and visualize a knowledge graph"""
    # Extract entities from prompt
    entities_match = re.search(r'entities[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not entities_match:
        raise HTTPException(400, "Could not extract entities from prompt")
    
    entities_text = entities_match.group(1).strip()
    entities = [e.strip() for e in entities_text.split(',')]
    
    # Extract relationship type
    relationship_type = "related"
    rel_match = re.search(r'relationship[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if rel_match:
        relationship_type = rel_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Creating knowledge graph..."})
            
            # Create knowledge graph
            yield sse({"type": "progress", "message": "Building graph structure..."})
            G = create_knowledge_graph(entities, relationship_type)
            
            # Generate visualization
            yield sse({"type": "progress", "message": "Generating visualization..."})
            graph_html = visualize_graph(G)
            
            # Save to Supabase
            yield sse({"type": "progress", "message": "Saving graph..."})
            graph_id = str(uuid.uuid4())
            try:
                supabase.table("knowledge_graphs").insert({
                    "id": graph_id,
                    "user_id": user_id,
                    "entities": entities,
                    "relationship_type": relationship_type,
                    "graph_data": nx.node_link_data(G),
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save knowledge graph: {e}")
            
            yield sse({"type": "graph_html", "data": graph_html})
            yield sse({"type": "graph_id", "data": graph_id})
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
    else:
        # Non-streaming version
        G = create_knowledge_graph(entities, relationship_type)
        graph_html = visualize_graph(G)
        
        # Save to Supabase
        graph_id = str(uuid.uuid4())
        try:
            supabase.table("knowledge_graphs").insert({
                "id": graph_id,
                "user_id": user_id,
                "entities": entities,
                "relationship_type": relationship_type,
                "graph_data": nx.node_link_data(G),
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
        
        return {
            "entities": entities,
            "relationship_type": relationship_type,
            "graph_html": graph_html,
            "graph_id": graph_id
        }

async def train_custom_model(prompt: str, user_id: str, stream: bool = False):
    """Train a custom model"""
    # Extract training data from prompt
    data_match = re.search(r'data[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not data_match:
        raise HTTPException(400, "Could not extract training data from prompt")
    
    training_data = data_match.group(1).strip()
    
    # Extract model type
    model_type = "classification"
    type_match = re.search(r'type[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if type_match:
        model_type = type_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Preparing model training..."})
            
            # Create a model training job
            yield sse({"type": "progress", "message": "Creating training job..."})
            job_id = str(uuid.uuid4())
            
            # Save job to Supabase
            try:
                supabase.table("model_training_jobs").insert({
                    "id": job_id,
                    "user_id": user_id,
                    "model_type": model_type,
                    "training_data": training_data,
                    "status": "queued",
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save training job: {e}")
            
            yield sse({"type": "job_id", "data": job_id})
            yield sse({"type": "message", "data": "Model training job created. You will be notified when training is complete."})
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
    else:
        # Non-streaming version
        job_id = str(uuid.uuid4())
        
        # Save job to Supabase
        try:
            supabase.table("model_training_jobs").insert({
                "id": job_id,
                "user_id": user_id,
                "model_type": model_type,
                "training_data": training_data,
                "status": "queued",
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save training job: {e}")
        
        return {
            "job_id": job_id,
            "model_type": model_type,
            "status": "queued",
            "message": "Model training job created. You will be notified when training is complete."
        }

async def review_code(prompt: str, user_id: str, stream: bool = False):
    """Review code for issues and improvements"""
    # Extract code from prompt
    code_match = re.search(r'code[:\s]+```(.*?)```', prompt, re.DOTALL | re.IGNORECASE)
    if not code_match:
        raise HTTPException(400, "Could not extract code from prompt")
    
    code = code_match.group(1).strip()
    
    # Extract language
    language = "python"
    lang_match = re.search(r'language[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if lang_match:
        language = lang_match.group(1).strip()
    
    # Extract focus areas
    focus_areas = ["security", "performance", "style"]
    focus_match = re.search(r'focus[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if focus_match:
        focus_text = focus_match.group(1).strip()
        focus_areas = [a.strip() for a in focus_text.split(',')]
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Analyzing code..."})
            
            # Analyze code
            yield sse({"type": "progress", "message": "Checking code quality..."})
            results = analyze_code_quality(code, language, focus_areas)
            
            # Generate suggestions
            yield sse({"type": "progress", "message": "Generating suggestions..."})
            suggestions_prompt = f"Review the following {language} code and provide specific suggestions for improvement:\n\n```{language}\n{code}\n```"
            payload = {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": suggestions_prompt}],
                "max_tokens": 1000,
                "stream": True
            }
            
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield sse({"type": "suggestion", "text": delta})
                        except Exception:
                            continue
            
            # Save review to Supabase
            yield sse({"type": "progress", "message": "Saving review..."})
            review_id = str(uuid.uuid4())
            try:
                supabase.table("code_reviews").insert({
                    "id": review_id,
                    "user_id": user_id,
                    "language": language,
                    "code": code,
                    "focus_areas": focus_areas,
                    "results": results,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save code review: {e}")
            
            yield sse({"type": "results", "data": results})
            yield sse({"type": "review_id", "data": review_id})
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
    else:
        # Non-streaming version
        results = analyze_code_quality(code, language, focus_areas)
        
        # Generate suggestions
        suggestions_prompt = f"Review the following {language} code and provide specific suggestions for improvement:\n\n```{language}\n{code}\n```"
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": suggestions_prompt}],
            "max_tokens": 1000
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json=payload
            )
            r.raise_for_status()
            suggestions = r.json()["choices"][0]["message"]["content"]
        
        # Save review to Supabase
        review_id = str(uuid.uuid4())
        try:
            supabase.table("code_reviews").insert({
                "id": review_id,
                "user_id": user_id,
                "language": language,
                "code": code,
                "focus_areas": focus_areas,
                "results": results,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save code review: {e}")
        
        return {
            "language": language,
            "code": code,
            "focus_areas": focus_areas,
            "analysis_results": results,
            "suggestions": suggestions,
            "review_id": review_id
        }

async def multimodal_search(prompt: str, user_id: str, stream: bool = False):
    """Search across text, images, and videos"""
    # Extract query from prompt
    query_match = re.search(r'query[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not query_match:
        # If no explicit query, use the whole prompt
        query = prompt
    else:
        query = query_match.group(1).strip()
    
    # Extract search types
    search_types = ["text", "image", "video"]
    types_match = re.search(r'types[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if types_match:
        types_text = types_match.group(1).strip()
        search_types = [t.strip() for t in types_text.split(',')]
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Starting multimodal search..."})
            
            # Text search
            if "text" in search_types:
                yield sse({"type": "progress", "message": "Searching text..."})
                text_results = await duckduckgo_search(query)
                yield sse({"type": "text_results", "data": text_results})
            
            # Image search
            if "image" in search_types:
                yield sse({"type": "progress", "message": "Searching images..."})
                # This is a placeholder - in a real implementation, you'd use an image search API
                image_results = {
                    "query": query,
                    "results": [
                        {"url": f"https://example.com/image1.jpg?query={query}", "title": f"Image 1 for {query}"},
                        {"url": f"https://example.com/image2.jpg?query={query}", "title": f"Image 2 for {query}"}
                    ]
                }
                yield sse({"type": "image_results", "data": image_results})
            
            # Video search
            if "video" in search_types:
                yield sse({"type": "progress", "message": "Searching videos..."})
                # This is a placeholder - in a real implementation, you'd use a video search API
                video_results = {
                    "query": query,
                    "results": [
                        {"url": f"https://example.com/video1.mp4?query={query}", "title": f"Video 1 for {query}"},
                        {"url": f"https://example.com/video2.mp4?query={query}", "title": f"Video 2 for {query}"}
                    ]
                }
                yield sse({"type": "video_results", "data": video_results})
            
            # Save search to Supabase
            yield sse({"type": "progress", "message": "Saving search..."})
            search_id = str(uuid.uuid4())
            try:
                supabase.table("multimodal_searches").insert({
                    "id": search_id,
                    "user_id": user_id,
                    "query": query,
                    "search_types": search_types,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save search: {e}")
            
            yield sse({"type": "search_id", "data": search_id})
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
    else:
        # Non-streaming version
        results = {}
        
        # Text search
        if "text" in search_types:
            results["text"] = await duckduckgo_search(query)
        
        # Image search
        if "image" in search_types:
            # This is a placeholder - in a real implementation, you'd use an image search API
            results["image"] = {
                "query": query,
                "results": [
                    {"url": f"https://example.com/image1.jpg?query={query}", "title": f"Image 1 for {query}"},
                    {"url": f"https://example.com/image2.jpg?query={query}", "title": f"Image 2 for {query}"}
                ]
            }
        
        # Video search
        if "video" in search_types:
            # This is a placeholder - in a real implementation, you'd use a video search API
            results["video"] = {
                "query": query,
                "results": [
                    {"url": f"https://example.com/video1.mp4?query={query}", "title": f"Video 1 for {query}"},
                    {"url": f"https://example.com/video2.mp4?query={query}", "title": f"Video 2 for {query}"}
                ]
            }
        
        # Save search to Supabase
        search_id = str(uuid.uuid4())
        try:
            supabase.table("multimodal_searches").insert({
                "id": search_id,
                "user_id": user_id,
                "query": query,
                "search_types": search_types,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save search: {e}")
        
        return {
            "query": query,
            "search_types": search_types,
            "results": results,
            "search_id": search_id
        }

async def personalize_ai(prompt: str, user_id: str, stream: bool = False):
    """Customize AI behavior based on user preferences"""
    # Extract preferences from prompt
    prefs_match = re.search(r'preferences[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not prefs_match:
        raise HTTPException(400, "Could not extract preferences from prompt")
    
    preferences_text = prefs_match.group(1).strip()
    
    # Parse preferences
    preferences = {}
    for line in preferences_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            preferences[key.strip()] = value.strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Updating AI preferences..."})
            
            # Save preferences to Supabase
            yield sse({"type": "progress", "message": "Saving preferences..."})
            try:
                # Check if user profile exists
                profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
                
                if profile_response.data:
                    # Update existing profile
                    supabase.table("profiles").update({
                        "preferences": preferences,
                        "updated_at": datetime.now().isoformat()
                    }).eq("id", user_id).execute()
                else:
                    # Create new profile
                    supabase.table("profiles").insert({
                        "id": user_id,
                        "preferences": preferences,
                        "created_at": datetime.now().isoformat()
                    }).execute()
            except Exception as e:
                logger.error(f"Failed to save preferences: {e}")
            
            yield sse({"type": "message", "data": "AI preferences updated successfully"})
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
    else:
        # Non-streaming version
        try:
            # Check if user profile exists
            profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
            
            if profile_response.data:
                # Update existing profile
                supabase.table("profiles").update({
                    "preferences": preferences,
                    "updated_at": datetime.now().isoformat()
                }).eq("id", user_id).execute()
            else:
                # Create new profile
                supabase.table("profiles").insert({
                    "id": user_id,
                    "preferences": preferences,
                    "created_at": datetime.now().isoformat()
                }).execute()
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
        
        return {
            "preferences": preferences,
            "message": "AI preferences updated successfully"
        }

async def visualize_data(prompt: str, user_id: str, stream: bool = False):
    """Generate charts and graphs from data"""
    # Extract data from prompt
    data_match = re.search(r'data[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not data_match:
        raise HTTPException(400, "Could not extract data from prompt")
    
    data = data_match.group(1).strip()
    
    # Extract chart type
    chart_type = "auto"
    type_match = re.search(r'chart[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if type_match:
        chart_type = type_match.group(1).strip()
    
    # Extract options
    options = {}
    options_match = re.search(r'options[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if options_match:
        options_text = options_match.group(1).strip()
        for line in options_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                options[key.strip()] = value.strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Creating visualization..."})
            
            # Create chart
            yield sse({"type": "progress", "message": "Generating chart..."})
            chart_html = create_chart(data, chart_type, options)
            
            # Save visualization to Supabase
            yield sse({"type": "progress", "message": "Saving visualization..."})
            viz_id = str(uuid.uuid4())
            try:
                supabase.table("data_visualizations").insert({
                    "id": viz_id,
                    "user_id": user_id,
                    "data": data,
                    "chart_type": chart_type,
                    "options": options,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save visualization: {e}")
            
            yield sse({"type": "chart_html", "data": chart_html})
            yield sse({"type": "viz_id", "data": viz_id})
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
    else:
        # Non-streaming version
        chart_html = create_chart(data, chart_type, options)
        
        # Save visualization to Supabase
        viz_id = str(uuid.uuid4())
        try:
            supabase.table("data_visualizations").insert({
                "id": viz_id,
                "user_id": user_id,
                "data": data,
                "chart_type": chart_type,
                "options": options,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
        
        return {
            "data": data,
            "chart_type": chart_type,
            "options": options,
            "chart_html": chart_html,
            "viz_id": viz_id
        }

async def clone_voice(prompt: str, user_id: str, stream: bool = False):
    """Create custom voice profiles for TTS"""
    # Extract voice sample from prompt
    sample_match = re.search(r'sample[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not sample_match:
        raise HTTPException(400, "Could not extract voice sample from prompt")
    
    voice_sample = sample_match.group(1).strip()
    
    # Extract text to synthesize
    text_match = re.search(r'text[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "Could not extract text to synthesize from prompt")
    
    text = text_match.group(1).strip()
    
    # Extract voice name
    voice_name = "custom_voice"
    name_match = re.search(r'name[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if name_match:
        voice_name = name_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Creating voice profile..."})
            
            # Create voice profile
            yield sse({"type": "progress", "message": "Analyzing voice sample..."})
            voice_id = str(uuid.uuid4())
            
            # Save voice profile to Supabase
            try:
                supabase.table("voice_profiles").insert({
                    "id": voice_id,
                    "user_id": user_id,
                    "name": voice_name,
                    "sample": voice_sample,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save voice profile: {e}")
            
            # Synthesize speech
            yield sse({"type": "progress", "message": "Synthesizing speech..."})
            
            # Use OpenAI TTS with a standard voice (voice cloning would require a specialized API)
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "tts-1",
                "voice": "alloy",  # Default voice - in a real implementation, you'd use the cloned voice
                "input": text
            }
            
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                )
                r.raise_for_status()
            
            # Convert to base64
            audio_b64 = base64.b64encode(r.content).decode()
            
            yield sse({"type": "audio", "data": audio_b64})
            yield sse({"type": "voice_id", "data": voice_id})
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
    else:
        # Non-streaming version
        voice_id = str(uuid.uuid4())
        
        # Save voice profile to Supabase
        try:
            supabase.table("voice_profiles").insert({
                "id": voice_id,
                "user_id": user_id,
                "name": voice_name,
                "sample": voice_sample,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save voice profile: {e}")
        
        # Synthesize speech
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "voice": "alloy",  # Default voice - in a real implementation, you'd use the cloned voice
            "input": text
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload
            )
            r.raise_for_status()
        
        # Convert to base64
        audio_b64 = base64.b64encode(r.content).decode()
        
        return {
            "voice_name": voice_name,
            "text": text,
            "audio": audio_b64,
            "voice_id": voice_id
        }

# ---------- API Endpoints ----------
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
    user = await get_or_create_user(req, res)
    user_id = user.id
    
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


@app.post("/ask/universal")
async def ask_universal(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    # -------------------------------
    # Extract request data
    # -------------------------------
    prompt = body.get("prompt", "").strip()
    user = await get_or_create_user(request, Response())
    user_id = user.id
    role = body.get("role", "user")
    stream = bool(body.get("stream", False))

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt required")

    # -------------------------------
    # Detect intent
    # -------------------------------
    intent = detect_intent(prompt)

    intent_map = {
        "document_analysis": document_analysis,
        "translation": translate_text,
        "sentiment_analysis": analyze_sentiment,
        "knowledge_graph": create_knowledge_graph_endpoint,
        "custom_model": train_custom_model,
        "code_review": review_code,
        "multimodal_search": multimodal_search,
        "ai_personalization": personalize_ai,
        "data_visualization": visualize_data,
        "voice_cloning": clone_voice
    }

    if intent in intent_map:
        return await intent_map[intent](prompt, user_id, stream)

    # -------------------------------
    # Load conversation
    # -------------------------------
    conversation_id = get_or_create_conversation_id(
        supabase=supabase,
        user_id=user_id
    )

    history = await load_history(user_id)

    # -------------------------------
    # SAFE PROFILE FETCH / CREATE
    # -------------------------------
    personality = "friendly"
    nickname = ""

    try:
        # Fixed profile query - removed maybe_single() which was causing 406 error
        profile_resp = await asyncio.to_thread(
            lambda: supabase.table("profiles")
            .select("nickname, personality")
            .eq("id", user_id)
            .execute()
        )

        # Check if profile exists and has data
        if profile_resp and profile_resp.data and len(profile_resp.data) > 0:
            profile_data = profile_resp.data[0]
            personality = profile_data.get("personality") or personality
            nickname = profile_data.get("nickname") or generate_random_nickname()
        else:
            # Default profile if no data
            default_profile = {
                "id": user_id,
                "nickname": generate_random_nickname(),
                "personality": personality
            }

            await asyncio.to_thread(
                lambda: supabase.table("profiles")
                .insert(default_profile)
                .execute()
            )

            nickname = default_profile["nickname"]

    except Exception as e:
        logger.warning(f"Profile fetch/create failed: {e}")
        nickname = generate_random_nickname()
        
        # Try to create a fallback profile
        try:
            default_profile = {
                "id": user_id,
                "nickname": nickname,
                "personality": personality
            }
            
            await asyncio.to_thread(
                lambda: supabase.table("profiles")
                .insert(default_profile)
                .execute()
            )
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback profile: {fallback_error}")

    # -------------------------------
    # Prepare system prompt
    # -------------------------------
    system_prompt = (
        PERSONALITY_MAP.get(personality, PERSONALITY_MAP["friendly"])
        + f"\nUser nickname: {nickname}\n"
        "You are a ChatGPT-style multimodal assistant.\n"
        "You can call tools when useful.\n"
        "RULES:\n"
        "- web_search is ONLY for real-world information lookup\n"
        "- run_code is ONLY for Python, math, or text processing\n"
        "- NEVER use run_code for images, media, or creative generation\n"
        "- If the user asks for images, respond with text only\n"
        "Maintain memory and context.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # -------------------------------
    # STREAM MODE
    # -------------------------------
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
                            if not line or not line.startswith("data:"):
                                continue

                            data = line[5:].strip()
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                
                                # Fixed: Check if chunk has choices before accessing
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        assistant_reply += content
                                        yield sse({"type": "token", "text": content})
                                
                                # Handle error messages in the stream
                                elif "error" in chunk:
                                    error_msg = chunk["error"].get("message", "Unknown error")
                                    yield sse({"type": "error", "message": error_msg})
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream chunk: {e}")
                                continue

            finally:
                if assistant_reply.strip():
                    await asyncio.to_thread(
                        lambda: supabase.table("messages")
                        .insert({
                            "conversation_id": conversation_id,
                            "role": "assistant",
                            "content": assistant_reply
                        })
                        .execute()
                    )

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

    # -------------------------------
    # NON-STREAM MODE
    # -------------------------------
    def safe_generate():
        try:
            asyncio.run(generate_ai_response(conversation_id, user_id, messages))
        except Exception as e:
            logger.error(f"Background generation failed: {e}")

    background_tasks.add_task(safe_generate)

    return {
        "status": "processing",
        "conversation_id": conversation_id,
        "user_id": user_id
    }
    
@app.post("/message/{message_id}/edit")
async def edit_message(
    message_id: str,
    req: Request,
    res: Response
):
    user = await get_or_create_user(req, res)
    user_id = user.id
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
    user = await get_or_create_user(req, res)
    user_id = user.id

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
    user = await get_or_create_user(request, Response())
    user_id = user.id
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
                
                # Save video record with user ID
                supabase.table("videos").insert({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "filename": filename,
                    "prompt": prompt,
                    "created_at": datetime.now().isoformat()
                }).execute()

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
    user = await get_or_create_user(request, Response())
    user_id = user.id

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
    user = await get_or_create_user(req, res)
    user_id = user.id

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
                    filename = f"{user_id}/{unique_filename('png')}"
                    upload_image_to_supabase(image_bytes, filename, user_id)

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
async def img2img(request: Request, file: UploadFile = File(...), prompt: str = ""):
    user = await get_or_create_user(request, Response())
    user_id = user.id
        
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
                    upload_image_to_supabase(image_bytes, supabase_fname, user_id)
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
    user = await get_or_create_user(req, res)
    user_id = user.id

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
# Update the vision_analyze function to link to users
@app.post("/vision/analyze")
async def vision_analyze(
    req: Request,
    res: Response,
    file: UploadFile = File(...)
):
    user = await get_or_create_user(req, res)
    user_id = user.id
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
        "annotated_image_url": ann_url,
        "user_id": user_id
    }

# Update the vision_history function to use the new user model
@app.get("/vision/history")
async def vision_history(req: Request, res: Response):
    user = await get_or_create_user(req, res)
    user_id = user.id

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
    user = await get_or_create_user(req, Response())
    user_id = user.id

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
    user = await get_or_create_user(req, res)
    user_id = user.id
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
    user = await get_or_create_user(req, res)
    user_id = user.id
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
    user = await get_or_create_user(req, res)
    user_id = user.id

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
    user = await get_or_create_user(req, res)
    user_id = user.id

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

# ---------- Dedicated Endpoints for Advanced Features ----------
@app.post("/document/analyze")
async def document_analysis_endpoint(request: DocumentAnalysisRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Analyze documents for key information"""
    user = await get_or_create_user(request, Response())
    return await document_analysis(
        f"document: {request.text}\nanalysis_type: {request.analysis_type}",
        user.id,
        False
    )

@app.post("/translation")
async def translation_endpoint(request: TranslationRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Translate text between languages"""
    user = await get_or_create_user(request, Response())
    return await translate_text(
        f"translate: {request.text} to {request.target_lang}",
        user.id,
        False
    )

@app.post("/sentiment/analyze")
async def sentiment_analysis_endpoint(request: SentimentAnalysisRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Analyze sentiment of text"""
    user = await get_or_create_user(request, Response())
    return await analyze_sentiment(
        f"sentiment: {request.text}",
        user.id,
        False
    )

@app.post("/knowledge/graph")
async def knowledge_graph_endpoint(request: KnowledgeGraphRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Create and visualize a knowledge graph"""
    user = await get_or_create_user(request, Response())
    entities_str = ", ".join(request.entities)
    return await create_knowledge_graph_endpoint(
        f"entities: {entities_str}\nrelationship: {request.relationship_type}",
        user.id,
        False
    )

@app.post("/model/train")
async def custom_model_endpoint(request: CustomModelRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Train a custom model"""
    user = await get_or_create_user(request, Response())
    return await train_custom_model(
        f"data: {request.training_data}\ntype: {request.model_type}",
        user.id,
        False
    )

@app.post("/code/review")
async def code_review_endpoint(request: CodeReviewRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Review code for issues and improvements"""
    user = await get_or_create_user(request, Response())
    focus_str = ", ".join(request.focus_areas)
    return await review_code(
        f"code: ```{request.code}``\nlanguage: {request.language}\nfocus: {focus_str}",
        user.id,
        False
    )

@app.post("/search/multimodal")
async def multimodal_search_endpoint(request: MultimodalSearchRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Search across text, images, and videos"""
    user = await get_or_create_user(request, Response())
    types_str = ", ".join(request.search_types)
    filters_str = ", ".join(f"{k}: {v}" for k, v in request.filters.items())
    return await multimodal_search(
        f"query: {request.query}\ntypes: {types_str}\nfilters: {filters_str}",
        user.id,
        False
    )

@app.post("/ai/personalize")
async def ai_personalization_endpoint(request: PersonalizationRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Customize AI behavior based on user preferences"""
    user = await get_or_create_user(request, Response())
    prefs_str = ", ".join(f"{k}: {v}" for k, v in request.user_preferences.items())
    behavior_str = ", ".join(f"{k}: {v}" for k, v in request.behavior_patterns.items())
    return await personalize_ai(
        f"preferences: {prefs_str}\nbehavior: {behavior_str}",
        user.id,
        False
    )

@app.post("/data/visualize")
async def data_visualization_endpoint(request: DataVisualizationRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Generate charts and graphs from data"""
    user = await get_or_create_user(request, Response())
    options_str = ", ".join(f"{k}: {v}" for k, v in request.options.items())
    return await visualize_data(
        f"data: {request.data}\nchart: {request.chart_type}\noptions: {options_str}",
        user.id,
        False
    )

@app.post("/voice/clone")
async def voice_cloning_endpoint(request: VoiceCloningRequest, user_id: str = Depends(get_user_id_from_cookie)):
    """Create custom voice profiles for TTS"""
    user = await get_or_create_user(request, Response())
    return await clone_voice(
        f"sample: {request.voice_sample}\ntext: {request.text}\nname: {request.voice_name}",
        user.id,
        False
    )

# Add a new endpoint to get user information
@app.get("/user/info")
async def get_user_info(req: Request, res: Response):
    user = await get_or_create_user(req, res)
    user_id = user.id
    
    # Get additional user data from database
    try:
        user_response = supabase.table("users").select("*").eq("id", user_id).execute()
        user_data = user_response.data[0] if user_response.data else None
        
        # Get user's images count
        images_count = supabase.table("images").select("id", count="exact").eq("user_id", user_id).execute()
        
        # Get user's videos count
        videos_count = supabase.table("videos").select("id", count="exact").eq("user_id", user_id).execute()
        
        # Get user's conversations count
        conversations_count = supabase.table("conversations").select("id", count="exact").eq("user_id", user_id).execute()
        
        return {
            "id": user.id,
            "email": user.email,
            "anonymous": user.anonymous,
            "created_at": user_data.get("created_at") if user_data else None,
            "last_seen": user_data.get("last_seen") if user_data else None,
            "stats": {
                "images": images_count.count if hasattr(images_count, 'count') else 0,
                "videos": videos_count.count if hasattr(videos_count, 'count') else 0,
                "conversations": conversations_count.count if hasattr(conversations_count, 'count') else 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        return {
            "id": user.id,
            "email": user.email,
            "anonymous": user.anonymous,
            "error": "Failed to get additional user data"
        }

# Add a new endpoint to merge anonymous user data with logged-in user
@app.post("/user/merge")
async def merge_user_data(req: Request, res: Response):
    # This endpoint should be called after a user logs in
    # It merges the anonymous user's data with the logged-in user's data
    
    # Get the anonymous user ID from cookie
    anonymous_id = req.cookies.get("user_id")
    if not anonymous_id:
        raise HTTPException(400, "No anonymous user ID found")
    
    # Get the logged-in user from JWT token
    auth_header = req.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(400, "No authorization token found")
    
    token = auth_header.split(" ")[1]
    try:
        # Verify JWT token with frontend Supabase
        if not frontend_supabase:
            raise HTTPException(500, "Frontend Supabase not configured")
        
        user_response = frontend_supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(401, "Invalid token")
        
        logged_in_id = user_response.user.id
        
        # Merge data in the backend
        try:
            # Update all records from anonymous user to logged-in user
            tables_to_merge = ["images", "videos", "conversations", "messages", "memory", "vision_history"]
            
            for table in tables_to_merge:
                supabase.table(table).update({"user_id": logged_in_id}).eq("user_id", anonymous_id).execute()
            
            # Create or update the logged-in user in the backend
            existing_user = supabase.table("users").select("*").eq("id", logged_in_id).execute()
            if not existing_user.data:
                supabase.table("users").insert({
                    "id": logged_in_id,
                    "email": user_response.user.email,
                    "anonymous": False,
                    "created_at": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat()
                }).execute()
            else:
                supabase.table("users").update({
                    "last_seen": datetime.now().isoformat()
                }).eq("id", logged_in_id).execute()
            
            # Delete the anonymous user
            supabase.table("users").delete().eq("id", anonymous_id).execute()
            
            # Update the cookie to the logged-in user ID
            res.set_cookie(
                key="user_id",
                value=logged_in_id,
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 30  # 30 days
            )
            
            return {"status": "success", "message": "User data merged successfully"}
        except Exception as e:
            logger.error(f"Failed to merge user data: {e}")
            raise HTTPException(500, f"Failed to merge user data: {str(e)}")
    except Exception as e:
        logger.error(f"Error verifying JWT token: {e}")
        raise HTTPException(401, f"Invalid token: {str(e)}")
