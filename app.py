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
import hashlib
from typing import Optional, Dict, Any, List, Union
from io import BytesIO, StringIO
import re
import math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import aiohttp
import torch
from PIL import Image
from fastapi import BackgroundTasks, FastAPI, Request, Header, UploadFile, File, HTTPException, Query, Form, Depends, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from supabase import create_client
from ultralytics import YOLO
from torchvision import models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ---------- CONFIG & LOGGING ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("zynara-server")

app = FastAPI(
    title="ZyNaraAI1.0 Multimodal Server",
    redirect_slashes=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9898", "https://zynara.xyz", "https://www.zynara.xyz"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Set global configuration for all Pydantic models
BaseModel.model_config["protected_namespaces"] = ()

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
class User:
    def __init__(
        self,
        id: str,
        anonymous: bool = False,
        session_token: str | None = None,
        device_fingerprint: str | None = None,
    ):
        self.id = id
        self.anonymous = anonymous
        self.session_token = session_token
        self.device_fingerprint = device_fingerprint

# -----------------------------
# Device fingerprint
# -----------------------------
def generate_device_fingerprint(request: Request) -> str:
    # simple + stable fingerprint
    return request.headers.get("user-agent", "unknown-device")

# -----------------------------
# Background task system
# -----------------------------
background_executor = ThreadPoolExecutor(max_workers=10)

class TaskManager:
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, user_id: str, task_type: str, params: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())

        task = {
            "id": task_id,
            "user_id": user_id,
            "type": task_type,
            "params": params,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "result": None,
            "error": None,
        }

        self.active_tasks[task_id] = task

        try:
            supabase.table("background_tasks").insert(
                {
                    "id": task_id,
                    "user_id": user_id,
                    "task_type": task_type,
                    "params": json.dumps(params),
                    "status": "queued",
                    "created_at": task["created_at"],
                }
            ).execute()
        except Exception as e:
            logger.error(f"Failed to persist task: {e}")

        return task_id

    def update_task_status(self, task_id: str, status: str, result=None, error=None):
        if task_id not in self.active_tasks:
            return

        task = self.active_tasks[task_id]
        task["status"] = status
        task["result"] = result
        task["error"] = error

        try:
            update_data = {"status": status}
            if result is not None:
                update_data["result"] = json.dumps(result)
            if error is not None:
                update_data["error"] = str(error)

            supabase.table("background_tasks").update(update_data).eq(
                "id", task_id
            ).execute()
        except Exception as e:
            logger.error(f"Failed to update task: {e}")

    def get_task(self, task_id: str) -> Dict[str, Any] | None:
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        try:
            resp = supabase.table("background_tasks").select("*").eq(
                "id", task_id
            ).execute()
            if resp.data:
                task = resp.data[0]
                task["params"] = json.loads(task["params"])
                if task.get("result"):
                    task["result"] = json.loads(task["result"])
                self.active_tasks[task_id] = task
                return task
        except Exception as e:
            logger.error(f"Failed to fetch task: {e}")

        return None

    def get_user_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            resp = (
                supabase.table("background_tasks")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )

            tasks = resp.data or []
            for t in tasks:
                t["params"] = json.loads(t["params"])
                if t.get("result"):
                    t["result"] = json.loads(t["result"])
            return tasks
        except Exception as e:
            logger.error(f"Failed to fetch user tasks: {e}")
            return []

# Create the global task_manager instance
task_manager = TaskManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # user_id -> websocket
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# -----------------------------
# Get or create anonymous user
# -----------------------------
async def get_or_create_user(request: Request, response: Response) -> User:
    try:
        # 1️⃣ Check for Supabase Auth user (logged-in users)
        auth_header = request.headers.get("Authorization")

        if auth_header:
            token = auth_header.replace("Bearer ", "")

            user_resp = await asyncio.to_thread(
                lambda: supabase.auth.get_user(token)
            )

            if user_resp.user:
                return User(
                    id=user_resp.user.id,
                    anonymous=False,
                    session_token=None,
                    device_fingerprint=None,
                )

    except Exception as e:
        logger.warning(f"Supabase auth check failed: {e}")

    # 2️⃣ Fallback to anonymous session
    session_token = request.cookies.get("session_token")

    if session_token:
        try:
            user_resp = await asyncio.to_thread(
                lambda: supabase.table("users")
                .select("*")
                .eq("session_token", session_token)
                .limit(1)
                .execute()
            )

            if user_resp.data:
                user_data = user_resp.data[0]

                await asyncio.to_thread(
                    lambda: supabase.table("users")
                    .update({"last_seen": datetime.utcnow().isoformat()})
                    .eq("id", user_data["id"])
                    .execute()
                )

                return User(
                    id=user_data["id"],
                    anonymous=True,
                    session_token=session_token,
                    device_fingerprint=user_data.get("device_fingerprint"),
                )

        except Exception as e:
            logger.warning(f"Anonymous lookup failed: {e}")

    # 3️⃣ Create new anonymous user
    device_fingerprint = generate_device_fingerprint(request)
    new_session_token = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    await asyncio.to_thread(
        lambda: supabase.table("users").insert({
            "id": user_id,
            "session_token": new_session_token,
            "device_fingerprint": device_fingerprint,
            "created_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat()
        }).execute()
    )

    response.set_cookie(
        key="session_token",
        value=new_session_token,
        max_age=60 * 60 * 24 * 30,
        path="/",
        secure=True,  # ✅ Use True in production
        httponly=True,
        samesite="lax"
    )

    return User(
        id=user_id,
        anonymous=True,
        session_token=new_session_token,
        device_fingerprint=device_fingerprint,
    )
        
# -----------------------------
# Merge visitor → real user
# -----------------------------
def merge_visitor_to_user(user_id: str, session_token: str):
    try:
        visitor_resp = (
            supabase.table("visitor_users")
            .select("id")
            .eq("session_token", session_token)
            .limit(1)
            .execute()
        )

        if not visitor_resp.data:
            return  # Nothing to merge

        visitor_id = visitor_resp.data[0]["id"]

        # 1️⃣ Move conversations
        supabase.table("conversations") \
            .update({"user_id": user_id}) \
            .eq("user_id", visitor_id) \
            .execute()

        # 2️⃣ Delete visitor record
        supabase.table("visitor_users") \
            .delete() \
            .eq("id", visitor_id) \
            .execute()

        logger.info(f"Merged visitor {visitor_id} → user {user_id}")

    except Exception as e:
        logger.error(f"Failed to merge visitor to user: {e}")

# Initialize scheduler
scheduler = AsyncIOScheduler()

async def check_available_models():
    print("Checking models...")

def process_tasks():
    tasks = supabase.table("background_tasks") \
        .select("*") \
        .eq("status", "queued") \
        .limit(5) \
        .execute()

    for task in tasks.data:
        # mark as processing
        supabase.table("background_tasks") \
            .update({"status": "processing"}) \
            .eq("id", task["id"]) \
            .execute()

        try:
            result = run_ai_task(task)

            supabase.table("background_tasks") \
                .update({
                    "status": "completed",
                    "result": json.dumps(result)
                }) \
                .eq("id", task["id"]) \
                .execute()

        except Exception as e:
            supabase.table("background_tasks") \
                .update({
                    "status": "failed",
                    "error": str(e)
                }) \
                .eq("id", task["id"]) \
                .execute()
            
# Fixed upload_image_to_supabase function to store in anonymous folder
def upload_image_to_supabase(image_bytes: bytes, filename: str, user_id: str):
    # Extract just the filename without the user_id prefix for storage
    storage_filename = filename.split("/")[-1] if "/" in filename else filename
    
    # Use the anonymous folder in the bucket
    storage_path = f"anonymous/{storage_filename}"
    
    upload = supabase.storage.from_("ai-images").upload(
        path=storage_path,
        file=image_bytes,
        file_options={"content-type": "image/png"}
    )

    # Fix: Check if upload has an error attribute
    if hasattr(upload, 'error') and upload.error:
        raise Exception(upload.error["message"])

    # Save image record with user ID
    try:
        supabase.table("images").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "image_path": storage_path,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save image record: {e}")

    return upload

def get_user_profile(user_id: str) -> dict:
    try:
        profile_resp = supabase.table("profiles").select("*").eq("id", user_id).limit(1).execute()
        if profile_resp.data:
            return profile_resp.data[0]
        
        # Create default profile if it doesn't exist
        default_profile = {
            "id": user_id,
            "nickname": f"User{user_id[:8]}",
            "personality": "friendly",
            "preferences": {}
        }
        
        supabase.table("profiles").insert(default_profile).execute()
        return default_profile
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        return {"nickname": "User", "personality": "friendly"}

def get_public_url(bucket: str, path: str) -> str:
    """Get public URL from Supabase Storage"""
    # Make sure to use the full URL including the protocol
    if not SUPABASE_URL.startswith(('http://', 'https://')):
        base_url = f"https://{SUPABASE_URL}"
    else:
        base_url = SUPABASE_URL
    
    return f"{base_url}/storage/v1/object/public/{bucket}/{path}"

@app.get("/test/image/{image_path:path}")
async def test_image_url(image_path: str):
    """Test endpoint to verify image URLs are accessible"""
    public_url = get_public_url("ai-images", f"anonymous/{image_path}")
    
    # Check if the image exists in Supabase
    try:
        response = requests.head(public_url)
        if response.status_code == 200:
            return {"url": public_url, "status": "accessible"}
        else:
            return {"url": public_url, "status": f"error: {response.status_code}"}
    except Exception as e:
        return {"url": public_url, "status": f"error: {str(e)}"}
        
async def generate_video_internal(prompt: str, samples: int = 1, user_id: str = None) -> dict:
    """
    Generate videos using Stable Video Diffusion (open-source alternative)
    """
    # Check for available video generation APIs
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
    
    # Try different services in order of preference
    if STABILITY_API_KEY:
        return await generate_video_stability(prompt, samples, user_id)
    elif HUGGINGFACE_API_KEY:
        return await generate_video_huggingface(prompt, samples, user_id)
    elif RUNWAYML_API_KEY:
        return await generate_video_runwayml(prompt, samples, user_id)
    else:
        # Fallback to a simple text-based video placeholder
        return await generate_placeholder_video(prompt, samples, user_id)

async def generate_video_stability(prompt: str, samples: int = 1, user_id: str = None) -> dict:
    """
    Generate videos using Stability AI's video generation API
    """
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    if not STABILITY_API_KEY:
        raise HTTPException(500, "STABILITY_API_KEY not configured")
    
    urls = []
    
    for i in range(samples):
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "width": 1024,
                "height": 576,  # 16:9 aspect ratio
                "samples": 1,
                "num_frames": 25,  # Number of frames in the video
                "seed": random.randint(0, 4294967295)
            }
            
            # Submit the request - Updated endpoint
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.stability.ai/v1/generation/stable-video-diffusion/text-to-video",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                # Get the video generation ID
                generation_id = result.get("id")
                if not generation_id:
                    raise HTTPException(500, "Failed to get video generation ID")
                
                # Poll for completion
                video_url = None
                max_attempts = 60  # Maximum polling attempts (5 minutes)
                
                for attempt in range(max_attempts):
                    # Check generation status
                    status_response = await client.get(
                        f"https://api.stability.ai/v1/generation/stable-video-diffusion/text-to-video/{generation_id}",
                        headers=headers
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                    status = status_data.get("status")
                    
                    if status == "completed":
                        video_url = status_data.get("video")
                        break
                    elif status == "failed":
                        error_message = status_data.get("error", "Unknown error")
                        raise HTTPException(500, f"Video generation failed: {error_message}")
                    
                    # Wait before polling again
                    await asyncio.sleep(5)
                
                if not video_url:
                    raise HTTPException(500, "Video generation timed out")
                
                # Download the video
                video_response = await client.get(video_url)
                video_response.raise_for_status()
                video_bytes = video_response.content
                
                # Upload to Supabase
                filename = f"{uuid.uuid4().hex[:8]}.mp4"
                storage_path = f"anonymous/{filename}"
                
                supabase.storage.from_("ai-videos").upload(
                    path=storage_path,
                    file=video_bytes,
                    file_options={"content-type": "video/mp4"}
                )
                
                # Save video record
                try:
                    supabase.table("videos").insert({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "video_path": storage_path,
                        "prompt": prompt,
                        "provider": "stability-ai",
                        "created_at": datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save video record: {e}")
                
                # Get public URL
                public_url = get_public_url("ai-videos", storage_path)
                urls.append(public_url)
                
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            # Continue with other samples if one fails
            continue
    
    if not urls:
        raise HTTPException(500, "No videos were generated successfully")
    
    return {
        "provider": "stability-ai",
        "videos": [{"url": url, "type": "video/mp4"} for url in urls]
    }

async def generate_video_huggingface(prompt: str, samples: int = 1, user_id: str = None) -> dict:
    """
    Generate videos using Hugging Face's video generation models
    """
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        raise HTTPException(500, "HF_API_KEY not configured")
    
    urls = []
    
    for i in range(samples):
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Using a text-to-video model from Hugging Face
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            }
            
            # Submit the request
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                # The response should contain the video data
                video_bytes = response.content
                
                # Upload to Supabase
                filename = f"{uuid.uuid4().hex[:8]}.mp4"
                storage_path = f"anonymous/{filename}"
                
                supabase.storage.from_("ai-videos").upload(
                    path=storage_path,
                    file=video_bytes,
                    file_options={"content-type": "video/mp4"}
                )
                
                # Save video record
                try:
                    supabase.table("videos").insert({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "video_path": storage_path,
                        "prompt": prompt,
                        "provider": "huggingface-damo",
                        "created_at": datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save video record: {e}")
                
                # Get public URL
                public_url = get_public_url("ai-videos", storage_path)
                urls.append(public_url)
                
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            # Continue with other samples if one fails
            continue
    
    if not urls:
        raise HTTPException(500, "No videos were generated successfully")
    
    return {
        "provider": "huggingface-damo",
        "videos": [{"url": url, "type": "video/mp4"} for url in urls]
    }

async def generate_video_runwayml(prompt: str, samples: int = 1, user_id: str = None) -> dict:
    """
    Generate videos using RunwayML Gen-2 API
    """
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
    if not RUNWAYML_API_KEY:
        raise HTTPException(500, "RUNWAYML_API_KEY not configured")
    
    urls = []
    
    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {RUNWAYML_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for i in range(samples):
        try:
            # Create a task for video generation
            task_payload = {
                "model": "gen-2",  # Using Gen-2 model
                "text_prompt": prompt,
                "watermark": False,
                "duration": 4,  # 4 seconds duration
                "ratio": "16:9",  # Widescreen format
                "upscale": True  # Higher quality output
            }
            
            # Submit the task
            async with httpx.AsyncClient(timeout=60.0) as client:
                task_response = await client.post(
                    "https://api.runwayml.com/v1/video_tasks",
                    headers=headers,
                    json=task_payload
                )
                task_response.raise_for_status()
                task_data = task_response.json()
                task_id = task_data.get("id")
                
                if not task_id:
                    raise HTTPException(500, "Failed to create video generation task")
                
                # Poll for task completion
                max_attempts = 60  # Maximum polling attempts (5 minutes)
                video_url = None
                
                for attempt in range(max_attempts):
                    # Check task status
                    status_response = await client.get(
                        f"https://api.runwayml.com/v1/video_tasks/{task_id}",
                        headers=headers
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                    status = status_data.get("status")
                    
                    if status == "SUCCEEDED":
                        video_url = status_data.get("output", {}).get("url")
                        break
                    elif status == "FAILED":
                        error_message = status_data.get("failure_reason", "Unknown error")
                        raise HTTPException(500, f"Video generation failed: {error_message}")
                    
                    # Wait before polling again
                    await asyncio.sleep(5)
                
                if not video_url:
                    raise HTTPException(500, "Video generation timed out")
                
                # Download the video
                video_response = await client.get(video_url)
                video_response.raise_for_status()
                video_bytes = video_response.content
                
                # Upload to Supabase
                filename = f"{uuid.uuid4().hex[:8]}.mp4"
                storage_path = f"anonymous/{filename}"
                
                supabase.storage.from_("ai-videos").upload(
                    path=storage_path,
                    file=video_bytes,
                    file_options={"content-type": "video/mp4"}
                )
                
                # Save video record
                try:
                    supabase.table("videos").insert({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "video_path": storage_path,
                        "prompt": prompt,
                        "provider": "runwayml-gen2",
                        "created_at": datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save video record: {e}")
                
                # Get public URL
                public_url = get_public_url("ai-videos", storage_path)
                urls.append(public_url)
                
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            # Continue with other samples if one fails
            continue
    
    if not urls:
        raise HTTPException(500, "No videos were generated successfully")
    
    return {
        "provider": "runwayml-gen2",
        "videos": [{"url": url, "type": "video/mp4"} for url in urls]
    }

async def generate_placeholder_video(prompt: str, samples: int = 1, user_id: str = None) -> dict:
    """
    Generate placeholder videos when no video generation API is available
    """
    urls = []
    
    for i in range(samples):
        # Create a simple text-based video placeholder
        try:
            # Create a simple text image as a placeholder
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # Create a black image
            width, height = 1024, 576  # 16:9 aspect ratio
            img = Image.new('RGB', (width, height), color='black')
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                # Try to use a larger font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            text = f"Video Generation Placeholder\n\nPrompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}\n\nNo video generation API configured"
            
            # Calculate text position
            text_width, text_height = draw.textsize(text, font=font)
            x = (width - text_width) / 2
            y = (height - text_height) / 2
            
            # Draw text
            draw.text((x, y), text, fill='white', font=font)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            # Upload to Supabase
            filename = f"{uuid.uuid4().hex[:8]}.mp4"
            storage_path = f"anonymous/{filename}"
            
            supabase.storage.from_("ai-videos").upload(
                path=storage_path,
                file=img_bytes,
                file_options={"content-type": "image/png"}  # Using PNG as placeholder
            )
            
            # Save video record
            try:
                supabase.table("videos").insert({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "video_path": storage_path,
                    "prompt": prompt,
                    "provider": "placeholder",
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save video record: {e}")
            
            # Get public URL
            public_url = get_public_url("ai-videos", storage_path)
            urls.append(public_url)
            
        except Exception as e:
            logger.error(f"Placeholder video generation failed: {e}")
            continue
    
    if not urls:
        raise HTTPException(500, "No videos were generated successfully")
    
    return {
        "provider": "placeholder",
        "videos": [{"url": url, "type": "image/png"} for url in urls],  # Using PNG as placeholder
        "message": "No video generation API configured. Please set STABILITY_API_KEY or HF_API_KEY environment variable to enable video generation."
    }

# Update the image generation handler
async def image_generation_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle image generation requests"""
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Generating image..."})
            try:
                # Extract any sample count from the prompt
                samples = 1
                sample_match = re.search(r'(\d+)\s+(image|images)', prompt.lower())
                if sample_match:
                    samples = min(int(sample_match.group(1)), 4)  # Cap at 4 images
                
                # Generate the image
                result = await _generate_image_core(prompt, samples, user_id, return_base64=False)
                
                yield sse({
                    "type": "images",
                    "provider": result["provider"],
                    "images": result["images"]  # Already in the correct format
                })
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
        samples = 1
        sample_match = re.search(r'(\d+)\s+(image|images)', prompt.lower())
        if sample_match:
            samples = min(int(sample_match.group(1)), 4)  # Cap at 4 images
        
        return await _generate_image_core(prompt, samples, user_id, return_base64=False)

# Update the video generation handler
async def video_generation_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle video generation requests with RunwayML Gen-2"""
    # Extract sample count from prompt
    samples = 1
    sample_match = re.search(r'(\d+)\s+(video|videos)', prompt.lower())
    if sample_match:
        samples = min(int(sample_match.group(1)), 2)  # Cap at 2 videos
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Generating video..."})
            try:
                result = await generate_video_internal(prompt, samples, user_id)
                yield sse({
                    "type": "videos",
                    "provider": result["provider"],
                    "videos": result["videos"]
                })
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
        return await generate_video_internal(prompt, samples, user_id)

# Add a function to check bucket visibility
def check_bucket_visibility():
    try:
        # Check images bucket
        images_url = f"{SUPABASE_URL}/storage/v1/bucket/ai-images"
        images_response = requests.get(images_url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        })
        
        # Check videos bucket
        videos_url = f"{SUPABASE_URL}/storage/v1/bucket/ai-videos"
        videos_response = requests.get(videos_url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        })
        
        logger.info(f"Images bucket public: {images_response.status_code == 200}")
        logger.info(f"Videos bucket public: {videos_response.status_code == 200}")
        
        return images_response.status_code == 200 and videos_response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking bucket visibility: {e}")
        return False
        
# Fixed cache_result function
def cache_result(prompt: str, provider: str, result: dict):
    # Store cache in Supabase
    try:
        supabase.table("cache").insert({
            "prompt": prompt,
            "provider": provider,
            "response": json.dumps(result),  # Use the correct column name
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to cache result: {e}")

# Fixed save_image_record function
def save_image_record(user_id, prompt, path, is_nsfw):
    try:
        supabase.table("images").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "prompt": prompt,
            "image_path": path,
            "is_nsfw": is_nsfw,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save image record: {e}")

# Fixed save_code_generation_record function
async def save_code_generation_record(user_id: str, language: str, prompt: str, code: str):
    try:
        supabase.table("code_generations").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "language": language,
            "prompt": prompt,
            "code": code,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save code generation record: {e}")

# Fixed get_or_create_user function
def init_supabase_tables():
    print("Supabase schema assumed present — skipping RPC table creation")

# Initialize on startup (safe no-op)
init_supabase_tables()

groq_client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
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
RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")

# Quick log so you can confirm key presence without printing the key itself
logger.info(f"GROQ key present: {bool(GROQ_API_KEY)}")
logger.info(f"RunwayML key present: {bool(RUNWAYML_API_KEY)}")

# -------------------
# Models
# -------------------
# Update this line in your code
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")  # Using a currently supported model
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions" # // Added missing URL

# TTS/STT are handled via ElevenLabs now
# ---------- Creator info ----------
CREATOR_INFO = {
    "name": "GoldYLocks",
    "age": 17,
    "country": "England",
    "projects": ["MZ", "LS", "SX", "CB"],
    "socials": { "discord":"@nexisphere123_89431", "twitter":"@NexiSphere"},
    "bio": "Created by GoldBoy (17, England). Projects: MZ, LS, SX, CB. Socials: Discord @nexisphere123_89431 Twitter @NexiSphere."
}
    # --- C / C++ ---
JUDGE0_LANGUAGES = {
    # --- C / C++ ---
    "c": 50,
    "c_clang": 49,
    "cpp": 54,
    "cpp_clang": 53,
    "cpp17": 54,
    "c++": 54,

    # --- Java ---
    "java": 62,

    # --- Python ---
    "python": 71,
    "python3": 71,
    "python2": 70,
    "py": 71,
    "py3": 71,

    #--- JavaScript / TypeScript ---
    "javascript": 63,
    "js": 63,
    "nodejs": 63,
    "node": 63,
    "typescript": 74,
    "ts": 74,

    #--- Go ---
    "go": 60,
    "golang": 60,

    #--- Rust ---
    "rust": 73,
    "rs": 73,

    #--- .NET ---
    "csharp": 51,
    "c#": 51,
    "fsharp": 87,
    "f#": 87,
    "dotnet": 51,
    "vb.net": 94,

    #--- PHP ---
    "php": 68,

    #--- Ruby ---
    "ruby": 72,
    "rb": 72,

    #--- Swift ---
    "swift": 83,

    #--- Kotlin ---
    "kotlin": 78,
    "kt": 78,

    #--- Scala ---
    "scala": 81,

    #--- Objective-C ---
    "objective_c": 52,
    "objc": 52,

    #--- Bash / Shell ---
    "bash": 46,
    "sh": 46,
    "shell": 46,
    "zsh": 46,

    #--- PowerShell ---
    "powershell": 88,
    "ps1": 88,

    #--- Perl ---
    "perl": 85,
    "pl": 85,

    #--- Lua ---
    "lua": 64,

    #--- R ---
    "r": 80,

    #--- Dart ---
    "dart": 75,

    #--- Julia ---
    "julia": 84,

    #--- Haskell ---
    "haskell": 61,
    "hs": 61,

    #--- Elixir ---
    "elixir": 57,
    "ex": 57,

    #--- Erlang ---
    "erlang": 58,
    "erl": 58,

    #--- OCaml ---
    "ocaml": 65,
    "ml": 65,

    #--- Crystal ---
    "crystal": 76,
    "cr": 76,

    #--- Nim ---
    "nim": 77,
    "nim": 77,

    #--- Zig ---
    "zig": 86,

    #--- Assembly ---
    "assembly": 45,
    "asm": 45,
    "nasm": 45,

    #--- COBOL ---
    "cobol": 55,
    "cbl": 55,

    #--- Fortran ---
    "fortran": 59,
    "f90": 59,

    #--- Prolog ---
    "prolog": 69,
    "pl": 69,

    #--- Scheme ---
    "scheme": 82,
    "scm": 82,

    #--- Common Lisp ---
    "lisp": 66,
    "cl": 66,

    #--- Brainf*ck ---
    "brainfuck": 47,
    "bf": 47,

    #--- V ---
    "vlang": 91,
    "v": 91,

    #--- Groovy ---
    "groovy": 56,

    #--- Hack ---
    "hack": 67,

    #--- Pascal ---
    "pascal": 67,

    #--- Scratch ---
    "scratch": 92,

    #--- Solidity ---
    "solidity": 94,
    "sol": 94,

    #--- SQL ---
    "sql": 82,

    #--- Text / Plain ---
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
    analysis_type: str = "summary" # // summary, entities, sentiment, keywords

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
    data: str  #// JSON or CSV format
    chart_type: str = "auto"  #// auto, bar, line, pie, scatter, heatmap
    options: Dict[str, Any] = {}

class VoiceCloningRequest(BaseModel):
    voice_sample: str # // Base64 encoded audio
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
    return {k: v for k, v in entities.items() if v}  #// Remove empty lists

def extract_keywords(text, num_keywords=10):
    """Extract keywords from text using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
      #  // Get top keywords
        top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

# Add these endpoints to help debug the frontend connection

@app.get("/debug/frontend-config")
async def debug_frontend_config():
    """Debug endpoint to check frontend configuration"""
    return {
        "backend_url": "http://0.0.0.0:8080",  # Or whatever your backend URL is
        "cors_origins": ["http://localhost:9898", "https://zynara.xyz", "https://www.zynara.xyz"],
        "supabase_url": SUPABASE_URL,
        "frontend_supabase_url": FRONTEND_SUPABASE_URL,
        "has_frontend_supabase": frontend_supabase is not None
    }

@app.get("/debug/health")
async def debug_health():
    """Enhanced health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "1.0.0",
        "services": {
            "supabase": "connected" if SUPABASE_URL else "not configured",
            "groq": "configured" if GROQ_API_KEY else "not configured",
            "openai": "configured" if OPENAI_API_KEY else "not configured",
            "stability": "configured" if STABILITY_API_KEY else "not configured",
            "runwayml": "configured" if RUNWAYML_API_KEY else "not configured"
        }
    }

@app.get("/debug/supabase")
async def debug_supabase():
    """Debug endpoint to test Supabase connection"""
    try:
        # Test a simple query
        response = supabase.table("users").select("count").execute()
        return {
            "status": "success",
            "message": "Supabase connection working",
            "data": response.data
        }
    except Exception as e:
        logger.error(f"Supabase debug error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/debug/auth")
async def debug_auth(request: Request, response: Response):
    """Debug endpoint to test authentication"""
    try:
        user = await get_or_create_user(request, response)
        return {
            "status": "success",
            "user_id": user.id,
            "anonymous": user.anonymous,
            "session_token": user.session_token
        }
    except Exception as e:
        logger.error(f"Auth debug error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/test")
async def test_page():
    """Simple test page to verify frontend connection"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Test</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .status { 
                padding: 15px; 
                margin: 15px 0; 
                border-radius: 5px; 
                font-weight: bold;
            }
            .success { 
                background-color: #d4edda; 
                color: #155724; 
                border: 1px solid #c3e6cb;
            }
            .error { 
                background-color: #f8d7da; 
                color: #721c24; 
                border: 1px solid #f5c6cb;
            }
            .info { 
                background-color: #d1ecf1; 
                color: #0c5460; 
                border: 1px solid #bee5eb;
            }
            pre { 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto; 
                border: 1px solid #e9ecef;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                font-size: 14px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
            }
            .test-section {
                margin: 20px 0;
                padding: 15px;
                border: 1px solid #dee2e6;
                border-radius: 5px;
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 API Connection Test</h1>
            <div id="status" class="status info">
                <span id="status-text">Testing connection...</span>
                <span id="loading" class="loading" style="display: none;"></span>
            </div>
            
            <div class="test-section">
                <h2>📊 Response:</h2>
                <pre id="response">Loading...</pre>
            </div>
            
            <div class="test-section">
                <h2>🧪 Test API Calls</h2>
                <button onclick="testHealth()">Test Health Endpoint</button>
                <button onclick="testUniversal()">Test Universal Endpoint</button>
                <button onclick="testSupabase()">Test Supabase Connection</button>
                <button onclick="testAuth()">Test Authentication</button>
                <button onclick="testFrontendConfig()">Test Frontend Config</button>
            </div>
            
            <div class="test-section">
                <h2>🔍 Debug Information</h2>
                <button onclick="showDebugInfo()">Show Debug Info</button>
                <pre id="debug-info" style="display: none;"></pre>
            </div>
        </div>
        
        <script>
            let currentRequest = null;
            
            function setLoading(isLoading, message = 'Loading...') {
                const statusEl = document.getElementById('status');
                const statusText = document.getElementById('status-text');
                const loadingEl = document.getElementById('loading');
                
                statusText.textContent = message;
                if (isLoading) {
                    statusEl.className = 'status info';
                    loadingEl.style.display = 'inline-block';
                } else {
                    loadingEl.style.display = 'none';
                }
            }
            
            function setStatus(message, type = 'info') {
                const statusEl = document.getElementById('status');
                const statusText = document.getElementById('status-text');
                
                statusText.textContent = message;
                statusEl.className = `status ${type}`;
            }
            
            function setResponse(data) {
                document.getElementById('response').textContent = JSON.stringify(data, null, 2);
            }
            
            async function makeRequest(url, options = {}) {
                if (currentRequest) {
                    currentRequest.abort();
                }
                
                const controller = new AbortController();
                currentRequest = controller;
                
                try {
                    const response = await fetch(url, {
                        ...options,
                        signal: controller.signal
                    });
                    
                    currentRequest = null;
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(`HTTP ${response.status}: ${errorData.detail || response.statusText}`);
                    }
                    
                    return await response.json();
                } catch (error) {
                    currentRequest = null;
                    if (error.name === 'AbortError') {
                        throw new Error('Request aborted');
                    }
                    throw error;
                }
            }
            
            async function testHealth() {
                setLoading(true, 'Testing health endpoint...');
                try {
                    const data = await makeRequest('/debug/health');
                    setStatus('Health check successful!', 'success');
                    setResponse(data);
                } catch (error) {
                    setStatus('Error: ' + error.message, 'error');
                    setResponse({ error: error.message, stack: error.stack });
                }
            }
            
            async function testUniversal() {
                setLoading(true, 'Testing universal endpoint...');
                try {
                    const data = await makeRequest('/ask/universal', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            prompt: 'Hello, this is a test message'
                        })
                    });
                    setStatus('Universal endpoint test successful!', 'success');
                    setResponse(data);
                } catch (error) {
                    setStatus('Error: ' + error.message, 'error');
                    setResponse({ error: error.message, stack: error.stack });
                }
            }
            
            async function testSupabase() {
                setLoading(true, 'Testing Supabase connection...');
                try {
                    const data = await makeRequest('/debug/supabase');
                    if (data.status === 'success') {
                        setStatus('Supabase connection successful!', 'success');
                    } else {
                        setStatus('Supabase connection failed', 'error');
                    }
                    setResponse(data);
                } catch (error) {
                    setStatus('Error: ' + error.message, 'error');
                    setResponse({ error: error.message, stack: error.stack });
                }
            }
            
            async function testAuth() {
                setLoading(true, 'Testing authentication...');
                try {
                    const data = await makeRequest('/debug/auth');
                    if (data.status === 'success') {
                        setStatus('Authentication successful!', 'success');
                    } else {
                        setStatus('Authentication failed', 'error');
                    }
                    setResponse(data);
                } catch (error) {
                    setStatus('Error: ' + error.message, 'error');
                    setResponse({ error: error.message, stack: error.stack });
                }
            }
            
            async function testFrontendConfig() {
                setLoading(true, 'Testing frontend configuration...');
                try {
                    const data = await makeRequest('/debug/frontend-config');
                    setStatus('Frontend config retrieved!', 'success');
                    setResponse(data);
                } catch (error) {
                    setStatus('Error: ' + error.message, 'error');
                    setResponse({ error: error.message, stack: error.stack });
                }
            }
            
            function showDebugInfo() {
                const debugInfo = {
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    cookies: document.cookie,
                    localStorage: Object.keys(localStorage),
                    sessionStorage: Object.keys(sessionStorage)
                };
                
                document.getElementById('debug-info').style.display = 'block';
                document.getElementById('debug-info').textContent = JSON.stringify(debugInfo, null, 2);
            }
            
            // Test health endpoint on page load
            window.onload = () => {
                testHealth();
            };
            
            // Handle page visibility change
            document.addEventListener('visibilitychange', () => {
                if (!document.hidden && currentRequest) {
                    currentRequest.abort();
                    currentRequest = null;
                    setLoading(false, 'Request cancelled due to page visibility change');
                }
            });
        </script>
    </body>
    </html>
    """)

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
    
def create_knowledge_graph(entities, relationship_type="related"):
    """Create a simple knowledge graph from entities"""
    G = nx.Graph()
    
   # // Add nodes
    for entity in entities:
        G.add_node(entity)
    
   # // Add edges (simple example - in a real implementation, you'd use NLP to find relationships)
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
           # // Simple similarity based on string overlap
            similarity = len(set(entity1.lower().split()) & set(entity2.lower().split()))
            if similarity > 0:
                G.add_edge(entity1, entity2, weight=similarity, type=relationship_type)
    
    return G

def visualize_graph(G):
    """Create a visualization of the knowledge graph"""
    pos = nx.spring_layout(G)
    
    #// Create a plotly figure
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
    
   # // Security checks
    if "security" in focus_areas:
     #   // Check for common security issues
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
    
  #  // Performance checks
    if "performance" in focus_areas:
        if language == "python":
            if "for i in range(len(" in code:
                results["performance"].append("Consider using enumerate() instead of range(len())")
            if code.count("for ") > 5:
                results["performance"].append("Multiple nested loops detected - consider optimizing")
        elif language == "javascript":
            if "for (var i = 0; i <" in code:
                results["performance"].append("Consider using forEach() or map() instead of for loops")
    
 #   // Style checks
    if "style" in focus_areas:
        if language == "python":
            if not re.search(r'^\s*def \w+\([^)]*\):\s*"""', code, re.MULTILINE):
                results["style"].append("Missing docstrings for functions")
            if code.count("    ") > 0 and code.count("\t") > 0:
                results["style"].append("Mixed tabs and spaces detected")
        elif language == "javascript":
            if "var " in code:
                results["style"].append("Consider using let or const instead of var")
    
  #  // Calculate overall score
    total_issues = sum(len(issues) for issues in results.values() if isinstance(issues, list))
    results["overall_score"] = max(0, 100 - (total_issues * 10))
    
    return results

async def get_or_create_conversation(user_id: str):
    result = await asyncio.to_thread(
        lambda: supabase
        .table("conversations")
        .select("id")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(1)
        .execute()
    )

    if result.data:
        return result.data[0]["id"]

    new_id = str(uuid.uuid4())

    await asyncio.to_thread(
        lambda: supabase
        .table("conversations")
        .insert({
            "id": new_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        .execute()
    )

    return new_id
    
def run_check():
    return {"status": "ok", "message": "System check passed"}

def create_chart(data, chart_type, options):
    """Create a chart from data"""
    try:
       # // Parse data
        if data.strip().startswith('{'):
       #     // JSON data
            df = pd.read_json(data)
        else:
       #     // CSV data
            df = pd.read_csv(StringIO(data))
        
       # // Create chart based on type
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
          #  // Default to bar chart
            fig = px.bar(df, **options)
        
       # // Convert to HTML
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return f"<p>Error creating chart: {str(e)}</p>"

#// ---------- Helper Functions ----------
# Fix: Add get_judge0_client function
def get_judge0_client():
    """Get Judge0 client for code execution"""
    if not JUDGE0_KEY:
        logger.warning("Judge0 client not available - missing API key")
        return None
    
    try:
        from judge0 import Judge0Client
        client = Judge0Client(
            base_url=JUDGE0_URL,
            api_key=JUDGE0_KEY
        )
        return client
    except ImportError:
        logger.error("Judge0 library not installed - code execution disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Judge0 client: {e}")
        return None

async def run_code_judge0(code: str, language_id: int, stdin: str = ""):
    """Execute code using Judge0 API with proper error handling"""
    client = get_judge0_client()
    
    if not client:
        return {
            "error": "Judge0 client not initialized. Check JUDGE0_KEY.",
            "status": "error"
        }
    
    try:
        # Submit the code for execution
        submission = client.submissions.create(
            source_code=code,
            language_id=language_id,
            stdin=stdin,
            wait=True,
            base64_encoded=False
        )
        
        # Get the submission token
        token = submission.get("token")
        
        if not token:
            return {
                "error": "Failed to get submission token from Judge0",
                "status": "error"
            }
        
        # Get the submission result
        result = client.submissions.get(token)
        
        # Format the result for our application
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "compile_output": result.get("compile_output", ""),
            "status": {
                "id": result.get("status", {}).get("id"),
                "description": result.get("status", {}).get("description")
            },
            "time": result.get("time"),
            "memory": result.get("memory"),
            "exit_code": result.get("exit_code")
        }
    except Exception as e:
        logger.error(f"Judge0 execution failed: {str(e)}")
        return {
            "error": f"Judge0 execution failed: {str(e)}",
            "status": "error"
        }

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
    tool_calls = []

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
                    delta = chunk["choices"][0]["delta"]

                  #  // -------------------------
                 #   // TOOL CALLS
                #    // -------------------------
                    if "tool_calls" in delta:
                    #    // Collect tool calls
                        for tool_call in delta.get("tool_calls", []):
                            if "id" in tool_call:
                                tool_calls.append({
                                    "id": tool_call["id"],
                                    "type": tool_call.get("type", "function"),
                                    "function": tool_call.get("function", {})
                                })
                            elif "function" in tool_call:
                           #     // Update the last tool call with function details
                                if tool_calls:
                                    last_call = tool_calls[-1]
                                    if "name" in tool_call["function"]:
                                        last_call["function"]["name"] = tool_call["function"]["name"]
                                    if "arguments" in tool_call["function"]:
                                        last_call["function"]["arguments"] = tool_call["function"]["arguments"]
                        continue

                  #  // -------------------------
                   # // NORMAL TEXT STREAMING
               #     // -------------------------
                    content = delta.get("content")
                    if content:
                       # // 🚫 Prevent tool leakage
                        if "<function=" in content:
                            pass
                        else:
                            assistant_reply += content
                            yield sse({
                                "type": "token",
                                "text": content
                            })
                except Exception as e:
                    logger.error(f"Error processing stream chunk: {e}")
                    continue

 #   // -------------------------
   # // EXECUTE TOOL CALLS
  #  // -------------------------
    if tool_calls:
        yield sse({"type": "tool_calls_start", "count": len(tool_calls)})
        
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])

            if name == "web_search":
                result = await duckduckgo_search(args["query"])
            elif name == "run_code":
                result = await run_code_safely(args["task"])
            else:
                result = {"error": f"Unknown tool: {name}"}

            yield sse({
                "type": "tool_result",
                "tool": name,
                "result": result
            })

         #   // Add tool result to messages for context
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(result)
            })

      #  // Continue conversation with tool results
        yield sse({"type": "continuing_with_tools"})
        
       # // Get final response with tool results
        final_payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": 1500,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                GROQ_URL,
                headers=get_groq_headers(),
                json=final_payload,
            ) as response:

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line.replace("data:", "", 1).strip()

                    if not data or data == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content")
                        if content:
                            assistant_reply += content
                            yield sse({
                                "type": "token",
                                "text": content
                            })
                    except Exception as e:
                        logger.error(f"Error processing final stream chunk: {e}")
                        continue

#    // Save the complete response
    if assistant_reply.strip():
        await persist_reply(user_id, conversation_id, assistant_reply)

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

# Fix: Complete the run_code_safely function
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
    
    return {
        "code": code,
        "execution": execution
    }
    
async def duckduckgo_search(q: str):
    """
    Use DuckDuckGo Instant Answer API (no API key required).
    Returns a simple structured result with abstract, answer and a list of related topics.
    """
    # Truncate the query if it's too long (URLs have length limits)
    max_query_length = 500  # Safe limit for URL parameters
    if len(q) > max_query_length:
        q = q[:max_query_length-3] + "..."  # Truncate and add indicator
    
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
    
    try:
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
    except httpx.HTTPStatusError as e:
        logger.error(f"DuckDuckGo returned HTTP error: {e.response.status_code}")
        raise HTTPException(502, "duckduckgo_error")
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        raise HTTPException(500, "search_failed")
        

def get_cached_result(prompt: str, provider: str) -> Optional[dict]:
    try:
        response = supabase.table("cache").select("response").eq("prompt", prompt).eq("provider", provider).order("created_at", desc=True).limit(1).execute()
        if response.data:
            return json.loads(response.data[0]["response"])
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
    """Build system prompt with user memory and conversation history"""
    try:
        # Get user's recent conversations
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        
        if not conv_response.data:
            # New user, no history
            return f"You are a helpful AI assistant. User message: {message}"
        
        conversation_id = conv_response.data[0]["id"]
        
        # Get recent messages from this conversation
        msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at", asc=True).limit(10).execute()
        
        # Build context from messages
        context = "Recent conversation:\n"
        for msg in msg_response.data:
            context += f"{msg['role']}: {msg['content']}\n"
        
        # Get user preferences if they exist
        profile_response = supabase.table("profiles").select("preferences").eq("id", user_id).execute()
        if profile_response.data:
            preferences = profile_response.data[0].get("preferences", {})
            if preferences:
                context += f"\nUser preferences: {json.dumps(preferences)}\n"
        
        return f"""You are a helpful AI assistant with memory of this user.

{context}

Current message: {message}"""
    except Exception as e:
        logger.error(f"Failed to build contextual prompt: {e}")
        return f"You are a helpful AI assistant. User message: {message}"

def persist_message(user_id: str, conversation_id: str, role: str, content: str):
    """Store message in database"""
    try:
        supabase.table("messages").insert({
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
      #  // Update conversation timestamp
        supabase.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
    except Exception as e:
        logger.error(f"Failed to persist message: {e}")

def get_or_create_conversation(user_id: str) -> str:
    """Get existing conversation or create new one"""
    try:
        # Try to get most recent conversation
        response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]["id"]
        
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        supabase.table("conversations").insert({
            "id": conversation_id,
            "user_id": user_id,
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).execute()
        
        return conversation_id
    except Exception as e:
        logger.error(f"Failed to get or create conversation: {e}")
        return str(uuid.uuid4())  # Fallback

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

def route_query(user_id: str, query: str):
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

@app.post("/user/logout")
async def logout_user(request: Request, response: Response):
    # Clear the cookie
    response.delete_cookie(
        key="session_token",
        path="/",
        domain=None,
        secure=False,
        httponly=True,
        samesite="lax"
    )
    
    return {"status": "success", "message": "Logged out successfully"}

async def nsfw_check(prompt: str) -> bool:
    if not OPENAI_API_KEY: 
        return False
    
    try:
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
            
            # Log the full moderation result for debugging
            logger.info(f"Moderation result: {result}")
            
            # Check for any flagged category
            return result["flagged"]
    except Exception as e:
        logger.error(f"NSFW check failed: {e}")
        return False
    
    return False
    
async def _generate_image_core(
    prompt: str,
    samples: int,
    user_id: str,
    return_base64: bool = False
):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    # Sanitize prompt to avoid content policy violations
    prompt = sanitize_prompt(prompt)
    
    # Content moderation check
    is_flagged = await nsfw_check(prompt)
    if is_flagged:
        raise HTTPException(
            status_code=400, 
            detail="Image generation prompt violates content policy."
        )

    provider_used = "openai"
    urls = []

    # Clean and validate the prompt
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(400, "Empty prompt provided")
    
    # Limit prompt length to avoid API issues
    if len(clean_prompt) > 1000:
        clean_prompt = clean_prompt[:1000] + "..."
        logger.warning(f"Prompt truncated to 1000 characters")

    payload = {
        "model": "dall-e-3",
        "prompt": clean_prompt,
        "n": 1,  # DALL·E 3 supports only 1 image
        "size": "1024x1024",
        "response_format": "b64_json",
        "quality": "standard"  # Add quality parameter
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "content-type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/images/generations",
                json=payload,
                headers=headers
            )
            
            # Log the response for debugging
            if r.status_code != 200:
                logger.error(f"OpenAI API error: {r.status_code} - {r.text}")
                try:
                    error_data = r.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                except:
                    error_msg = r.text
                raise HTTPException(400, f"Image generation failed: {error_msg}")
            
            result = r.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from OpenAI: {e.response.status_code} - {e.response.text}")
        raise HTTPException(500, "Image generation provider error")
    except Exception as e:
        logger.exception("OpenAI image API call failed")
        raise HTTPException(500, "Image generation provider error")

    if not result or not result.get("data"):
        logger.error("OpenAI returned empty image response: %s", result)
        raise HTTPException(500, "Image generation failed")

    for img in result["data"]:
        try:
            b64 = img.get("b64_json")
            if not b64:
                continue

            image_bytes = base64.b64decode(b64)
            filename = f"{uuid.uuid4().hex}.png"
            storage_path = f"anonymous/{filename}"

            # Upload to Supabase
            upload = supabase.storage.from_("ai-images").upload(
                path=storage_path,
                file=image_bytes,
                file_options={"content-type": "image/png"}
            )

            # Save image record with user ID
            try:
                supabase.table("images").insert({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "image_path": storage_path,
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save image record: {e}")

            # Get public URL instead of signed URL
            public_url = get_public_url("ai-images", storage_path)
            urls.append(public_url)

        except Exception:
            logger.exception("Failed processing or uploading image")
            continue

    if not urls:
        raise HTTPException(500, "No images generated")

    cache_result(prompt, provider_used, {"images": urls})

    return {
        "provider": provider_used,
        "images": [{"url": url, "type": "image/png"} for url in urls],  # Updated format
        "user_id": user_id
    }

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
        conv_response = (
            supabase.table("conversations")
            .select("id")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )

        if conv_response.data:
            conversation_id = conv_response.data[0]["id"]

            msg_response = (
                supabase.table("messages")
                .select("role, content")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=False)  # oldest → newest
                .limit(limit)
                .execute()
            )

            rows = msg_response.data or []
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    except Exception as e:
        logger.error(f"Failed to load history: {e}")

    return []


def load_memory(conversation_id: str, limit: int = 20):
    try:
        # Fix: Changed order("created_at", desc=False) to order("created_at")
        response = (
            supabase.table("messages")
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .limit(limit)
            .execute()
        )

        rows = response.data or []
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
                    continue

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
                    continue

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
  #  // Simple auth function that extracts user_id from cookie
    user = await get_or_create_user(request, Response())
    if not user:
        raise HTTPException(401, "Not authenticated")
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

#// Tracks currently active SSE/streaming tasks per user
active_streams: Dict[str, asyncio.Task] = {}

#// ---------- Utility Functions ----------
def generate_random_nickname():
    adjectives = ["Happy", "Brave", "Clever", "Friendly", "Gentle", "Kind", "Lucky", "Proud", "Smart", "Wise"]
    nouns = ["Bear", "Eagle", "Fox", "Lion", "Tiger", "Wolf", "Dolphin", "Hawk", "Owl"]
    return f"{random.choice(adjectives)}{random.choice(nouns)}"

#// ---------- Advanced Feature Implementations ----------
async def document_analysis(prompt: str, user_id: str, stream: bool = False):
    """Analyze documents for key information"""
  #  // Extract text from prompt
    text_match = re.search(r'document[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "No document text found in prompt")
    
    text = text_match.group(1).strip()
    
  #  // Determine analysis type
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
            
       #     // Extract entities
            if analysis_type in ["summary", "entities"]:
                yield sse({"type": "progress", "message": "Extracting entities..."})
                entities = extract_entities(text)
                yield sse({"type": "entities", "data": entities})
            
       #     // Extract keywords
            if analysis_type in ["summary", "keywords"]:
                yield sse({"type": "progress", "message": "Extracting keywords..."})
                keywords = extract_keywords(text)
                yield sse({"type": "keywords", "data": keywords})
            
      #      // Generate summary
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
            
     #       // Analyze sentiment
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
    #    // Non-streaming version
        result = {
            "entities": extract_entities(text),
            "keywords": extract_keywords(text),
        }
        
   #     // Generate summary
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
        
    #    // Analyze sentiment
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
 #   // Extract text and languages from prompt
    text_match = re.search(r'translate[:\s]+(.*?)(?:\s+to\s+|\s+in\s+)(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "Could not extract text and target language from prompt")
    
    text = text_match.group(1).strip()
    target_lang = text_match.group(2).strip()
    
#    // Map common language names to language codes
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
            
        #    // Use Groq for translation
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
   #     // Non-streaming version
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
#    // Extract text from prompt
    text_match = re.search(r'sentiment[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
 #       // If no explicit text, use the whole prompt
        text = prompt
    else:
        text = text_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Analyzing sentiment..."})
            
       #     // Use Groq for sentiment analysis
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
      #  // Non-streaming version
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
 #   // Extract entities from prompt
    entities_match = re.search(r'entities[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not entities_match:
        raise HTTPException(400, "Could not extract entities from prompt")
    
    entities_text = entities_match.group(1).strip()
    entities = [e.strip() for e in entities_text.split(',')]
    
   # // Extract relationship type
    relationship_type = "related"
    rel_match = re.search(r'relationship[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if rel_match:
        relationship_type = rel_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Creating knowledge graph..."})
            
          #  // Create knowledge graph
            yield sse({"type": "progress", "message": "Building graph structure..."})
            G = create_knowledge_graph(entities, relationship_type)
            
          #  // Generate visualization
            yield sse({"type": "progress", "message": "Generating visualization..."})
            graph_html = visualize_graph(G)
            
          #  // Save to Supabase
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
       # // Non-streaming version
        G = create_knowledge_graph(entities, relationship_type)
        graph_html = visualize_graph(G)
        
     #   // Save to Supabase
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
 #   // Extract training data from prompt
    data_match = re.search(r'data[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not data_match:
        raise HTTPException(400, "Could not extract training data from prompt")
    
    training_data = data_match.group(1).strip()
    
  #  // Extract model type
    model_type = "classification"
    type_match = re.search(r'type[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if type_match:
        model_type = type_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Preparing model training..."})
            
          #  // Create a model training job
            yield sse({"type": "progress", "message": "Creating training job..."})
            job_id = str(uuid.uuid4())
            
         #   // Save job to Supabase
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
    #    // Non-streaming version
        job_id = str(uuid.uuid4())
        
      #  // Save job to Supabase
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
 #   // Extract code from prompt
    code_match = re.search(r'code[:\s]+```(.*?)```', prompt, re.DOTALL | re.IGNORECASE)
    if not code_match:
        raise HTTPException(400, "Could not extract code from prompt")
    
    code = code_match.group(1).strip()
    
  #  // Extract language
    language = "python"
    lang_match = re.search(r'language[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if lang_match:
        language = lang_match.group(1).strip()
    
  #  // Extract focus areas
    focus_areas = ["security", "performance", "style"]
    focus_match = re.search(r'focus[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if focus_match:
        focus_text = focus_match.group(1).strip()
        focus_areas = [a.strip() for a in focus_text.split(',')]
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Analyzing code..."})
            
          #  // Analyze code
            yield sse({"type": "progress", "message": "Checking code quality..."})
            results = analyze_code_quality(code, language, focus_areas)
            
           # // Generate suggestions
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
            
           # // Save review to Supabase
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
       # // Non-streaming version
        results = analyze_code_quality(code, language, focus_areas)
        
       # // Generate suggestions
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
        
       # // Save review to Supabase
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
  #  // Extract query from prompt
    query_match = re.search(r'query[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not query_match:
       # // If no explicit query, use the whole prompt
        query = prompt
    else:
        query = query_match.group(1).strip()
    
  #  // Extract search types
    search_types = ["text", "image", "video"]
    types_match = re.search(r'types[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if types_match:
        types_text = types_match.group(1).strip()
        search_types = [t.strip() for t in types_text.split(',')]
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Starting multimodal search..."})
            
          #  // Text search
            if "text" in search_types:
                yield sse({"type": "progress", "message": "Searching text..."})
                text_results = await duckduckgo_search(query)
                yield sse({"type": "text_results", "data": text_results})
            
            #// Image search
            if "image" in search_types:
                yield sse({"type": "progress", "message": "Searching images..."})
               # // This is a placeholder - in a real implementation, you'd use an image search API
                image_results = {
                    "query": query,
                    "results": [
                        {"url": f"https://example.com/image1.jpg?query={query}", "title": f"Image 1 for {query}"},
                        {"url": f"https://example.com/image2.jpg?query={query}", "title": f"Image 2 for {query}"}
                    ]
                }
                yield sse({"type": "image_results", "data": image_results})
            
            #// Video search
            if "video" in search_types:
                yield sse({"type": "progress", "message": "Searching videos..."})
               # // This is a placeholder - in a real implementation, you'd use a video search API
                video_results = {
                    "query": query,
                    "results": [
                        {"url": f"https://example.com/video1.mp4?query={query}", "title": f"Video 1 for {query}"},
                        {"url": f"https://example.com/video2.mp4?query={query}", "title": f"Video 2 for {query}"}
                    ]
                }
                yield sse({"type": "video_results", "data": video_results})
            
           # // Save search to Supabase
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
        #// Non-streaming version
        results = {}
        
       # // Text search
        if "text" in search_types:
            results["text"] = await duckduckgo_search(query)
        
      #  // Image search
        if "image" in search_types:
           # // This is a placeholder - in a real implementation, you'd use an image search API
            results["image"] = {
                "query": query,
                "results": [
                    {"url": f"https://example.com/image1.jpg?query={query}", "title": f"Image 1 for {query}"},
                    {"url": f"https://example.com/image2.jpg?query={query}", "title": f"Image 2 for {query}"}
                ]
            }
        
      #  // Video search
        if "video" in search_types:
           # // This is a placeholder - in a real implementation, you'd use a video search API
            results["video"] = {
                "query": query,
                "results": [
                    {"url": f"https://example.com/video1.mp4?query={query}", "title": f"Video 1 for {query}"},
                    {"url": f"https://example.com/video2.mp4?query={query}", "title": f"Video 2 for {query}"}
                ]
            }
        
       # // Save search to Supabase
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
    #// Extract preferences from prompt
    prefs_match = re.search(r'preferences[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not prefs_match:
        raise HTTPException(400, "Could not extract preferences from prompt")
    
    preferences_text = prefs_match.group(1).strip()
    
   # // Parse preferences
    preferences = {}
    for line in preferences_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            preferences[key.strip()] = value.strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Updating AI preferences..."})
            
            #// Save preferences to Supabase
            yield sse({"type": "progress", "message": "Saving preferences..."})
            try:
               # // Check if user profile exists
                profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
                
                if profile_response.data:
                    #// Update existing profile
                    supabase.table("profiles").update({
                        "preferences": preferences,
                        "updated_at": datetime.now().isoformat()
                    }).eq("id", user_id).execute()
                else:
                   # // Create new profile
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
       # // Non-streaming version
        try:
            #// Check if user profile exists
            profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
            
            if profile_response.data:
              #  // Update existing profile
                supabase.table("profiles").update({
                    "preferences": preferences,
                    "updated_at": datetime.now().isoformat()
                }).eq("id", user_id).execute()
            else:
               # // Create new profile
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
    #// Extract data from prompt
    data_match = re.search(r'data[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not data_match:
        raise HTTPException(400, "Could not extract data from prompt")
    
    data = data_match.group(1).strip()
    
    #// Extract chart type
    chart_type = "auto"
    type_match = re.search(r'chart[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if type_match:
        chart_type = type_match.group(1).strip()
    
    #// Extract options
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
            
            #// Create chart
            yield sse({"type": "progress", "message": "Generating chart..."})
            chart_html = create_chart(data, chart_type, options)
            
            #// Save visualization to Supabase
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
       # // Non-streaming version
        chart_html = create_chart(data, chart_type, options)
        
        #// Save visualization to Supabase
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
    #// Extract voice sample from prompt
    sample_match = re.search(r'sample[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not sample_match:
        raise HTTPException(400, "Could not extract voice sample from prompt")
    
    voice_sample = sample_match.group(1).strip()
    
    #// Extract text to synthesize
    text_match = re.search(r'text[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if not text_match:
        raise HTTPException(400, "Could not extract text to synthesize from prompt")
    
    text = text_match.group(1).strip()
    
   # // Extract voice name
    voice_name = "custom_voice"
    name_match = re.search(r'name[:\s]+(.*?)(?:\n\n|\n$|$)', prompt, re.DOTALL | re.IGNORECASE)
    if name_match:
        voice_name = name_match.group(1).strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Creating voice profile..."})
            
           # // Create voice profile
            yield sse({"type": "progress", "message": "Analyzing voice sample..."})
            voice_id = str(uuid.uuid4())
            
           # // Save voice profile to Supabase
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
            
          #  // Synthesize speech
            yield sse({"type": "progress", "message": "Synthesizing speech..."})
            
          #  // Use OpenAI TTS with a standard voice (voice cloning would require a specialized API)
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "tts-1",
                "voice": "alloy", #  // Default voice - in a real implementation, you'd use the cloned voice
                "input": text
            }
            
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers=headers,
                    json=payload
                )
                r.raise_for_status()
            
          #  // Convert to base64
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
        #// Non-streaming version
        voice_id = str(uuid.uuid4())
        
        #// Save voice profile to Supabase
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
        
       # // Synthesize speech
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "voice": "alloy",  #// Default voice - in a real implementation, you'd use the cloned voice
            "input": text
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload
            )
            r.raise_for_status()
        
        #// Convert to base64
        audio_b64 = base64.b64encode(r.content).decode()
        
        return {
            "voice_name": voice_name,
            "text": text,
            "audio": audio_b64,
            "voice_id": voice_id
        }

#// Helper functions for all the missing intents that work with streaming
async def img2img_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle image-to-image editing requests"""
   # // For now, we'll just return an error message since we need an image file
    if stream:
        async def event_generator():
            yield sse({"type": "error", "message": "Image editing requires uploading an image file. Please use the /img2img endpoint directly."})
        
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
        return {"error": "Image editing requires uploading an image file. Please use the /img2img endpoint directly."}

async def vision_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle vision/analysis requests"""
  #  // For now, we'll just return an error message since we need an image file
    if stream:
        async def event_generator():
            yield sse({"type": "error", "message": "Image analysis requires uploading an image file. Please use the /vision/analyze endpoint directly."})
        
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
        return {"error": "Image analysis requires uploading an image file. Please use the /vision/analyze endpoint directly."}

async def stt_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle speech-to-text requests"""
  #  // For now, we'll just return an error message since we need an audio file
    if stream:
        async def event_generator():
            yield sse({"type": "error", "message": "Speech transcription requires uploading an audio file. Please use the /stt endpoint directly."})
        
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
        return {"error": "Speech transcription requires uploading an audio file. Please use the /stt endpoint directly."}

async def tts_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle text-to-speech requests"""
   # // Extract text to speak
    text = prompt
    if "say" in prompt.lower():
        text = prompt.lower().split("say", 1)[1].strip()
    elif "speak" in prompt.lower():
        text = prompt.lower().split("speak", 1)[1].strip()
    elif "read" in prompt.lower():
        text = prompt.lower().split("read", 1)[1].strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Generating speech..."})
            try:
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
                
               # // Convert to base64
                audio_b64 = base64.b64encode(r.content).decode()
                
                yield sse({
                    "type": "audio",
                    "text": text,
                    "audio": audio_b64
                })
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
       # // Non-streaming version
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
        
        #// Convert to base64
        audio_b64 = base64.b64encode(r.content).decode()
        
        return {
            "text": text,
            "audio": audio_b64
        }

async def code_generation_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle code generation requests"""
   # // Extract language from prompt
    language = "python"
    lang_match = re.search(r'(python|javascript|java|c\+\+|c#|php|ruby|go|rust|swift|kotlin)\s+code', prompt.lower())
    if lang_match:
        language = lang_match.group(1)
    
  #  // Extract run flag
    run_flag = "run" in prompt.lower() or "execute" in prompt.lower()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": f"Generating {language} code..."})
            try:
               # // Generate code
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
                
                yield sse({
                    "type": "code",
                    "language": language,
                    "code": code
                })
                
                #// Run code if requested
                if run_flag:
                    yield sse({"type": "progress", "message": "Running code..."})
                    lang_id = JUDGE0_LANGUAGES.get(language, 71)
                    execution = await run_code_judge0(code, lang_id)
                    yield sse({
                        "type": "execution",
                        "result": execution
                    })
                
                #// Save code generation record
                try:
                    supabase.table("code_generations").insert({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "language": language,
                        "prompt": prompt,
                        "code": code,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    logger.error(f"Failed to save code generation record: {e}")
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
     #   // Non-streaming version
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
        
        result = {
            "language": language,
            "generated_code": code,
            "user_id": user_id
        }
        
      #  // Run code if requested
        if run_flag:
            lang_id = JUDGE0_LANGUAGES.get(language, 71)
            execution = await run_code_judge0(code, lang_id)
            result["execution"] = execution
        
   #     // Save code generation record
        try:
            supabase.table("code_generations").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "language": language,
                "prompt": prompt,
                "code": code,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to save code generation record: {e}")
        
        return result

async def search_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle web search requests"""
 #   // Extract query from prompt
    query = prompt
    if "search for" in prompt.lower():
        query = prompt.lower().split("search for", 1)[1].strip()
    elif "look up" in prompt.lower():
        query = prompt.lower().split("look up", 1)[1].strip()
    elif "find" in prompt.lower():
        query = prompt.lower().split("find", 1)[1].strip()
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Searching..."})
            try:
                result = await duckduckgo_search(query)
                yield sse({
                    "type": "search_results",
                    "query": query,
                    "results": result
                })
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Search failed: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
     #   // Non-streaming version
        return await duckduckgo_search(query)

async def new_chat_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle new chat creation"""
    cid = str(uuid.uuid4())

    try:
        supabase.table("conversations").insert({
            "id": cid,
            "user_id": user_id,
            "title": prompt[:50] if len(prompt) > 50 else prompt,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to create new chat: {e}")

    if stream:
        async def event_generator():
            yield sse({
                "type": "new_chat",
                "conversation_id": cid
            })
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
        return {"conversation_id": cid}

async def send_message_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle sending a message to a conversation"""
 #   // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
    if not conv_id:
   #     // Get the most recent conversation
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        if conv_response.data:
            conv_id = conv_response.data[0]["id"]
    
    if not conv_id:
        return {"error": "No conversation found"}
    
#    // Extract message
    message = prompt
    if "message:" in prompt.lower():
        message = prompt.lower().split("message:", 1)[1].strip()
    
#    // Save user message
    msg_id = str(uuid.uuid4())
    try:
        supabase.table("messages").insert({
            "id": msg_id,
            "conversation_id": conv_id,
            "role": "user",
            "content": message,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save user message: {e}")
    
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Processing message..."})
            try:
          #      // Get conversation history
                msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conv_id).order("created_at").execute()
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

             #   // Stream the reply
                for char in reply:
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.01) # // Small delay for streaming effect

             #   // Save assistant reply
                reply_id = str(uuid.uuid4())
                supabase.table("messages").insert({
                    "id": reply_id,
                    "conversation_id": conv_id,
                    "role": "assistant",
                    "content": reply,
                    "created_at": datetime.now().isoformat()
                }).execute()
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
       # // Non-streaming version
        try:
            msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conv_id).order("created_at").execute()
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

          #  // Save assistant reply
            reply_id = str(uuid.uuid4())
            supabase.table("messages").insert({
                "id": reply_id,
                "conversation_id": conv_id,
                "role": "assistant",
                "content": reply,
                "created_at": datetime.now().isoformat()
            }).execute()

            return {"reply": reply}
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise HTTPException(500, "Failed to process message")

async def list_chats_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle listing chats"""
    try:
        response = supabase.table("conversations").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        rows = response.data if response.data else []
        
        if stream:
            async def event_generator():
                yield sse({"type": "chats", "data": rows})
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
            return rows
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        return []

async def search_chats_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle searching chats"""
  #  // Extract query from prompt
    query = prompt
    if "search for" in prompt.lower():
        query = prompt.lower().split("search for", 1)[1].strip()
    elif "find" in prompt.lower():
        query = prompt.lower().split("find", 1)[1].strip()
    
    try:
        response = supabase.table("conversations").select("id, title").eq("user_id", user_id).ilike("title", f"%{query}%").order("updated_at", desc=True).execute()
        rows = response.data if response.data else []
        
        if stream:
            async def event_generator():
                yield sse({"type": "search_results", "query": query, "data": rows})
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
            return rows
    except Exception as e:
        logger.error(f"Failed to search chats: {e}")
        return []

async def pin_chat_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle pinning a chat"""
   # // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
    if not conv_id:
        return {"error": "No conversation ID provided"}
    
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", conv_id).execute()
        
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": "Chat pinned"})
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
            return {"status": "pinned"}
    except Exception as e:
        logger.error(f"Failed to pin chat: {e}")
        return {"error": "Failed to pin chat"}

async def archive_chat_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle archiving a chat"""
   # // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
    if not conv_id:
        return {"error": "No conversation ID provided"}
    
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", conv_id).execute()
        
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": "Chat archived"})
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
            return {"status": "archived"}
    except Exception as e:
        logger.error(f"Failed to archive chat: {e}")
        return {"error": "Failed to archive chat"}

async def move_folder_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle moving a chat to a folder"""
   # // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
  #  // Extract folder from prompt
    folder = None
    folder_match = re.search(r'folder[:\s]+(.+)', prompt.lower())
    if folder_match:
        folder = folder_match.group(1).strip()
    
    if not conv_id:
        return {"error": "No conversation ID provided"}
    
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", conv_id).execute()
        
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": f"Chat moved to {folder}"})
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
            return {"status": "moved", "folder": folder}
    except Exception as e:
        logger.error(f"Failed to move chat to folder: {e}")
        return {"error": "Failed to move chat to folder"}

async def share_chat_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle sharing a chat"""
  #  // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
    if not conv_id:
        return {"error": "No conversation ID provided"}
    
    token = uuid.uuid4().hex

    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", conv_id).execute()
        
        if stream:
            async def event_generator():
                yield sse({"type": "share_url", "url": f"/share/{token}"})
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
            return {"share_url": f"/share/{token}"}
    except Exception as e:
        logger.error(f"Failed to share chat: {e}")
        return {"error": "Failed to share chat"}

async def view_shared_chat_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle viewing a shared chat"""
#    // Extract token from prompt
    token = None
    token_match = re.search(r'token[:\s]+([a-f0-9]+)', prompt.lower())
    if token_match:
        token = token_match.group(1)
    
    if not token:
        return {"error": "No token provided"}
    
#    // In a real implementation, you would look up the shared chat in the database
#    // For now, we'll return a placeholder
    
    if stream:
        async def event_generator():
            yield sse({"type": "shared_chat", "title": "Shared Chat", "messages": []})
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
        return {
            "title": "Shared Chat",
            "messages": []
        }

async def edit_message_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle editing a message"""
#    // Extract message ID from prompt
    msg_id = None
    msg_match = re.search(r'message[:\s]+([a-f0-9-]+)', prompt.lower())
    if msg_match:
        msg_id = msg_match.group(1)
    
#    // Extract new content from prompt
    new_content = None
    content_match = re.search(r'content[:\s]+(.+)', prompt.lower())
    if content_match:
        new_content = content_match.group(1).strip()
    
    if not msg_id or not new_content:
        return {"error": "Message ID and new content required"}
    
    try:
        #// Get message
        msg_response = supabase.table("messages").select("id, role, conversation_id, created_at").eq("id", msg_id).execute()
        if not msg_response.data:
            return {"error": "message not found"}
        
        msg_row = msg_response.data[0]
        if msg_row["role"] != "user":
            return {"error": "only user messages can be edited"}

        conversation_id = msg_row["conversation_id"]
        edited_at = msg_row["created_at"]

   #     // Update message content
        supabase.table("messages").update({
            "content": new_content
        }).eq("id", msg_id).execute()

  #      // Delete all assistant messages after this message
        supabase.table("messages").delete().eq("conversation_id", conversation_id).gt("created_at", edited_at).eq("role", "assistant").execute()
        
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": "Message edited"})
                yield sse({"type": "conversation_id", "id": conversation_id})
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
            return {
                "status": "edited",
                "conversation_id": conversation_id
            }
    except Exception as e:
        logger.error(f"Failed to edit message: {e}")
        return {"error": "Failed to edit message"}

async def regenerate_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle regenerating a response"""
#    // Extract conversation ID from prompt
    conv_id = None
    conv_match = re.search(r'conversation[:\s]+([a-f0-9-]+)', prompt.lower())
    if conv_match:
        conv_id = conv_match.group(1)
    
    if not conv_id:
 #       // Get the most recent conversation
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        if conv_response.data:
            conv_id = conv_response.data[0]["id"]
    
    if not conv_id:
        return {"error": "No conversation found"}
    
#    // Get the last user message
    msg_response = supabase.table("messages").select("content").eq("conversation_id", conv_id).eq("role", "user").order("created_at", desc=True).limit(1).execute()
    if not msg_response.data:
        return {"error": "No user message found"}
    
    last_user_message = msg_response.data[0]["content"]
    
#    // Delete the last assistant message
    last_assistant_msg = supabase.table("messages").select("id").eq("conversation_id", conv_id).eq("role", "assistant").order("created_at", desc=True).limit(1).execute()
    if last_assistant_msg.data:
        supabase.table("messages").delete().eq("id", last_assistant_msg.data[0]["id"]).execute()
    
#    // Generate a new response
    if stream:
        async def event_generator():
            yield sse({"type": "starting", "message": "Regenerating response..."})
            try:
              #  // Get conversation history
                msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conv_id).order("created_at").execute()
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

               # // Stream the reply
                for char in reply:
                    yield sse({"type": "token", "text": char})
                    await asyncio.sleep(0.01)  #// Small delay for streaming effect

             #   // Save assistant reply
                reply_id = str(uuid.uuid4())
                supabase.table("messages").insert({
                    "id": reply_id,
                    "conversation_id": conv_id,
                    "role": "assistant",
                    "content": reply,
                    "created_at": datetime.now().isoformat()
                }).execute()
                
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Failed to regenerate response: {e}")
                yield sse({"type": "error", "message": str(e)})
        
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
        #// Non-streaming version
        try:
            msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conv_id).order("created_at").execute()
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

           # // Save assistant reply
            reply_id = str(uuid.uuid4())
            supabase.table("messages").insert({
                "id": reply_id,
                "conversation_id": conv_id,
                "role": "assistant",
                "content": reply,
                "created_at": datetime.now().isoformat()
            }).execute()

            return {"reply": reply}
        except Exception as e:
            logger.error(f"Failed to regenerate response: {e}")
            raise HTTPException(500, "Failed to regenerate response")

async def stop_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle stopping a stream"""
    task = active_streams.get(user_id)
    if task:
        task.cancel()
        del active_streams[user_id]
        
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": "Stream stopped"})
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
            return {"stopped": True}
    else:
        if stream:
            async def event_generator():
                yield sse({"type": "status", "message": "No active stream"})
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
            return {"stopped": False}

async def vision_history_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle getting vision history"""
    try:
        response = supabase.table("vision_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        rows = response.data if response.data else []
        
        if stream:
            async def event_generator():
                yield sse({"type": "vision_history", "data": rows})
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
            return rows
    except Exception as e:
        logger.error(f"Failed to get vision history: {e}")
        return []

async def get_user_info_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle getting user information"""
    try:
        user_response = supabase.table("users").select("*").eq("id", user_id).execute()
        user_data = user_response.data[0] if user_response.data else None
        
      #  // Get user's images count
        images_response = supabase.table("images").select("id", count="exact").eq("user_id", user_id).execute()
        images_count = images_response.count if hasattr(images_response, 'count') else 0
        
      #  // Get user's videos count
        videos_response = supabase.table("videos").select("id", count="exact").eq("user_id", user_id).execute()
        videos_count = videos_response.count if hasattr(videos_response, 'count') else 0
        
       # // Get user's conversations count
        conversations_response = supabase.table("conversations").select("id", count="exact").eq("user_id", user_id).execute()
        conversations_count = conversations_response.count if hasattr(conversations_response, 'count') else 0
        
        result = {
            "id": user_id,
            "email": user_data.get("email") if user_data else None,
            "created_at": user_data.get("created_at") if user_data else None,
            "last_seen": user_data.get("last_seen") if user_data else None,
            "stats": {
                "images": images_count,
                "videos": videos_count,
                "conversations": conversations_count
            }
        }
        
        if stream:
            async def event_generator():
                yield sse({"type": "user_info", "data": result})
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
            return result
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        error_result = {
            "id": user_id,
            "error": "Failed to get additional user data"
        }
        
        if stream:
            async def event_generator():
                yield sse({"type": "user_info", "data": error_result})
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
            return error_result

async def merge_user_data_handler(prompt: str, user_id: str, stream: bool = False):
    """Handle merging user data"""
  #  // This is a complex operation that requires authentication
  #  // For now, we'll just return an error message
    if stream:
        async def event_generator():
            yield sse({"type": "error", "message": "User data merging requires authentication. Please use the /user/merge endpoint directly."})
        
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
        return {"error": "User data merging requires authentication. Please use the /user/merge endpoint directly."}

def detect_intent(prompt: str) -> str:
    if not prompt:
        return "chat"

    p = prompt.lower()

    #// 🖼 Image generation
    if any(w in p for w in [
        "image of", "draw", "picture of", "generate image",
        "make me an image", "photo of", "art of"
    ]):
        return "image"

    #// 🖼 Image → Image
    if any(w in p for w in [
        "edit this image", "change this image",
        "modify image", "img2img"
    ]):
        return "img2img"

    #// 👁 Vision / analysis
    if any(w in p for w in [
        "analyze this image", "what is in this image",
        "describe this image", "vision"
    ]):
        return "vision"

   # // 🎙 Speech → Text
    if any(w in p for w in [
        "transcribe", "speech to text", "stt"
    ]):
        return "stt"

  #  // 🔊 Text → Speech
    if any(w in p for w in [
        "say this", "speak", "tts", "read this", "read aloud"
    ]):
        return "tts"

   # // 🎥 Video (future-ready)
    if any(w in p for w in [
        "video of", "make a video", "animation of", "clip of"
    ]):
        return "video"

  #  // 💻 Code
    if any(w in p for w in [
        "write code", "generate code", "python code",
        "javascript code", "fix this code"
    ]):
        return "code"

  #  // 🔍 Search
    if any(w in p for w in [
        "search", "look up", "find info", "who is", "what is"
    ]):
        return "search"

   # // 📄 Document Analysis
    if any(w in p for w in [
        "analyze document", "extract information", "summarize document",
        "document analysis", "extract entities", "find keywords"
    ]):
        return "document_analysis"
    
   # // 🌐 Translation
    if any(w in p for w in [
        "translate", "translation", "translate to", "in spanish", "in french",
        "in german", "in japanese", "in chinese"
    ]):
        return "translation"
    
   # // 😊 Sentiment Analysis
    if any(w in p for w in [
        "sentiment", "emotion", "feeling", "analyze sentiment", "mood"
    ]):
        return "sentiment_analysis"
    
  #  // 🕸️ Knowledge Graph
    if any(w in p for w in [
        "knowledge graph", "relationship map", "entity graph", "concept map"
    ]):
        return "knowledge_graph"
    
  #  // 🤖 Custom Model Training
    if any(w in p for w in [
        "train model", "custom model", "fine-tune", "model training"
    ]):
        return "custom_model"
    
  #  // 🔍 Code Review
    if any(w in p for w in [
        "review code", "code review", "analyze code", "code quality"
    ]):
        return "code_review"
    
  #  // 🔍 Multi-modal Search
    if any(w in p for w in [
        "search everything", "multimodal search", "search all", "comprehensive search"
    ]):
        return "multimodal_search"
    
   # // 🧠 AI Personalization
    if any(w in p for w in [
        "personalize ai", "ai personality", "custom ai behavior", "ai preferences"
    ]):
        return "ai_personalization"
    
   # // 📊 Data Visualization
    if any(w in p for w in [
        "create chart", "visualize data", "make graph", "data visualization"
    ]):
        return "data_visualization"
    
    #// 🎤 Voice Cloning
    if any(w in p for w in [
        "clone voice", "custom voice", "voice profile", "voice synthesis"
    ]):
        return "voice_cloning"
    
    #// 💬 Chat Operations
    if any(w in p for w in [
        "new chat", "start conversation", "create conversation"
    ]):
        return "new_chat"
    
    if any(w in p for w in [
        "send message", "add message", "reply to"
    ]):
        return "send_message"
    
    if any(w in p for w in [
        "list chats", "show conversations", "my conversations"
    ]):
        return "list_chats"
    
    if any(w in p for w in [
        "search chats", "find conversation", "search conversation"
    ]):
        return "search_chats"
    
    if any(w in p for w in [
        "pin chat", "pin conversation"
    ]):
        return "pin_chat"
    
    if any(w in p for w in [
        "archive chat", "archive conversation"
    ]):
        return "archive_chat"
    
    if any(w in p for w in [
        "move to folder", "change folder", "folder"
    ]):
        return "move_folder"
    
    if any(w in p for w in [
        "share chat", "share conversation"
    ]):
        return "share_chat"
    
    if any(w in p for w in [
        "view shared chat", "shared conversation"
    ]):
        return "view_shared_chat"
    
    if any(w in p for w in [
        "edit message", "change message"
    ]):
        return "edit_message"
    
    if any(w in p for w in [
        "regenerate", "regenerate response", "try again"
    ]):
        return "regenerate"
    
    if any(w in p for w in [
        "stop", "cancel", "abort"
    ]):
        return "stop"
    
    if any(w in p for w in [
        "vision history", "image history", "analysis history"
    ]):
        return "vision_history"
    
    if any(w in p for w in [
        "user info", "my profile", "my account"
    ]):
        return "get_user_info"
    
    if any(w in p for w in [
        "merge user data", "merge account", "merge profile"
    ]):
        return "merge_user_data"

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

#// Background task functions
async def generate_chat_response_background(
    task_id: str,
    user_id: str,
    conversation_id: str,
    message: str
):
    """Generate chat response in the background"""
    try:
      #  // Update task status
        task_manager.update_task_status(task_id, "processing")
        
       # // Get conversation history
        msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").execute()
        rows = msg_response.data if msg_response.data else []
        messages = [{"role": row["role"], "content": row["content"]} for row in rows]
        
       # // Build contextual prompt
        system_prompt = build_contextual_prompt(user_id, message)
        
        #// Prepare messages for API call
        api_messages = [
            {"role": "system", "content": system_prompt},
            *messages
        ]
        
       # // Call language model
        payload = {
            "model": CHAT_MODEL,
            "messages": api_messages,
            "max_tokens": 1500
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(GROQ_URL, headers=headers, json=payload)
            r.raise_for_status()
            response_data = r.json()
        
        assistant_reply = response_data["choices"][0]["message"]["content"]
        
        #// Store assistant message
        persist_message(user_id, conversation_id, "assistant", assistant_reply)
        
       # // Update task with result
        task_manager.update_task_status(
            task_id, 
            "completed",
            result={
                "reply": assistant_reply,
                "conversation_id": conversation_id
            }
        )
        
        #// Send WebSocket notification if connected
        await manager.send_personal_message(
            json.dumps({
                "type": "task_completed",
                "task_id": task_id,
                "result": {
                    "reply": assistant_reply,
                    "conversation_id": conversation_id
                }
            }),
            user_id
        )
    except Exception as e:
        logger.error(f"Background chat generation failed: {e}")
        task_manager.update_task_status(task_id, "failed", error=str(e))

async def generate_image_background(
    task_id: str,
    user_id: str,
    prompt: str,
    samples: int
):
    """Generate images in the background"""
    try:
        task_manager.update_task_status(task_id, "processing")
        
        result = await _generate_image_core(prompt, samples, user_id, return_base64=False)
        
        task_manager.update_task_status(
            task_id, 
            "completed",
            result=result
        )
        
       # // Send WebSocket notification if connected
        await manager.send_personal_message(
            json.dumps({
                "type": "task_completed",
                "task_id": task_id,
                "result": result
            }),
            user_id
        )
    except Exception as e:
        logger.error(f"Background image generation failed: {e}")
        task_manager.update_task_status(task_id, "failed", error=str(e))

#// Cleanup job for old tasks
@scheduler.scheduled_job('interval', hours=24)
async def cleanup_old_tasks():
    """Clean up tasks older than 7 days"""
    cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
    
    try:
        #// Delete from database
        supabase.table("background_tasks").delete().lt("created_at", cutoff_date).execute()
        
       # // Clean up memory
        old_task_ids = [
            task_id for task_id, task in task_manager.active_tasks.items()
            if task["created_at"] < cutoff_date
        ]
        
        for task_id in old_task_ids:
            del task_manager.active_tasks[task_id]
            
        logger.info(f"Cleaned up {len(old_task_ids)} old tasks")
    except Exception as e:
        logger.error(f"Failed to cleanup old tasks: {e}")

#// ---------- API Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}
    
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

   # // ✅ COOKIE USER
    user = await get_or_create_user(req, res)
    user_id = user.id
    
#// ---------- Chat endpoint ----------
@app.post("/chat")
async def chat_endpoint(request: Request, response: Response):
    """Main chat endpoint with user memory"""
   # // Get or create user with stable UUID
    user = await get_or_create_user(request, response)
    user_id = user.id
    
   # // Parse request
    body = await request.json()
    message = body.get("message", "")
    
    if not message:
        raise HTTPException(400, "message required")
    
  #  // Get or create conversation for this user
    conversation_id = get_or_create_conversation(user_id)
    
    #// Store user message
    persist_message(user_id, conversation_id, "user", message)
    
  #  // Build contextual prompt with user history
    system_prompt = build_contextual_prompt(user_id, message)
    
    #// Prepare messages for API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
  #  // Call language model
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 1500
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(GROQ_URL, headers=headers, json=payload)
        r.raise_for_status()
        response_data = r.json()
    
    assistant_reply = response_data["choices"][0]["message"]["content"]
    
  #  // Store assistant message
    persist_message(user_id, conversation_id, "assistant", assistant_reply)
    
    return {
        "reply": assistant_reply,
        "conversation_id": conversation_id,
        "user_id": user_id
    }

#// Background chat endpoint
@app.post("/chat/background")
async def chat_background(
    request: Request, 
    response: Response,
    background_tasks: BackgroundTasks
):
    """Start a chat response in the background"""
#    // Get user
    user = await get_or_create_user(request, response)
    user_id = user.id
    
 #   // Parse request
    body = await request.json()
    message = body.get("message", "")
    conversation_id = body.get("conversation_id")
    
    if not message:
        raise HTTPException(400, "message required")
    
 #   // Create or get conversation
    if not conversation_id:
        conversation_id = get_or_create_conversation(user_id)
    
  #  // Store user message
    persist_message(user_id, conversation_id, "user", message)
    
   # // Create background task
    task_id = task_manager.create_task(
        user_id=user_id,
        task_type="chat_response",
        params={
            "message": message,
            "conversation_id": conversation_id
        }
    )
    
 #   // Add to background tasks
    background_tasks.add_task(
        generate_chat_response_background,
        task_id,
        user_id,
        conversation_id,
        message
    )
    
    return {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "status": "queued"
    }

#// Background image generation endpoint
@app.post("/image/background")
async def image_background(
    request: Request, 
    response: Response,
    background_tasks: BackgroundTasks
):
    """Start image generation in the background"""
    user = await get_or_create_user(request, response)
    user_id = user.id
    
    body = await request.json()
    prompt = body.get("prompt", "")
    samples = max(1, int(body.get("samples", 1)))
    
    if not prompt:
        raise HTTPException(400, "prompt required")
    
 #   // Create background task
    task_id = task_manager.create_task(
        user_id=user_id,
        task_type="image_generation",
        params={
            "prompt": prompt,
            "samples": samples
        }
    )
    
  #  // Add to background tasks
    background_tasks.add_task(
        generate_image_background,
        task_id,
        user_id,
        prompt,
        samples
    )
    
    return {
        "task_id": task_id,
        "status": "queued"
    }

#// Task status endpoints
@app.get("/task/{task_id}")
async def get_task_status(request: Request, response: Response):
    """Get the status of a background task"""
    user = await get_or_create_user(request, response)
    user_id = user.id
    
    task_id = request.path_params["task_id"]
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(404, "Task not found")
    
    if task["user_id"] != user_id:
        raise HTTPException(403, "Not authorized to access this task")
    
    return task

@app.get("/tasks")
async def get_user_tasks(request: Request, response: Response):
    """Get all tasks for the current user"""
    user = await get_or_create_user(request, response)
    user_id = user.id
    
    return task_manager.get_user_tasks(user_id)

#// WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
           # // Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(user_id)

#// =========================================================
#// 🚀 UNIVERSAL MULTIMODAL ENDPOINT — /ask/universal
#// =========================================================

# First, let's fix the load_conversation_history function to use the correct order syntax
def load_conversation_history(user_id: str, limit: int = 20):
    """Load conversation history for a user"""
    try:
        # Get most recent conversation
        conv_response = supabase.table("conversations").select("id").eq("user_id", user_id).order("updated_at", desc=True).limit(1).execute()
        
        if not conv_response.data:
            return []
            
        conversation_id = conv_response.data[0]["id"]
            
        # Get recent messages from this conversation
        # Fix: Changed order("created_at", asc=True) to order("created_at")
        msg_response = supabase.table("messages").select("role, content").eq("conversation_id", conversation_id).order("created_at").limit(limit).execute()
            
        return [{"role": row["role"], "content": row["content"]} for row in msg_response.data]
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
    return []

# Now, let's fix the ask_universal function to handle the missing background_tasks table
@app.post("/ask/universal")
async def ask_universal(request: Request, response: Response):
    try:
        body = await request.json()
        prompt = body.get("prompt", "").strip()
        conversation_id = body.get("conversation_id")

        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required")

        # ----------------------------------
        # Resolve user
        # ----------------------------------
        user = await get_or_create_user(request, response)
        user_id = user.id

        # ----------------------------------
        # Get or create conversation
        # ----------------------------------
        if not conversation_id:
            conversation_id = await get_or_create_conversation(user_id)
        else:
            conv_check = await asyncio.to_thread(
                lambda: supabase
                .table("conversations")
                .select("id")
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )

            if not conv_check.data:
                conversation_id = await get_or_create_conversation(user_id)

        # ----------------------------------
        # Load history
        # ----------------------------------
        history = load_conversation_history(user_id, limit=20)

        # ----------------------------------
        # Load profile
        # ----------------------------------
        profile = get_user_profile(user_id)
        personality = profile.get("personality", "friendly")
        nickname = profile.get("nickname", f"User{user_id[:8]}")

        # ----------------------------------
        # Save user message
        # ----------------------------------
        await asyncio.to_thread(
            lambda: supabase.table("messages").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "user",
                "content": prompt,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
        )

        await asyncio.to_thread(
            lambda: supabase.table("conversations").update({
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()
        )

        # ----------------------------------
        # Build system prompt
        # ----------------------------------
        system_prompt = (
            PERSONALITY_MAP.get(personality, PERSONALITY_MAP["friendly"])
            + f"\nUser name: {nickname}\n"
            "You are a helpful AI assistant.\n"
            "Maintain memory and context.\n"
        )

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            if msg["role"] != "system":
                messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        # ----------------------------------
        # Call Groq API
        # ----------------------------------
        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": 1500
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(GROQ_URL, headers=headers, json=payload)
            r.raise_for_status()
            response_data = r.json()

        assistant_reply = response_data["choices"][0]["message"]["content"]

        # ----------------------------------
        # Save assistant message
        # ----------------------------------
        await asyncio.to_thread(
            lambda: supabase.table("messages").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": assistant_reply,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
        )

        return {
            "status": "completed",
            "reply": assistant_reply,
            "conversation_id": conversation_id,
            "user_id": user_id
        }

    except Exception as e:
        logger.error(f"/ask/universal failed: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")
        
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

   # // Get message
    try:
        msg_response = supabase.table("messages").select("id, role, conversation_id, created_at").eq("id", message_id).execute()
        if not msg_response.data:
            raise HTTPException(404, "message not found")
        
        msg_row = msg_response.data[0]
        if msg_row["role"] != "user":
            raise HTTPException(403, "only user messages can be edited")

        conversation_id = msg_row["conversation_id"]
        edited_at = msg_row["created_at"]

      #  // Update message content
        supabase.table("messages").update({
            "content": new_text
        }).eq("id", message_id).execute()

        #// 🔥 DELETE ALL ASSISTANT MESSAGES AFTER THIS MESSAGE
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
async def stream_endpoint(request: Request):
    async def event_generator():
        for i in range(1, 6):
           # // Check if client disconnected
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
    
#// -----------------------------
#// Stop endpoint
#// -----------------------------
@app.post("/stop")
async def stop(request: Request, response: Response):
    user = await get_or_create_user(request, response)
    user_id = user.id
    
    task = active_streams.get(user_id)
    if task:
        task.cancel()
        del active_streams[user_id]
        return {"stopped": True}
    return {"stopped": False}
    
#// -----------------------------
#// Regenerate endpoint
#// -----------------------------
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

#    // ✅ COOKIE USER
    user = await get_or_create_user(req, res)
    user_id = user.id

 #   // ✅ CANCEL EXISTING STREAM (IF ANY)
    old_task = active_streams.get(user_id)
    if old_task and not old_task.done():
        old_task.cancel()

    async def event_generator():
       # // ✅ REGISTER NEW STREAM
        task = asyncio.current_task()
        active_streams[user_id] = task

     #   // Also register in database
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
        #    // --- IMAGE (OPTIONAL) ---
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

           # // --- CHAT ---
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

        #    // --- TTS (OPTIONAL) ---
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
       #     // ✅ CLEANUP
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

#// ---------- Img2Img (DALL·E edits) ----------
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
            files = {"image": (file.filename, content, file.content_type or "video/mpeg")}
            data = {"prompt": prompt, "n": 1, "size": "1024x1024"}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            r = await client.post("https://api.openai.com/v1/images/edits", headers=headers, files=files, data=data)
            r.raise_for_status()
            jr = r.json()
            for d in jr.get("data", []):
                b64 = d.get("b64_json")
                if b64:
                    fname = unique_filename("png")
             #       // Upload to anonymous folder
                    image_bytes = base64.b64decode(b64)
                    supabase_fname = f"anonymous/{fname}"
                    upload_image_to_supabase(image_bytes, supabase_fname, user_id)
                    signed = supabase.storage.from_("ai-images").create_signed_url(supabase_fname, 60*60)
                    urls.append(signed["signedURL"])
    except Exception:
        logger.exception("img2img DALL-E edit failed")
        raise HTTPException(400, "img2img failed")

    return {"provider": "dalle3-edit", "images": urls}
    
#// ---------- TTS ----------
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

  #  // Try JSON first
    try:
        data = await request.json()
        text = data.get("text", None)
    except Exception:
  #      // Fallback: read raw text from body
        text = (await request.body()).decode("utf-8")

    if not text or not text.strip():
        raise HTTPException(400, "Missing 'text' in request")

    payload = {
        "model": "tts-1", 
        "voice": "alloy", # // default voice
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

#    // ✅ COOKIE USER
    user = await get_or_create_user(req, res)
    user_id = user.id

    async def audio_streamer():
  #      // ✅ REGISTER STREAM TASK
        task = asyncio.current_task()
        active_streams[user_id] = task

  #      // Also register in database
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
            #// Audio streams cannot emit JSON errors safely mid-stream

        finally:
           # // ✅ CLEANUP
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

#// ---------- Vision analyze ----------
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

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
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
        # Fix: Define pixels variable before using it
        pixels = np_img.reshape(-1, 3)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=0).fit(pixels)
        hex_colors = [
            '#%02x%02x%02x' % tuple(map(int, c))
            for c in kmeans.cluster_centers_
        ]
    except ImportError:
        logger.warning("sklearn not installed, skipping color analysis")
    except Exception:
        logger.exception("Color clustering failed")

    # =========================
    # 4️⃣ UPLOAD TO SUPABASE
    # =========================
    raw_path = f"anonymous/raw/{uuid.uuid4().hex}.png"
    ann_path = f"anonymous/annotated/{uuid.uuid4().hex}.png"

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
            "created_at": datetime.now(timezone.utc).isoformat()
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

#// Update the vision_history function to use the new user model
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

#// ---------- Code generation ----------
@app.post("/code")
async def code_gen(req: Request, res: Response):
    body = await req.json()
    prompt = body.get("prompt", "")
    language = body.get("language", "python").lower()
    run_flag = bool(body.get("run", False))
    
    #// Get the user with our new ID system
    user = await get_or_create_user(req, res)
    user_id = user.id

    if not prompt:
        raise HTTPException(400, "prompt required")

   # // Generate code using Groq
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
        "generated_code": code,
        "user_id": user_id  # Include user_id in response
    }

   # // ✅ Run via Judge0
    if run_flag:
        lang_id = JUDGE0_LANGUAGES.get(language, 71)
        execution = await run_code_judge0(code, lang_id)
        response["execution"] = execution

  #  // Save code generation record (with error handling for missing table)
    try:
        supabase.table("code_generations").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "language": language,
            "prompt": prompt,
            "code": code,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        logger.error(f"Failed to save code generation record (table might not exist): {e}")
       # // Don't fail the request, just log the error
    return response

@app.get("/search")
async def duck_search(q: str = Query(..., min_length=1)):
    try:
        return await duckduckgo_search(q)
    except httpx.HTTPStatusError as e:
        logger.exception("DuckDuckGo returned HTTP error")
        raise HTTPException(502, "duckduckgo_error")
    except Exception:
        logger.exception("DuckDuckGo search failed")
        raise HTTPException(500, "search_failed")

#// ---------- STT ----------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(500, "Missing OPENAI_API_KEY")

    url = "https://api.openai.com/v1/audio/transcriptions"

   # // Whisper API requires multipart/form-data, NOT JSON
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

#// ----------------------------------
#// NEW CHAT
#// ----------------------------------

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

#// ----------------------------------
#// LIST CHATS
#// ----------------------------------
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

#// ----------------------------------
#// SEARCH CHATS
#// ----------------------------------
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

#// ----------------------------------
#// PIN / ARCHIVE
#// ----------------------------------
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

#// ----------------------------------
#// FOLDER
#// ----------------------------------
@app.post("/chat/{id}/folder")
async def move_folder(id: str, folder: Optional[str] = None):
    try:
        supabase.table("conversations").update({
            "updated_at": datetime.now().isoformat()
        }).eq("id", id).execute()
    except Exception as e:
        logger.error(f"Failed to move chat to folder: {e}")
    return {"status": "moved"}

#// ----------------------------------
#// SHARE CHAT
#// ----------------------------------
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

#// ----------------------------------
#// VIEW SHARED CHAT (READ ONLY)
#// ----------------------------------
@app.get("/share/{token}")
async def view_shared_chat(token: str):
   # // In a real implementation, you would store share tokens in the database
   # // For now, we'll return a placeholder
    return {
        "title": "Shared Chat",
        "messages": []
    }

#// Example FastAPI endpoint that fires AI response in the background
from fastapi.responses import StreamingResponse

@app.post("/chat/stream/{conversation_id}/{user_id}")
async def chat_stream_endpoint(conversation_id: str, user_id: str, messages: list):
    """
    Streams AI response tokens to the client, saving them in Supabase in real-time.
    """
    async def event_generator():
        async for token_sse in stream_llm(user_id, conversation_id, messages):
            yield token_sse  # only yield here, no return

    #// Return StreamingResponse from the endpoint, not inside the generator
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

#// ---------- Dedicated Endpoints for Advanced Features ----------
@app.post("/document/analyze")
async def document_analysis_endpoint(request: DocumentAnalysisRequest, req: Request, res: Response):
    """Analyze documents for key information"""
    user = await get_or_create_user(req, res)
    return await document_analysis(
        f"document: {request.text}\nanalysis_type: {request.analysis_type}",
        user.id,
        False
    )

@app.post("/translation")
async def translation_endpoint(request: TranslationRequest, req: Request, res: Response):
    """Translate text between languages"""
    user = await get_or_create_user(req, res)
    return await translate_text(
        f"translate: {request.text} to {request.target_lang}",
        user.id,
        False
    )

@app.post("/sentiment/analyze")
async def sentiment_analysis_endpoint(request: SentimentAnalysisRequest, req: Request, res: Response):
    """Analyze sentiment of text"""
    user = await get_or_create_user(req, res)
    return await analyze_sentiment(
        f"sentiment: {request.text}",
        user.id,
        False
    )

@app.post("/knowledge/graph")
async def knowledge_graph_endpoint(request: KnowledgeGraphRequest, req: Request, res: Response):
    """Create and visualize a knowledge graph"""
    user = await get_or_create_user(req, res)
    entities_str = ", ".join(request.entities)
    return await create_knowledge_graph_endpoint(
        f"entities: {entities_str}\nrelationship: {request.relationship_type}",
        user.id,
        False
    )

@app.post("/model/train")
async def custom_model_endpoint(request: CustomModelRequest, req: Request, res: Response):
    """Train a custom model"""
    user = await get_or_create_user(req, res)
    return await train_custom_model(
        f"data: {request.training_data}\ntype: {request.model_type}",
        user.id,
        False
    )

@app.post("/code/review")
async def code_review_endpoint(request: CodeReviewRequest, req: Request, res: Response):
    """Review code for issues and improvements"""
    user = await get_or_create_user(req, res)
    focus_str = ", ".join(request.focus_areas)
    return await review_code(
        f"code: ```{request.code}``\nlanguage: {request.language}\nfocus: {focus_str}",
        user.id,
        False
    )

@app.post("/search/multimodal")
async def multimodal_search_endpoint(request: MultimodalSearchRequest, req: Request, res: Response):
    """Search across text, images, and videos"""
    user = await get_or_create_user(req, res)
    types_str = ", ".join(request.search_types)
    filters_str = ", ".join(f"{k}: {v}" for k, v in request.filters.items())
    return await multimodal_search(
        f"query: {request.query}\ntypes: {types_str}\nfilters: {filters_str}",
        user.id,
        False
    )

@app.post("/user/preferences")
async def set_preferences(request: Request, response: Response):
    """Store user preferences"""
    user = await get_or_create_user(request, response)
    user_id = user.id
    
    body = await request.json()
    preferences = body.get("preferences", {})
    
  #  // Upsert preferences
    try:
        existing = supabase.table("profiles").select("id").eq("id", user_id).execute()
        
        if existing.data:
          #  // Update existing profile
            supabase.table("profiles").update({
                "preferences": preferences,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", user_id).execute()
        else:
          #  // Create new profile
            supabase.table("profiles").insert({
                "id": user_id,
                "preferences": preferences,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
    except Exception as e:
        logger.error(f"Failed to save preferences: {e}")
        raise HTTPException(500, "Failed to save preferences")
    
    return {"status": "success"}

@app.post("/ai/personalize")
async def ai_personalization_endpoint(request: PersonalizationRequest, req: Request, res: Response):
    """Customize AI behavior based on user preferences"""
    user = await get_or_create_user(req, res)
    prefs_str = ", ".join(f"{k}: {v}" for k, v in request.user_preferences.items())
    behavior_str = ", ".join(f"{k}: {v}" for k, v in request.behavior_patterns.items())
    return await personalize_ai(
        f"preferences: {prefs_str}\nbehavior: {behavior_str}",
        user.id,
        False
    )

@app.post("/data/visualize")
async def data_visualization_endpoint(request: DataVisualizationRequest, req: Request, res: Response):
    """Generate charts and graphs from data"""
    user = await get_or_create_user(req, res)
    options_str = ", ".join(f"{k}: {v}" for k, v in request.options.items())
    return await visualize_data(
        f"data: {request.data}\nchart: {request.chart_type}\noptions: {options_str}",
        user.id,
        False
    )

@app.post("/voice/clone")
async def voice_cloning_endpoint(request: VoiceCloningRequest, req: Request, res: Response):
    """Create custom voice profiles for TTS"""
    user = await get_or_create_user(req, res)
    return await clone_voice(
        f"sample: {request.voice_sample}\ntext: {request.text}\nname: {request.voice_name}",
        user.id,
        False
    )

#// Add a new endpoint to get user information
@app.get("/user/info")
async def get_user_info(req: Request, res: Response):
    user = await get_or_create_user(req, res)
    user_id = user.id
    
#    // Get additional user data from database
    try:
        user_response = supabase.table("users").select("*").eq("id", user_id).execute()
        user_data = user_response.data[0] if user_response.data else None
        
  #      // Get user's images count
        images_response = supabase.table("images").select("id", count="exact").eq("user_id", user_id).execute()
        images_count = images_response.count if hasattr(images_response, 'count') else 0
        
   #     // Get user's videos count
        videos_response = supabase.table("videos").select("id", count="exact").eq("user_id", user_id).execute()
        videos_count = videos_response.count if hasattr(videos_response, 'count') else 0
        
  #      // Get user's conversations count
        conversations_response = supabase.table("conversations").select("id", count="exact").eq("user_id", user_id).execute()
        conversations_count = conversations_response.count if hasattr(conversations_response, 'count') else 0
        
        return {
            "id": user.id,
            "email": user.email,
            "anonymous": user.anonymous,
            "created_at": user_data.get("created_at") if user_data else None,
            "last_seen": user_data.get("last_seen") if user_data else None,
            "stats": {
                "images": images_count,
                "videos": videos_count,
                "conversations": conversations_count
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

#// Add a new endpoint to merge anonymous user data with logged-in user
@app.post("/user/merge")
async def merge_user_data(req: Request, res: Response):
    
    session_token = req.cookies.get("session_token")
    if not session_token:
        raise HTTPException(400, "No session token found")
    
#    // Get the logged-in user from JWT token
    auth_header = req.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(400, "No authorization token found")
    
    token = auth_header.split(" ")[1]
    try:
   #     // Verify JWT token with frontend Supabase
        if not frontend_supabase:
            raise HTTPException(500, "Frontend Supabase not configured")
        
        user_response = frontend_supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(401, "Invalid token")
        
        logged_in_id = user_response.user.id
        
    #    // Get the anonymous user ID from session token
        try:
            visitor_response = supabase.table("users").select("id").eq("session_token", session_token).execute()
            if not visitor_response.data:
                raise HTTPException(404, "Anonymous user not found")
            
            anonymous_id = visitor_response.data[0]["id"]
            
         #   // Merge data in the backend
            try:
              #  // Update all records from anonymous user to logged-in user
                tables_to_merge = ["images", "videos", "conversations", "messages", "memory", "vision_history", "code_generations"]
                
                for table in tables_to_merge:
                    supabase.table(table).update({"user_id": logged_in_id}).eq("user_id", anonymous_id).execute()
                
             #   // Create or update the logged-in user in the backend
                existing_user = supabase.table("users").select("*").eq("id", logged_in_id).execute()
                if not existing_user.data:
                    supabase.table("users").insert({
                        "id": logged_in_id,
                        "email": user_response.user.email,
                        "created_at": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat()
                    }).execute()
                else:
                    supabase.table("users").update({
                        "last_seen": datetime.now().isoformat()
                    }).eq("id", logged_in_id).execute()
                
             #   // Update the cookie to the logged-in user ID
                res.set_cookie(
                    key="session_token",
                    value=logged_in_id,
                    httponly=True,
                    samesite="lax",
                    max_age=60 * 60 * 24 * 30  #// 30 days
                )
                
                return {"status": "success", "message": "User data merged successfully"}
            except Exception as e:
                logger.error(f"Failed to merge user data: {e}")
                raise HTTPException(500, f"Failed to merge user data: {str(e)}")
        except Exception as e:
            logger.error(f"Error finding anonymous user: {e}")
            raise HTTPException(404, f"Anonymous user not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error verifying JWT token: {e}")
        raise HTTPException(401, f"Invalid token: {str(e)}")

# These should be at the end of the file, after all endpoints are defined
@app.on_event("startup")
async def startup_event():
    # Start the scheduler
    scheduler.start()
    
    # Check available models
    await check_available_models()
    
    # Check bucket visibility
    if not check_bucket_visibility():
        logger.warning("Storage buckets may not be public. Images and videos might not load on frontend.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    
@app.post("/check")
async def check():
    conv_check = await asyncio.to_thread(run_check)
    return conv_check

# Update the video endpoint with real RunwayML Gen-2 implementation
@app.post("/video")
async def generate_video(request: Request):
    """
    Generate videos from text prompts using available video generation APIs.
    Returns URLs to the generated videos.
    """
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    user = await get_or_create_user(request, Response())
    user_id = user.id
    
    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1
    
    if not prompt:
        raise HTTPException(400, "prompt required")
    
    # Content moderation check
    is_flagged = await nsfw_check(prompt)
    if is_flagged:
        raise HTTPException(
            status_code=400, 
            detail="Video generation prompt violates content policy."
        )
    
    # Check for cached result
    # Try different providers in order of preference
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
    
    provider = None
    if STABILITY_API_KEY:
        provider = "stability-ai"
    elif HF_API_KEY:
        provider = "huggingface-damo"
    elif RUNWAYML_API_KEY:
        provider = "runwayml-gen2"
    
    if provider:
        cached = get_cached_result(prompt, provider)
        if cached:
            return cached
    
    try:
        result = await generate_video_internal(prompt, samples, user_id)
        cache_result(prompt, result["provider"], result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation failed")
        raise HTTPException(500, "Video generation failed")

# Add a new endpoint for video-to-video generation
@app.post("/video/img2vid")
async def img2vid(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    duration: int = Form(4)
):
    """
    Generate a video from an image and text prompt using RunwayML Gen-2.
    """
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
    if not RUNWAYML_API_KEY:
        raise HTTPException(500, "RUNWAYML_API_KEY not configured")
    
    user = await get_or_create_user(request, Response())
    user_id = user.id
    
    if not prompt:
        raise HTTPException(400, "prompt required")
    
    # Read and validate the image
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "empty file")
    
    try:
        # Upload the image to a temporary location
        image_filename = f"temp_{uuid.uuid4().hex[:8]}.png"
        image_path = f"temp/{image_filename}"
        
        supabase.storage.from_("ai-images").upload(
            path=image_path,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )
        
        # Get a public URL for the image
        image_url = get_public_url("ai-images", image_path)
        
        # Create a task for image-to-video generation
        headers = {
            "Authorization": f"Bearer {RUNWAYML_API_KEY}",
            "Content-Type": "application/json"
        }
        
        task_payload = {
            "model": "gen-2",
            "image_prompt": image_url,
            "text_prompt": prompt,
            "watermark": False,
            "duration": min(max(duration, 1), 14),  # Between 1-14 seconds
            "ratio": "16:9",
            "upscale": True
        }
        
        # Submit the task
        async with httpx.AsyncClient(timeout=60.0) as client:
            task_response = await client.post(
                "https://api.runwayml.com/v1/video_tasks",
                headers=headers,
                json=task_payload
            )
            task_response.raise_for_status()
            task_data = task_response.json()
            task_id = task_data.get("id")
            
            if not task_id:
                raise HTTPException(500, "Failed to create video generation task")
            
            # Poll for task completion
            max_attempts = 60  # Maximum polling attempts (5 minutes)
            video_url = None
            
            for attempt in range(max_attempts):
                # Check task status
                status_response = await client.get(
                    f"https://api.runwayml.com/v1/video_tasks/{task_id}",
                    headers=headers
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                status = status_data.get("status")
                
                if status == "SUCCEEDED":
                    video_url = status_data.get("output", {}).get("url")
                    break
                elif status == "FAILED":
                    error_message = status_data.get("failure_reason", "Unknown error")
                    raise HTTPException(500, f"Video generation failed: {error_message}")
                
                # Wait before polling again
                await asyncio.sleep(5)
            
            if not video_url:
                raise HTTPException(500, "Video generation timed out")
            
            # Download the video
            video_response = await client.get(video_url)
            video_response.raise_for_status()
            video_bytes = video_response.content
            
            # Upload to Supabase
            filename = f"{uuid.uuid4().hex[:8]}.mp4"
            storage_path = f"anonymous/{filename}"
            
            supabase.storage.from_("ai-videos").upload(
                path=storage_path,
                file=video_bytes,
                file_options={"content-type": "video/mp4"}
            )
            
            # Save video record
            try:
                supabase.table("videos").insert({
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "video_path": storage_path,
                    "prompt": prompt,
                    "provider": "runwayml-gen2-img2vid",
                    "created_at": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save video record: {e}")
            
            # Get public URL
            public_url = get_public_url("ai-videos", storage_path)
            
            # Clean up the temporary image
            try:
                supabase.storage.from_("ai-images").remove([image_path])
            except:
                pass
            
            return {
                "provider": "runwayml-gen2-img2vid",
                "video": {"url": public_url, "type": "video/mp4"}
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image-to-video generation failed")
        raise HTTPException(500, "Image-to-video generation failed")

# Add a video streaming endpoint for real-time progress updates
@app.post("/video/stream")
async def generate_video_stream(req: Request, res: Response):
    """
    Stream video generation progress to the client.
    """
    body = await req.json()
    prompt = body.get("prompt", "").strip()
    
    try:
        samples = max(1, int(body.get("samples", 1)))
    except Exception:
        samples = 1
    
    if not prompt:
        raise HTTPException(400, "prompt required")
    
    # Get user
    user = await get_or_create_user(req, res)
    user_id = user.id
    
    # Content moderation check
    is_flagged = await nsfw_check(prompt)
    if is_flagged:
        raise HTTPException(
            status_code=400, 
            detail="Video generation prompt violates content policy."
        )
    
    # Check for available video generation APIs
    STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    RUNWAYML_API_KEY = os.getenv("RUNWAYML_API_KEY")
    
    if not STABILITY_API_KEY and not HF_API_KEY and not RUNWAYML_API_KEY:
        # No API keys configured, use placeholder
        async def event_generator():
            yield sse({"status": "starting", "message": "No video generation API configured. Using placeholder."})
            
            # Generate placeholder video
            try:
                result = await generate_placeholder_video(prompt, samples, user_id)
                yield sse({
                    "status": "completed",
                    "message": "Placeholder video generated",
                    "videos": result["videos"]
                })
            except Exception as e:
                logger.error(f"Placeholder video generation failed: {e}")
                yield sse({"status": "error", "message": str(e)})
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Determine which service to use
    if STABILITY_API_KEY:
        service = "stability-ai"
    elif HF_API_KEY:
        service = "huggingface-damo"
    else:
        service = "runwayml-gen2"
    
    async def event_generator():
        # Register stream task
        task = asyncio.current_task()
        active_streams[user_id] = task
        
        # Register in database
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
            yield sse({"status": "starting", "message": f"Initializing video generation with {service}..."})
            
            video_urls = []
            
            for i in range(samples):
                yield sse({"status": "progress", "message": f"Generating video {i+1}/{samples}..."})
                
                if service == "stability-ai":
                    # Use Stability AI
                    try:
                        headers = {
                            "Authorization": f"Bearer {STABILITY_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "prompt": prompt,
                            "width": 1024,
                            "height": 576,  # 16:9 aspect ratio
                            "samples": 1,
                            "num_frames": 25,  # Number of frames in the video
                            "seed": random.randint(0, 4294967295)
                        }
                        
                        # Submit the request
                        async with httpx.AsyncClient(timeout=120.0) as client:
                            response = await client.post(
                                "https://api.stability.ai/v1/generation/stable-video-diffusion/text-to-video",
                                headers=headers,
                                json=payload
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            # Get the video generation ID
                            generation_id = result.get("id")
                            if not generation_id:
                                yield sse({"status": "error", "message": "Failed to get video generation ID"})
                                continue
                            
                            # Poll for completion
                            video_url = None
                            max_attempts = 60  # Maximum polling attempts (5 minutes)
                            
                            for attempt in range(max_attempts):
                                # Check generation status
                                status_response = await client.get(
                                    f"https://api.stability.ai/v1/generation/stable-video-diffusion/text-to-video/{generation_id}",
                                    headers=headers
                                )
                                status_response.raise_for_status()
                                status_data = status_response.json()
                                
                                status = status_data.get("status")
                                
                                if status == "completed":
                                    video_url = status_data.get("video")
                                    break
                                elif status == "failed":
                                    error_message = status_data.get("error", "Unknown error")
                                    yield sse({"status": "error", "message": f"Video generation failed: {error_message}"})
                                    break
                                
                                # Update progress
                                progress = min(int((attempt / max_attempts) * 100), 95)
                                yield sse({"status": "progress", "message": f"Processing video... {progress}%"})
                                
                                # Wait before polling again
                                await asyncio.sleep(5)
                            
                            if not video_url:
                                yield sse({"status": "error", "message": "Video generation timed out"})
                                continue
                            
                            yield sse({"status": "progress", "message": "Downloading video..."})
                            
                            # Download the video
                            video_response = await client.get(video_url)
                            video_response.raise_for_status()
                            video_bytes = video_response.content
                            
                            # Upload to Supabase
                            filename = f"{uuid.uuid4().hex[:8]}.mp4"
                            storage_path = f"anonymous/{filename}"
                            
                            supabase.storage.from_("ai-videos").upload(
                                path=storage_path,
                                file=video_bytes,
                                file_options={"content-type": "video/mp4"}
                            )
                            
                            # Save video record
                            try:
                                supabase.table("videos").insert({
                                    "id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "video_path": storage_path,
                                    "prompt": prompt,
                                    "provider": "stability-ai",
                                    "created_at": datetime.now().isoformat()
                                }).execute()
                            except Exception as e:
                                logger.error(f"Failed to save video record: {e}")
                            
                            # Get public URL
                            public_url = get_public_url("ai-videos", storage_path)
                            video_urls.append(public_url)
                    except Exception as e:
                        logger.error(f"Stability AI video generation failed: {e}")
                        yield sse({"status": "error", "message": f"Video generation failed: {str(e)}"})
                        continue
                
                elif service == "huggingface-damo":
                    # Use Hugging Face
                    try:
                        headers = {
                            "Authorization": f"Bearer {HF_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "num_inference_steps": 25,
                                "guidance_scale": 7.5
                            }
                        }
                        
                        # Submit the request
                        async with httpx.AsyncClient(timeout=120.0) as client:
                            response = await client.post(
                                "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b",
                                headers=headers,
                                json=payload
                            )
                            response.raise_for_status()
                            
                            # The response should contain the video data
                            video_bytes = response.content
                            
                            # Upload to Supabase
                            filename = f"{uuid.uuid4().hex[:8]}.mp4"
                            storage_path = f"anonymous/{filename}"
                            
                            supabase.storage.from_("ai-videos").upload(
                                path=storage_path,
                                file=video_bytes,
                                file_options={"content-type": "video/mp4"}
                            )
                            
                            # Save video record
                            try:
                                supabase.table("videos").insert({
                                    "id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "video_path": storage_path,
                                    "prompt": prompt,
                                    "provider": "huggingface-damo",
                                    "created_at": datetime.now().isoformat()
                                }).execute()
                            except Exception as e:
                                logger.error(f"Failed to save video record: {e}")
                            
                            # Get public URL
                            public_url = get_public_url("ai-videos", storage_path)
                            video_urls.append(public_url)
                    except Exception as e:
                        logger.error(f"Hugging Face video generation failed: {e}")
                        yield sse({"status": "error", "message": f"Video generation failed: {str(e)}"})
                        continue
                
                else:  # runwayml-gen2
                    # Use RunwayML
                    try:
                        headers = {
                            "Authorization": f"Bearer {RUNWAYML_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        task_payload = {
                            "model": "gen-2",
                            "text_prompt": prompt,
                            "watermark": False,
                            "duration": 4,
                            "ratio": "16:9",
                            "upscale": True
                        }
                        
                        # Submit the task
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            task_response = await client.post(
                                "https://api.runwayml.com/v1/video_tasks",
                                headers=headers,
                                json=task_payload
                            )
                            task_response.raise_for_status()
                            task_data = task_response.json()
                            task_id = task_data.get("id")
                            
                            if not task_id:
                                yield sse({"status": "error", "message": "Failed to create video generation task"})
                                continue
                            
                            # Poll for task completion
                            max_attempts = 60
                            video_url = None
                            
                            for attempt in range(max_attempts):
                                # Check task status
                                status_response = await client.get(
                                    f"https://api.runwayml.com/v1/video_tasks/{task_id}",
                                    headers=headers
                                )
                                status_response.raise_for_status()
                                status_data = status_response.json()
                                
                                status = status_data.get("status")
                                
                                if status == "SUCCEEDED":
                                    video_url = status_data.get("output", {}).get("url")
                                    break
                                elif status == "FAILED":
                                    error_message = status_data.get("failure_reason", "Unknown error")
                                    yield sse({"status": "error", "message": f"Video generation failed: {error_message}"})
                                    break
                                
                                # Update progress
                                progress = min(int((attempt / max_attempts) * 100), 95)
                                yield sse({"status": "progress", "message": f"Processing video... {progress}%"})
                                
                                # Wait before polling again
                                await asyncio.sleep(5)
                            
                            if not video_url:
                                yield sse({"status": "error", "message": "Video generation timed out"})
                                continue
                            
                            yield sse({"status": "progress", "message": "Downloading video..."})
                            
                            # Download the video
                            video_response = await client.get(video_url)
                            video_response.raise_for_status()
                            video_bytes = video_response.content
                            
                            # Upload to Supabase
                            filename = f"{uuid.uuid4().hex[:8]}.mp4"
                            storage_path = f"anonymous/{filename}"
                            
                            supabase.storage.from_("ai-videos").upload(
                                path=storage_path,
                                file=video_bytes,
                                file_options={"content-type": "video/mp4"}
                            )
                            
                            # Save video record
                            try:
                                supabase.table("videos").insert({
                                    "id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "video_path": storage_path,
                                    "prompt": prompt,
                                    "provider": "runwayml-gen2",
                                    "created_at": datetime.now().isoformat()
                                }).execute()
                            except Exception as e:
                                logger.error(f"Failed to save video record: {e}")
                            
                            # Get public URL
                            public_url = get_public_url("ai-videos", storage_path)
                            video_urls.append(public_url)
                    except Exception as e:
                        logger.error(f"RunwayML video generation failed: {e}")
                        yield sse({"status": "error", "message": f"Video generation failed: {str(e)}"})
                        continue
            
            if video_urls:
                yield sse({
                    "status": "completed",
                    "message": "Video generation completed",
                    "videos": [{"url": url, "type": "video/mp4"} for url in video_urls]
                })
                
                # Cache result
                cache_result(prompt, service, {
                    "provider": service,
                    "videos": [{"url": url, "type": "video/mp4"} for url in video_urls]
                })
            else:
                yield sse({"status": "error", "message": "No videos were generated successfully"})
        
        except asyncio.CancelledError:
            logger.info(f"Video stream cancelled for user {user_id}")
            yield sse({"status": "cancelled", "message": "Video generation cancelled"})
            raise
        
        except Exception as e:
            logger.exception("Video stream exception")
            yield sse({"status": "error", "message": str(e)})
        
        finally:
            # Cleanup
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

# Add missing function for prompt sanitization
def sanitize_prompt(prompt: str) -> str:
    """Sanitize prompt to avoid content policy violations"""
    # Remove any potentially problematic content
    # This is a simple implementation - you might want to enhance it
    return prompt

# Add missing function for running AI task
def run_ai_task(task_id: str):
    try:
        task = supabase.table("background_tasks") \
            .select("*") \
            .eq("id", task_id) \
            .single() \
            .execute()

        if not task.data:
            return

        data = task.data
        user_id = data["user_id"]
        params = json.loads(data["params"])

        conversation_id = params["conversation_id"]
        messages = params["messages"]

        models_to_try = ["llama-3.3-70b-versatile"]

        assistant_reply = None

        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1500
                }

                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }

                r = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=60)

                if r.status_code == 200:
                    assistant_reply = r.json()["choices"][0]["message"]["content"]
                    break

            except Exception:
                continue

        if assistant_reply:
            supabase.table("messages").insert({
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": "assistant",
                "content": assistant_reply,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

            supabase.table("conversations").update({
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()

            supabase.table("background_tasks").update({
                "status": "completed"
            }).eq("id", task_id).execute()

        else:
            supabase.table("background_tasks").update({
                "status": "failed"
            }).eq("id", task_id).execute()

    except Exception as e:
        logger.error(f"AI task failed: {e}")
