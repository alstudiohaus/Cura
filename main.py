"""
Cura Backend — FastAPI
Claude vision + Google Drive + FFmpeg pipeline
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
import uuid
import json

app = FastAPI(title="Cura API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# In-memory job store (use Redis/DB in production)
jobs = {}


# ── MODELS ──

class IndexRequest(BaseModel):
    drive_folder_id: str
    access_token: str

class GenerateRequest(BaseModel):
    prompt: str
    folder_id: Optional[str] = None
    duration_seconds: int = 30
    vibe: str = "warm"
    photo_count: int = 12

class ReelJob(BaseModel):
    job_id: str
    status: str  # pending | indexing | curating | assembling | done | error
    progress: int
    message: str
    result: Optional[dict] = None


# ── ENDPOINTS ──

@app.get("/")
def root():
    return {"service": "Cura API", "status": "running"}


@app.post("/api/index")
async def index_drive(req: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index a Google Drive folder:
    1. List all image files in the folder
    2. Download thumbnails
    3. Run Claude vision on each → extract scene, mood, aesthetic score
    4. Store embeddings in pgvector
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "indexing", "progress": 0, "message": "Starting…"}
    background_tasks.add_task(run_indexing, job_id, req.drive_folder_id, req.access_token)
    return {"job_id": job_id}


@app.post("/api/generate")
async def generate_reel(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate a reel from a natural language prompt:
    1. Parse brief with Claude
    2. Semantic search + vision rerank
    3. Sequence photos for narrative flow
    4. Assemble MP4 with FFmpeg
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued"}
    background_tasks.add_task(run_generation, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ── BACKGROUND TASKS ──

async def run_indexing(job_id: str, folder_id: str, access_token: str):
    """Index all photos in a Drive folder using Claude vision."""
    try:
        jobs[job_id] = {"status": "indexing", "progress": 10, "message": "Fetching file list from Drive…"}

        # TODO: Call Google Drive API to list images
        # files = drive_list_images(folder_id, access_token)

        jobs[job_id]["progress"] = 30
        jobs[job_id]["message"] = "Running vision analysis…"

        # For each image thumbnail, call Claude vision:
        # analysis = analyze_photo_with_claude(thumbnail_base64)
        # Store result in pgvector

        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "done"
        jobs[job_id]["message"] = "Indexing complete"

    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 0, "message": str(e)}


async def run_generation(job_id: str, req: GenerateRequest):
    """Full reel generation pipeline."""
    try:
        # Step 1: Parse the brief
        jobs[job_id] = {"status": "curating", "progress": 15, "message": "Parsing your brief…"}
        brief = await parse_brief(req.prompt, req.duration_seconds, req.vibe)

        # Step 2: Select photos
        jobs[job_id]["progress"] = 40
        jobs[job_id]["message"] = "Selecting best photos from your library…"
        selected = await select_photos(brief, req.photo_count)

        # Step 3: Sequence
        jobs[job_id]["progress"] = 65
        jobs[job_id]["message"] = "Sequencing for narrative flow…"
        sequence = await sequence_photos(selected, brief)

        # Step 4: Assemble (FFmpeg)
        jobs[job_id]["progress"] = 85
        jobs[job_id]["message"] = "Assembling reel…"
        output_path = assemble_reel(sequence, req.duration_seconds)

        jobs[job_id] = {
            "status": "done",
            "progress": 100,
            "message": "Reel ready!",
            "result": {
                "photo_count": len(sequence),
                "duration": req.duration_seconds,
                "output_path": output_path,
                "sequence": sequence,
                "brief": brief,
            }
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "progress": 0, "message": str(e)}


# ── CLAUDE HELPERS ──

async def parse_brief(prompt: str, duration: int, vibe: str) -> dict:
    """Use Claude to parse natural language into a structured brief."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="""You are a creative director AI. Parse the user's reel brief into structured JSON.
Return ONLY valid JSON with these fields:
{
  "themes": ["list of visual themes"],
  "mood": "emotional tone",
  "subjects": ["people", "food", "landscape", etc],
  "color_palette": "warm/cool/neutral/vibrant",
  "pace": "slow/medium/fast",
  "avoid": ["anything to exclude"]
}""",
        messages=[{"role": "user", "content": f"Brief: {prompt}\nDuration: {duration}s\nVibe: {vibe}"}]
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def analyze_photo_with_claude(image_base64: str, media_type: str = "image/jpeg") -> dict:
    """Score a single photo using Claude vision."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system="""Analyze this photo for content creation. Return ONLY JSON:
{
  "scene": "brief scene description",
  "subjects": ["list of main subjects"],
  "mood": "emotional tone",
  "quality_score": 0-100,
  "aesthetic_score": 0-100,
  "color_palette": "warm/cool/neutral/vibrant",
  "composition": "good/fair/poor",
  "tags": ["keyword", "tags"]
}""",
        messages=[{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            }, {
                "type": "text",
                "text": "Analyze this photo."
            }]
        }]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def select_photos(brief: dict, count: int) -> list:
    """
    Select best photos from the indexed library.
    In production: pgvector similarity search + Claude reranking.
    """
    # TODO: query pgvector with brief embeddings
    # TODO: rerank top candidates with Claude
    return []  # placeholder


async def sequence_photos(photos: list, brief: dict) -> list:
    """Use Claude to determine the best narrative ordering."""
    if not photos:
        return photos

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="You are a video editor. Given a list of photos with metadata, return the optimal sequence for a social media reel. Return ONLY a JSON array of photo IDs in order.",
        messages=[{
            "role": "user",
            "content": f"Photos: {json.dumps(photos)}\nBrief: {json.dumps(brief)}\nReturn the optimal sequence as a JSON array of IDs."
        }]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def assemble_reel(sequence: list, duration: int) -> str:
    """
    Assemble photos into a 9:16 MP4 using FFmpeg.
    Each photo gets duration/len(sequence) seconds.
    """
    # TODO: implement FFmpeg assembly
    # Example command:
    # ffmpeg -framerate 1/clip_duration -i input_%03d.jpg \
    #   -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" \
    #   -c:v libx264 -pix_fmt yuv420p output.mp4
    output_path = f"/tmp/cura_{uuid.uuid4().hex[:8]}.mp4"
    return output_path
