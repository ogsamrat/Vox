import uuid
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.pipeline.batch import BatchPipeline

app = FastAPI(title="Vox API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

jobs: Dict[str, Dict[str, Any]] = {}
pipeline: BatchPipeline = None


def get_pipeline() -> BatchPipeline:
    global pipeline
    if pipeline is None:
        pipeline = BatchPipeline()
    return pipeline


@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    allowed = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "uploading", "progress": 0, "stage": "Uploading...", "result": None, "error": None}
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "Starting..."
    thread = threading.Thread(target=process_audio_job, args=(job_id, tmp_path))
    thread.start()
    return JSONResponse({"job_id": job_id, "status": "processing"})


def process_audio_job(job_id: str, audio_path: str):
    def progress_callback(progress: int, stage: str):
        jobs[job_id]["progress"] = progress
        jobs[job_id]["stage"] = stage
    try:
        pipe = get_pipeline()
        result = pipe.process(audio_path, progress_callback=progress_callback)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Complete"
        jobs[job_id]["result"] = result
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return JSONResponse({"job_id": job_id, "status": job["status"], "progress": job["progress"], "stage": job["stage"], "error": job.get("error")})


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")
    return JSONResponse(job["result"])


@app.get("/health")
async def health():
    return {"status": "ok"}
