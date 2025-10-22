"""
FastAPI Server for Video Processing Pod
Manages video processing jobs with queue system
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import asyncio
import aiofiles
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import uuid
from enum import Enum
from queue import Queue

from adaptive_processor import AdaptiveVideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
UPLOAD_DIR = Path("/workspace/uploads")
RESULTS_DIR = Path("/workspace/results")
JOBS_DIR = Path("/workspace/jobs")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="Video Processing Pod",
    description="Adaptive video processing with Qwen3-VL-8B",
    version="1.0.0"
)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(BaseModel):
    job_id: str
    status: JobStatus
    video_filename: str
    mode: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    result_path: Optional[str] = None

# Job storage
jobs: Dict[str, Job] = {}

# Progress streams for SSE
progress_queues: Dict[str, Queue] = {}

# Global processor (initialized on startup)
processor: Optional[AdaptiveVideoProcessor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize processor on startup"""
    global processor
    logger.info("Initializing video processor...")
    try:
        processor = AdaptiveVideoProcessor(mode="screen_share")
        logger.info("Processor initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Video Processing Pod",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process": "/process",
            "jobs": "/jobs",
            "job_status": "/jobs/{job_id}",
            "result": "/results/{job_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_info = {}

    if gpu_available:
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                }
                for i in range(torch.cuda.device_count())
            ]
        }

    return {
        "status": "healthy",
        "processor_initialized": processor is not None,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "active_jobs": len([j for j in jobs.values() if j.status == JobStatus.PROCESSING]),
        "total_jobs": len(jobs)
    }

@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    mode: str = Form("screen_share")
):
    """
    Submit a video for processing

    Args:
        video: Video file (mp4, avi, mov, etc.)
        mode: Analysis mode (screen_share, ui_detection, meeting_analysis, app_demo)

    Returns:
        Job information with job_id for tracking
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    # Validate mode
    valid_modes = ["screen_share", "ui_detection", "meeting_analysis", "app_demo"]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"

    try:
        async with aiofiles.open(video_path, 'wb') as f:
            content = await video.read()
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Create job
    job = Job(
        job_id=job_id,
        status=JobStatus.PENDING,
        video_filename=video.filename,
        mode=mode,
        created_at=datetime.now().isoformat()
    )
    jobs[job_id] = job

    # Save job metadata
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(job.dict(), f, indent=2)

    # Queue processing task
    background_tasks.add_task(process_job, job_id, str(video_path), mode)

    logger.info(f"Job {job_id} created for {video.filename}")

    return {
        "job_id": job_id,
        "status": job.status,
        "message": "Job queued for processing"
    }

async def process_job(job_id: str, video_path: str, mode: str):
    """Background task to process video"""
    job = jobs[job_id]

    # Create progress queue for this job
    progress_queue = Queue()
    progress_queues[job_id] = progress_queue

    def progress_callback(update: Dict[str, Any]):
        """Callback to send progress updates"""
        progress_queue.put(update)
        job.progress = update.get('progress', 0.0)
        save_job(job)

    try:
        # Update status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now().isoformat()
        save_job(job)

        progress_callback({
            'status': 'processing',
            'progress': 0.0,
            'message': 'Starting video processing...'
        })

        logger.info(f"Starting job {job_id}")

        # Process video
        result_path = RESULTS_DIR / f"{job_id}_result.json"

        # Update processor mode
        processor.mode = mode
        processor.prompt = processor.PROMPTS.get(mode, processor.PROMPTS["screen_share"])

        # Process with progress callback
        result = processor.process_video(video_path, str(result_path), progress_callback=progress_callback)

        # Update job
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        job.result_path = str(result_path)
        job.progress = 100.0

        progress_callback({
            'status': 'completed',
            'progress': 100.0,
            'message': 'Processing completed successfully!'
        })

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now().isoformat()
        job.error = str(e)

        progress_callback({
            'status': 'failed',
            'progress': job.progress,
            'message': f'Processing failed: {str(e)}'
        })

    finally:
        save_job(job)
        # Signal end of stream
        progress_queue.put(None)

def save_job(job: Job):
    """Save job metadata to disk"""
    job_file = JOBS_DIR / f"{job.job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(job.dict(), f, indent=2)

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(jobs),
        "jobs": list(jobs.values())
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]

@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Stream job progress using Server-Sent Events (SSE)

    This endpoint provides real-time progress updates for a job.
    The client can connect to this endpoint immediately after submitting
    a job to receive live updates about processing status.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        """Generate SSE events from progress queue"""
        # Wait for progress queue to be created
        timeout = 10
        start_time = time.time()
        while job_id not in progress_queues:
            if time.time() - start_time > timeout:
                yield f"data: {json.dumps({'error': 'Processing not started'})}\n\n"
                return
            await asyncio.sleep(0.1)

        queue = progress_queues[job_id]

        # Send initial status
        job = jobs[job_id]
        yield f"data: {json.dumps({'status': job.status, 'progress': job.progress})}\n\n"

        # Stream progress updates
        while True:
            try:
                # Non-blocking queue check
                if not queue.empty():
                    update = queue.get_nowait()

                    # None signals end of stream
                    if update is None:
                        break

                    # Send update
                    yield f"data: {json.dumps(update)}\n\n"
                else:
                    # Send heartbeat to keep connection alive
                    await asyncio.sleep(1)
                    yield f": heartbeat\n\n"

            except Exception as e:
                logger.error(f"Error in event generator: {e}")
                break

        # Send final status
        final_job = jobs[job_id]
        yield f"data: {json.dumps({'status': final_job.status, 'progress': final_job.progress, 'completed': True})}\n\n"

        # Cleanup
        if job_id in progress_queues:
            del progress_queues[job_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable proxy buffering
        }
    )

@app.get("/results/{job_id}")
async def get_result(job_id: str):
    """Get processing result"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status})")

    if not job.result_path or not Path(job.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        job.result_path,
        media_type="application/json",
        filename=f"result_{job_id}.json"
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Delete files
    video_path = UPLOAD_DIR / f"{job_id}_{job.video_filename}"
    if video_path.exists():
        video_path.unlink()

    if job.result_path and Path(job.result_path).exists():
        Path(job.result_path).unlink()

    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        job_file.unlink()

    # Remove from memory
    del jobs[job_id]

    return {"message": f"Job {job_id} deleted"}

@app.get("/config")
async def get_config():
    """Get current processor configuration"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    return {
        "mode": processor.mode,
        "config": {
            "batch_size": processor.config.batch_size,
            "max_concurrent_batches": processor.config.max_concurrent_batches,
            "num_gpus": processor.config.num_gpus,
            "total_vram_gb": processor.config.total_vram_gb,
            "precision": processor.config.precision,
            "frame_sample_rate": processor.config.frame_sample_rate
        }
    }

if __name__ == "__main__":
    logger.info("Starting Video Processing Pod Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
