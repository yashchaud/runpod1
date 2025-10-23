"""
RunPod serverless handler for Qwen3-VL video processing.
Handles video upload, chunking, parallel processing, and result aggregation.
"""

import os
import sys
import logging
import tempfile
import shutil
from typing import Dict, Any, List, Optional
import traceback
import requests
from pathlib import Path

import runpod

# Import local modules
from video_processor import VideoProcessor, VideoChunk
from vllm_client import VLLMClient
from frame_extractor import FrameExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances (initialized once per worker)
vllm_client: Optional[VLLMClient] = None
video_processor: Optional[VideoProcessor] = None


def initialize_services():
    """Initialize global services (called once per worker)."""
    global vllm_client, video_processor

    if vllm_client is None:
        logger.info("Initializing vLLM client...")
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct-FP8")
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "131072"))
        gpu_memory_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
        tensor_parallel = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

        vllm_client = VLLMClient(
            model_name=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_util,
            tensor_parallel_size=tensor_parallel
        )
        vllm_client.initialize_model()
        logger.info("vLLM client initialized")

    if video_processor is None:
        logger.info("Initializing video processor...")
        max_frames = int(os.environ.get("MAX_FRAMES_PER_CHUNK", "768"))
        chunk_duration = float(os.environ.get("CHUNK_DURATION", "60.0"))
        overlap = float(os.environ.get("CHUNK_OVERLAP", "2.0"))

        video_processor = VideoProcessor(
            max_frames_per_chunk=max_frames,
            chunk_duration=chunk_duration,
            overlap_seconds=overlap
        )
        logger.info("Video processor initialized")


def download_video(url: str, output_path: str) -> str:
    """
    Download video from URL.

    Args:
        url: Video URL
        output_path: Local path to save video

    Returns:
        Path to downloaded video
    """
    logger.info(f"Downloading video from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(output_path)
        logger.info(f"Downloaded {file_size / (1024*1024):.2f} MB to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def validate_input(job_input: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate job input parameters.

    Args:
        job_input: Input dictionary from RunPod

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not job_input:
        return False, "No input provided"

    # Check for video source
    video_url = job_input.get("video_url")
    video_path = job_input.get("video_path")

    if not video_url and not video_path:
        return False, "Either 'video_url' or 'video_path' must be provided"

    # Validate prompt
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return False, "Prompt cannot be empty"

    return True, "Valid"


def process_video_job(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing function for video analysis job.

    Args:
        job_input: Dictionary containing:
            - video_url: URL to video file (optional)
            - video_path: Local path to video (optional)
            - prompt: Analysis prompt for the model
            - max_tokens: Maximum tokens per chunk response (default: 512)
            - chunk_duration: Override default chunk duration (optional)
            - aggregate: Whether to aggregate chunk results (default: True)
            - aggregation_prompt: Custom prompt for aggregation (optional)

    Returns:
        Dictionary with processing results
    """
    temp_dir = None
    video_file_path = None

    try:
        # Validate input
        is_valid, error_msg = validate_input(job_input)
        if not is_valid:
            return {
                "error": error_msg,
                "success": False
            }

        # Create temporary directory for this job
        temp_dir = tempfile.mkdtemp(prefix="runpod_job_")
        logger.info(f"Created temp directory: {temp_dir}")

        # Get or download video
        video_url = job_input.get("video_url")
        video_path = job_input.get("video_path")

        if video_url:
            video_file_path = os.path.join(temp_dir, "input_video.mp4")
            download_video(video_url, video_file_path)
        else:
            video_file_path = video_path
            if not os.path.exists(video_file_path):
                return {
                    "error": f"Video file not found: {video_file_path}",
                    "success": False
                }

        # Validate video
        is_valid, error_msg = video_processor.validate_video(video_file_path)
        if not is_valid:
            return {
                "error": f"Video validation failed: {error_msg}",
                "success": False
            }

        # Get video info
        video_info = video_processor.frame_extractor.get_video_info(video_file_path)
        logger.info(f"Video info: {video_info}")

        # Process video into chunks
        chunks_output_dir = os.path.join(temp_dir, "chunks")
        chunks = video_processor.process_video(
            video_file_path,
            output_dir=chunks_output_dir,
            extract_frames=False  # We'll use video chunks directly
        )

        logger.info(f"Created {len(chunks)} video chunks")

        # Prepare chunks for vLLM processing
        chunk_inputs = []
        for chunk in chunks:
            chunk_inputs.append({
                "video_path": chunk.chunk_path,
                "chunk_index": chunk.chunk_index,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time
            })

        # Get processing parameters
        prompt = job_input.get("prompt", "Describe what happens in this video segment.")
        max_tokens = job_input.get("max_tokens", 512)
        temperature = job_input.get("temperature", 0.7)
        top_p = job_input.get("top_p", 0.8)
        top_k = job_input.get("top_k", 20)

        # Process chunks
        logger.info(f"Starting vLLM processing for {len(chunk_inputs)} chunks")
        results = vllm_client.process_chunks_parallel(
            chunks=chunk_inputs,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        # Aggregate results if requested
        should_aggregate = job_input.get("aggregate", True)
        aggregated_response = None

        if should_aggregate:
            aggregation_prompt = job_input.get(
                "aggregation_prompt",
                "Provide a comprehensive summary of the entire video based on these segment descriptions:"
            )
            aggregated_response = vllm_client.aggregate_chunk_results(
                results,
                aggregation_prompt=aggregation_prompt
            )

        # Prepare response
        response = {
            "success": True,
            "video_info": video_info,
            "num_chunks": len(chunks),
            "chunk_results": [
                {
                    "chunk_index": r["chunk_index"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "success": r["success"],
                    "response": r["response"],
                    "processing_time": r["processing_time"],
                    "error": r.get("error")
                }
                for r in results
            ],
            "aggregated_response": aggregated_response,
            "total_successful_chunks": sum(1 for r in results if r["success"]),
            "total_processing_time": sum(r["processing_time"] for r in results)
        }

        logger.info(
            f"Job completed successfully: {response['total_successful_chunks']}/{len(chunks)} "
            f"chunks processed in {response['total_processing_time']:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Job processing failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "success": False,
            "traceback": traceback.format_exc()
        }

    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def handler(job):
    """
    RunPod handler function.

    Args:
        job: RunPod job object

    Returns:
        Processing results
    """
    try:
        job_input = job.get("input", {})
        logger.info(f"Received job with input: {job_input.keys()}")

        # Initialize services if not already done
        initialize_services()

        # Process the job
        result = process_video_job(job_input)

        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Handler error: {str(e)}",
            "success": False
        }


if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker for Qwen3-VL video processing")

    # Start the RunPod serverless worker
    runpod.serverless.start({
        "handler": handler
    })
