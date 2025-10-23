"""
Video chunking and processing logic for handling long-form videos.
Splits videos into processable chunks respecting frame budget limits.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

from frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


@dataclass
class VideoChunk:
    """Represents a video chunk with metadata."""
    chunk_path: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    frame_paths: Optional[List[str]] = None


class VideoProcessor:
    """Process long videos by chunking them into manageable segments."""

    def __init__(
        self,
        max_frames_per_chunk: int = 768,
        chunk_duration: float = 60.0,
        overlap_seconds: float = 2.0
    ):
        """
        Initialize video processor.

        Args:
            max_frames_per_chunk: Maximum frames per chunk (Qwen3-VL limit: 768)
            chunk_duration: Target duration for each chunk in seconds
            overlap_seconds: Overlap between chunks for context continuity
        """
        self.max_frames_per_chunk = max_frames_per_chunk
        self.chunk_duration = chunk_duration
        self.overlap_seconds = overlap_seconds
        self.frame_extractor = FrameExtractor(max_frames=max_frames_per_chunk)

    def calculate_optimal_chunks(self, video_info: dict) -> List[Tuple[float, float]]:
        """
        Calculate optimal chunk start/end times for a video.

        Args:
            video_info: Video metadata from FrameExtractor.get_video_info()

        Returns:
            List of (start_time, end_time) tuples for each chunk
        """
        duration = video_info.get("duration")
        fps = video_info.get("fps", 30.0)

        if not duration:
            raise ValueError("Video duration not available")

        # Calculate frames per second we can afford in each chunk
        max_fps_per_chunk = self.max_frames_per_chunk / self.chunk_duration

        # If original FPS is lower than our budget, use full chunk duration
        if fps <= max_fps_per_chunk:
            effective_chunk_duration = self.chunk_duration
        else:
            # Reduce chunk duration to stay within frame budget
            effective_chunk_duration = self.max_frames_per_chunk / fps
            logger.info(
                f"High FPS video ({fps:.2f}), reducing chunk duration to "
                f"{effective_chunk_duration:.2f}s to respect frame budget"
            )

        # Calculate number of chunks needed
        num_chunks = math.ceil(duration / effective_chunk_duration)

        chunks = []
        for i in range(num_chunks):
            start_time = max(0, i * effective_chunk_duration - self.overlap_seconds)
            end_time = min(duration, (i + 1) * effective_chunk_duration + self.overlap_seconds)

            # Ensure we don't create tiny chunks at the end
            if i == num_chunks - 1:
                end_time = duration

            chunks.append((start_time, end_time))

        logger.info(
            f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, "
            f"Creating {num_chunks} chunks of ~{effective_chunk_duration:.2f}s each"
        )

        return chunks

    def split_video_into_chunks(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        chunks: Optional[List[Tuple[float, float]]] = None
    ) -> List[VideoChunk]:
        """
        Split video into chunks using FFmpeg.

        Args:
            video_path: Path to input video
            output_dir: Directory for chunk output (temp dir if None)
            chunks: Pre-calculated chunk times, or None to auto-calculate

        Returns:
            List of VideoChunk objects
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video info
        video_info = self.frame_extractor.get_video_info(video_path)

        # Calculate chunks if not provided
        if chunks is None:
            chunks = self.calculate_optimal_chunks(video_info)

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="video_chunks_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        video_chunks = []

        for idx, (start_time, end_time) in enumerate(chunks):
            chunk_duration = end_time - start_time
            chunk_filename = f"chunk_{idx:04d}.mp4"
            chunk_path = os.path.join(output_dir, chunk_filename)

            # Use FFmpeg to extract chunk
            # Using -ss before -i for faster seeking
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(chunk_duration),
                "-c", "copy",  # Copy codec for fast extraction
                "-avoid_negative_ts", "1",
                chunk_path,
                "-y"  # Overwrite output file
            ]

            logger.info(
                f"Creating chunk {idx + 1}/{len(chunks)}: "
                f"{start_time:.2f}s - {end_time:.2f}s ({chunk_duration:.2f}s)"
            )

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                    raise RuntimeError(f"Failed to create chunk {idx}: {result.stderr}")

                if not os.path.exists(chunk_path):
                    raise RuntimeError(f"Chunk file not created: {chunk_path}")

                video_chunk = VideoChunk(
                    chunk_path=chunk_path,
                    chunk_index=idx,
                    start_time=start_time,
                    end_time=end_time,
                    duration=chunk_duration
                )

                video_chunks.append(video_chunk)

            except subprocess.TimeoutExpired:
                logger.error(f"Chunk {idx} creation timed out")
                raise
            except Exception as e:
                logger.error(f"Failed to create chunk {idx}: {e}")
                raise

        logger.info(f"Successfully created {len(video_chunks)} chunks in {output_dir}")
        return video_chunks

    def process_chunk_frames(self, chunk: VideoChunk, extract_frames: bool = True) -> VideoChunk:
        """
        Extract frames from a video chunk.

        Args:
            chunk: VideoChunk object
            extract_frames: Whether to extract frames immediately

        Returns:
            VideoChunk with frame_paths populated
        """
        if not extract_frames:
            return chunk

        try:
            # Create frames directory
            frames_dir = os.path.join(
                os.path.dirname(chunk.chunk_path),
                f"frames_chunk_{chunk.chunk_index:04d}"
            )

            # Extract frames
            frame_paths = self.frame_extractor.extract_frames(
                chunk.chunk_path,
                output_dir=frames_dir
            )

            chunk.frame_paths = frame_paths

            logger.info(
                f"Extracted {len(frame_paths)} frames from chunk {chunk.chunk_index}"
            )

            return chunk

        except Exception as e:
            logger.error(f"Failed to process frames for chunk {chunk.chunk_index}: {e}")
            raise

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        extract_frames: bool = True
    ) -> List[VideoChunk]:
        """
        Complete video processing pipeline: chunk and optionally extract frames.

        Args:
            video_path: Path to input video
            output_dir: Output directory for chunks and frames
            extract_frames: Whether to extract frames from chunks

        Returns:
            List of VideoChunk objects with frame data
        """
        logger.info(f"Processing video: {video_path}")

        # Split into chunks
        chunks = self.split_video_into_chunks(video_path, output_dir)

        # Process frames if requested
        if extract_frames:
            processed_chunks = []
            for chunk in chunks:
                processed_chunk = self.process_chunk_frames(chunk)
                processed_chunks.append(processed_chunk)
            return processed_chunks

        return chunks

    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate video file before processing.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(video_path):
            return False, f"File not found: {video_path}"

        if os.path.getsize(video_path) == 0:
            return False, "File is empty"

        try:
            video_info = self.frame_extractor.get_video_info(video_path)

            if not video_info.get("duration") or video_info["duration"] <= 0:
                return False, "Invalid or zero duration"

            if not video_info.get("width") or not video_info.get("height"):
                return False, "Invalid video dimensions"

            if video_info.get("total_frames", 0) == 0:
                return False, "Video has no frames"

            logger.info(f"Video validation passed: {video_info}")
            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"
