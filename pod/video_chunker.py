"""
Video Chunking Utility
Intelligently segments videos into optimal chunks for VLM processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import subprocess
import json
import tempfile

logger = logging.getLogger(__name__)


class VideoChunk:
    """Represents a segment of a video"""

    def __init__(
        self,
        chunk_id: int,
        start_time: float,
        end_time: float,
        duration: float,
        video_path: str,
        total_chunks: int
    ):
        self.chunk_id = chunk_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.video_path = video_path
        self.total_chunks = total_chunks

    def __repr__(self):
        return f"VideoChunk(id={self.chunk_id}/{self.total_chunks}, {self.start_time:.1f}s-{self.end_time:.1f}s)"


class VideoChunker:
    """
    Intelligently chunk videos for optimal VLM processing

    Strategies:
    - Fixed duration chunks (e.g., 30s, 60s)
    - Scene-based chunking (split on scene changes)
    - Adaptive chunking (adjust based on content complexity)
    """

    def __init__(
        self,
        chunk_duration: float = 30.0,
        overlap: float = 2.0,
        use_scene_detection: bool = True,
        min_chunk_duration: float = 10.0,
        max_chunk_duration: float = 120.0
    ):
        """
        Args:
            chunk_duration: Target duration per chunk in seconds
            overlap: Overlap between chunks for context continuity (seconds)
            use_scene_detection: Whether to align chunks with scene boundaries
            min_chunk_duration: Minimum chunk duration
            max_chunk_duration: Maximum chunk duration
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.use_scene_detection = use_scene_detection
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except Exception as e:
            logger.warning(f"ffprobe failed, using cv2: {e}")
            # Fallback to cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else 0

    def detect_scene_changes(self, video_path: str, threshold: float = 30.0) -> List[float]:
        """
        Detect scene changes in video

        Args:
            video_path: Path to video
            threshold: Scene change detection threshold

        Returns:
            List of timestamps where scene changes occur
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        scene_changes = [0.0]  # Always start at 0
        prev_frame = None
        frame_idx = 0

        # Sample every 5 frames for performance
        sample_rate = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Convert to grayscale and compute histogram
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)

                    # Detect scene change
                    if mean_diff > threshold:
                        timestamp = frame_idx / fps
                        scene_changes.append(timestamp)
                        logger.debug(f"Scene change detected at {timestamp:.1f}s (diff={mean_diff:.1f})")

                prev_frame = gray

            frame_idx += 1

        cap.release()

        logger.info(f"Detected {len(scene_changes)-1} scene changes")
        return scene_changes

    def create_chunks(self, video_path: str, temp_dir: str = None) -> List[VideoChunk]:
        """
        Create video chunks using intelligent segmentation

        Args:
            video_path: Path to source video
            temp_dir: Directory for temporary chunk files

        Returns:
            List of VideoChunk objects
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video duration
        duration = self.get_video_duration(str(video_path))
        logger.info(f"Video duration: {duration:.1f}s ({duration/60:.1f}m)")

        # Detect scene changes if enabled
        scene_changes = []
        if self.use_scene_detection and duration > self.chunk_duration * 2:
            logger.info("Detecting scene changes...")
            scene_changes = self.detect_scene_changes(str(video_path))

        # Create chunk boundaries
        chunks = []
        current_time = 0.0
        chunk_id = 0

        while current_time < duration:
            # Calculate chunk end time
            target_end = current_time + self.chunk_duration

            # Align with scene change if available
            if scene_changes:
                # Find nearest scene change to target end
                nearby_scenes = [s for s in scene_changes if abs(s - target_end) < self.chunk_duration * 0.3]
                if nearby_scenes:
                    # Use closest scene change
                    chunk_end = min(nearby_scenes, key=lambda s: abs(s - target_end))
                else:
                    chunk_end = min(target_end, duration)
            else:
                chunk_end = min(target_end, duration)

            # Ensure minimum chunk duration
            if chunk_end - current_time < self.min_chunk_duration and chunk_end < duration:
                chunk_end = min(current_time + self.min_chunk_duration, duration)

            # Ensure maximum chunk duration
            if chunk_end - current_time > self.max_chunk_duration:
                chunk_end = current_time + self.max_chunk_duration

            chunks.append({
                'start': current_time,
                'end': chunk_end,
                'duration': chunk_end - current_time
            })

            # Move to next chunk with overlap
            current_time = chunk_end - self.overlap
            chunk_id += 1

        # Create VideoChunk objects
        total_chunks = len(chunks)
        video_chunks = []

        logger.info(f"Created {total_chunks} chunks:")
        for i, chunk_info in enumerate(chunks):
            chunk = VideoChunk(
                chunk_id=i,
                start_time=chunk_info['start'],
                end_time=chunk_info['end'],
                duration=chunk_info['duration'],
                video_path=str(video_path),
                total_chunks=total_chunks
            )
            video_chunks.append(chunk)
            logger.info(f"  Chunk {i}: {chunk_info['start']:.1f}s - {chunk_info['end']:.1f}s ({chunk_info['duration']:.1f}s)")

        return video_chunks

    def extract_chunk_video(self, chunk: VideoChunk, output_path: str):
        """
        Extract a chunk as a separate video file using ffmpeg

        Args:
            chunk: VideoChunk to extract
            output_path: Path to save extracted chunk
        """
        cmd = [
            'ffmpeg',
            '-i', chunk.video_path,
            '-ss', str(chunk.start_time),
            '-t', str(chunk.duration),
            '-c', 'copy',  # Copy codec for speed
            '-y',  # Overwrite
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.debug(f"Extracted chunk {chunk.chunk_id} to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract chunk: {e.stderr.decode()}")
            raise


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_chunker.py <video_path> [chunk_duration]")
        sys.exit(1)

    video_path = sys.argv[1]
    chunk_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    logging.basicConfig(level=logging.INFO)

    chunker = VideoChunker(
        chunk_duration=chunk_duration,
        overlap=2.0,
        use_scene_detection=True
    )

    chunks = chunker.create_chunks(video_path)

    print(f"\n{len(chunks)} chunks created:")
    for chunk in chunks:
        print(f"  {chunk}")
