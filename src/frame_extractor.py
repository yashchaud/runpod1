"""
FFmpeg-based frame extraction with adaptive FPS for optimal frame budget utilization.
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from videos using FFmpeg with intelligent sampling."""

    # Qwen3-VL has a hard limit of 768 frames per video
    MAX_FRAMES = 768

    def __init__(self, max_frames: int = MAX_FRAMES):
        """
        Initialize frame extractor.

        Args:
            max_frames: Maximum number of frames to extract (default: 768 for Qwen3-VL)
        """
        self.max_frames = max_frames
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found or not working")
            logger.info("FFmpeg verified successfully")
        except Exception as e:
            raise RuntimeError(f"FFmpeg verification failed: {e}")

    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with duration, fps, width, height, total_frames
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            data = json.loads(result.stdout)
            stream = data["streams"][0]
            format_data = data.get("format", {})

            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den != 0 else 30.0

            # Get duration (try stream first, then format)
            duration = None
            if "duration" in stream:
                duration = float(stream["duration"])
            elif "duration" in format_data:
                duration = float(format_data["duration"])

            # Calculate total frames
            total_frames = None
            if "nb_frames" in stream:
                total_frames = int(stream["nb_frames"])
            elif duration:
                total_frames = int(duration * fps)

            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames
            }

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise

    def calculate_sampling_rate(self, video_info: dict) -> float:
        """
        Calculate optimal FPS for frame extraction to stay within frame budget.

        Args:
            video_info: Video metadata from get_video_info()

        Returns:
            Target FPS for extraction
        """
        duration = video_info.get("duration")
        total_frames = video_info.get("total_frames")
        original_fps = video_info.get("fps", 30.0)

        if not duration or duration <= 0:
            logger.warning("Invalid duration, using 1 FPS")
            return 1.0

        # Calculate FPS needed to get max_frames from video
        target_fps = self.max_frames / duration

        # Don't exceed original FPS
        target_fps = min(target_fps, original_fps)

        # Ensure minimum FPS for very long videos
        target_fps = max(target_fps, 0.1)

        logger.info(
            f"Video duration: {duration:.2f}s, Original FPS: {original_fps:.2f}, "
            f"Total frames: {total_frames}, Target FPS: {target_fps:.2f}"
        )

        return target_fps

    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        target_fps: Optional[float] = None
    ) -> List[str]:
        """
        Extract frames from video at specified FPS.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames (creates temp dir if None)
            target_fps: Target FPS for extraction (auto-calculated if None)

        Returns:
            List of paths to extracted frame images
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Get video metadata
        video_info = self.get_video_info(video_path)

        # Calculate optimal FPS if not provided
        if target_fps is None:
            target_fps = self.calculate_sampling_rate(video_info)

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="frames_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        output_pattern = os.path.join(output_dir, "frame_%06d.jpg")

        # Extract frames using FFmpeg with fps filter
        # Using fps filter is more accurate than -r flag
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={target_fps}",
            "-q:v", "2",  # High quality JPEG
            "-frames:v", str(self.max_frames),  # Limit total frames
            output_pattern
        ]

        logger.info(f"Extracting frames with command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg extraction failed: {result.stderr}")

            # Collect extracted frames
            frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))
            frame_paths = [str(f) for f in frame_files]

            logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")

            if len(frame_paths) == 0:
                raise RuntimeError("No frames were extracted")

            return frame_paths

        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            raise
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise

    def extract_keyframes_only(self, video_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract only keyframes (I-frames) from video for faster processing.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames

        Returns:
            List of paths to extracted keyframes
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="keyframes_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        output_pattern = os.path.join(output_dir, "keyframe_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "select='eq(pict_type,I)'",
            "-vsync", "vfr",
            "-q:v", "2",
            "-frames:v", str(self.max_frames),
            output_pattern
        ]

        logger.info("Extracting keyframes only")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(f"Keyframe extraction failed: {result.stderr}")

            frame_files = sorted(Path(output_dir).glob("keyframe_*.jpg"))
            frame_paths = [str(f) for f in frame_files]

            logger.info(f"Extracted {len(frame_paths)} keyframes")
            return frame_paths

        except Exception as e:
            logger.error(f"Keyframe extraction failed: {e}")
            raise
