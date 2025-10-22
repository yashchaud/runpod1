"""
Intelligent Frame Sampling for Video Processing
Efficiently samples frames from long videos based on scene changes and key moments
"""

import cv2
import numpy as np
from typing import List, Tuple, Iterator
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrameInfo:
    """Information about a sampled frame"""
    frame_number: int
    timestamp: float
    image: np.ndarray
    is_keyframe: bool = False
    scene_change_score: float = 0.0

class FrameSampler:
    """Smart frame sampling with scene change detection"""

    def __init__(self, sample_rate: int = 30, detect_scene_changes: bool = True):
        """
        Args:
            sample_rate: Sample 1 frame every N frames
            detect_scene_changes: If True, always include frames with scene changes
        """
        self.sample_rate = sample_rate
        self.detect_scene_changes = detect_scene_changes
        self.scene_change_threshold = 30.0  # Mean pixel difference threshold

    def calculate_scene_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate scene change score between two frames"""
        if prev_frame is None or curr_frame is None:
            return 0.0

        # Resize for faster comparison
        prev_small = cv2.resize(prev_frame, (320, 180))
        curr_small = cv2.resize(curr_frame, (320, 180))

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        # Calculate mean absolute difference
        diff = np.abs(prev_gray.astype(float) - curr_gray.astype(float))
        return np.mean(diff)

    def sample_video(self, video_path: str) -> Iterator[FrameInfo]:
        """
        Sample frames from video with intelligent scene change detection

        Yields:
            FrameInfo objects containing sampled frames
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video: {Path(video_path).name}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Total Frames: {total_frames:,}")
        logger.info(f"  Duration: {duration/60:.1f} minutes")
        logger.info(f"  Sample Rate: 1 every {self.sample_rate} frames")

        prev_frame = None
        frame_count = 0
        sampled_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            should_sample = (frame_count % self.sample_rate == 0)

            # Check for scene changes
            is_scene_change = False
            scene_score = 0.0

            if self.detect_scene_changes and prev_frame is not None:
                scene_score = self.calculate_scene_change(prev_frame, frame)
                if scene_score > self.scene_change_threshold:
                    is_scene_change = True
                    should_sample = True

            # Yield sampled frames
            if should_sample:
                yield FrameInfo(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    image=frame.copy(),
                    is_keyframe=is_scene_change,
                    scene_change_score=scene_score
                )
                sampled_count += 1

            prev_frame = frame.copy()
            frame_count += 1

        cap.release()

        logger.info(f"Sampled {sampled_count:,} frames from {total_frames:,} ({sampled_count/total_frames*100:.1f}%)")

    def batch_iterator(self, video_path: str, batch_size: int) -> Iterator[List[FrameInfo]]:
        """
        Sample frames and yield them in batches

        Args:
            video_path: Path to video file
            batch_size: Number of frames per batch

        Yields:
            Lists of FrameInfo objects
        """
        batch = []

        for frame_info in self.sample_video(video_path):
            batch.append(frame_info)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining frames
        if batch:
            yield batch

    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        info['duration_seconds'] = info['total_frames'] / info['fps']
        info['duration_minutes'] = info['duration_seconds'] / 60

        cap.release()
        return info

# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_sampler.py <video_path> [sample_rate]")
        sys.exit(1)

    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    sampler = FrameSampler(sample_rate=sample_rate, detect_scene_changes=True)

    # Get video info
    info = sampler.get_video_info(video_path)
    print(f"\nVideo Information:")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration_minutes']:.1f} minutes")
    print(f"  Total Frames: {info['total_frames']:,}")

    # Sample frames
    print(f"\nSampling frames...")
    scene_changes = 0

    for i, frame_info in enumerate(sampler.sample_video(video_path)):
        if frame_info.is_keyframe:
            scene_changes += 1
            print(f"  Scene change at {frame_info.timestamp:.1f}s (score: {frame_info.scene_change_score:.1f})")

        if i >= 10:  # Just show first 10 for testing
            print(f"  ... (continuing)")
            break

    print(f"\nDetected {scene_changes} scene changes")
