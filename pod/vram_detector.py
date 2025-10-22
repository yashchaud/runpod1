"""
VRAM Detection and Adaptive Configuration
Auto-adjusts batch sizes and concurrency based on available GPU memory
"""

import torch
import pynvml
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AdaptiveConfig:
    """Configuration adjusted based on available VRAM"""
    batch_size: int
    max_concurrent_batches: int
    num_gpus: int
    vram_per_gpu: List[int]
    total_vram_gb: float
    precision: str
    frame_sample_rate: int

    def __str__(self):
        return (
            f"Adaptive Configuration:\n"
            f"  GPUs: {self.num_gpus}\n"
            f"  Total VRAM: {self.total_vram_gb:.1f} GB\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Concurrent Batches: {self.max_concurrent_batches}\n"
            f"  Precision: {self.precision}\n"
            f"  Frame Sample Rate: 1 every {self.frame_sample_rate} frames"
        )

class VRAMDetector:
    """Detects GPU configuration and calculates optimal settings"""

    # Model memory footprint for Qwen3-VL-8B at different precisions
    MODEL_MEMORY = {
        'bfloat16': 16.0,  # GB (8B params × 2 bytes)
        'float16': 16.0,
        'int8': 8.0,       # 8B params × 1 byte
        'int4': 4.0        # 8B params × 0.5 bytes
    }

    # Approximate memory per image in batch (depends on resolution)
    MEMORY_PER_IMAGE = {
        'bfloat16': 0.5,   # GB per 1280x720 image
        'float16': 0.5,
        'int8': 0.3,
        'int4': 0.2
    }

    def __init__(self):
        pynvml.nvmlInit()
        self.num_gpus = torch.cuda.device_count()

    def get_gpu_memory(self) -> List[Tuple[int, int]]:
        """Get total and free memory for each GPU"""
        gpu_info = []
        for i in range(self.num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.append((info.total, info.free))
        return gpu_info

    def select_precision(self, vram_gb: float) -> str:
        """Select best precision based on available VRAM"""
        if vram_gb >= 24:
            return 'bfloat16'  # Best quality
        elif vram_gb >= 16:
            return 'float16'
        elif vram_gb >= 12:
            return 'int8'
        else:
            return 'int4'  # Smallest footprint

    def calculate_batch_size(self, free_vram_gb: float, precision: str) -> int:
        """Calculate optimal batch size based on free VRAM"""
        model_mem = self.MODEL_MEMORY[precision]
        mem_per_img = self.MEMORY_PER_IMAGE[precision]

        # Reserve 20% for safety margin
        usable_vram = free_vram_gb * 0.8 - model_mem

        if usable_vram <= 0:
            return 1

        batch_size = int(usable_vram / mem_per_img)

        # Clamp to reasonable range
        return max(1, min(batch_size, 32))

    def calculate_concurrency(self, batch_size: int, total_vram_gb: float) -> int:
        """Calculate number of concurrent batches to process"""
        # More VRAM = more concurrent batches
        if total_vram_gb >= 80:
            return 8  # Multiple A100s or H100s
        elif total_vram_gb >= 48:
            return 6  # A100 40GB or RTX 6000 Ada
        elif total_vram_gb >= 24:
            return 4  # RTX 4090 or A10G
        elif total_vram_gb >= 16:
            return 3  # RTX 4080
        else:
            return 2  # Smaller GPUs

    def calculate_frame_sample_rate(self, total_vram_gb: float, video_length_seconds: int) -> int:
        """Calculate frame sampling rate based on video length and VRAM"""
        # For long videos (>10 min), sample frames more aggressively
        if video_length_seconds > 3600:  # 1+ hour
            # Sample 1 frame every N frames based on VRAM
            if total_vram_gb >= 48:
                return 30  # 1 fps for 30fps video
            else:
                return 60  # 0.5 fps for 30fps video
        elif video_length_seconds > 600:  # 10+ min
            if total_vram_gb >= 48:
                return 15  # 2 fps
            else:
                return 30  # 1 fps
        else:
            # Short videos: process more frames
            return 10  # 3 fps

    def get_adaptive_config(self, video_length_seconds: int = 3600) -> AdaptiveConfig:
        """Get adaptive configuration based on available hardware"""
        gpu_memory = self.get_gpu_memory()

        if not gpu_memory:
            raise RuntimeError("No GPU detected!")

        # Calculate total VRAM
        total_vram = sum(total for total, _ in gpu_memory)
        free_vram = sum(free for _, free in gpu_memory)

        total_vram_gb = total_vram / (1024**3)
        free_vram_gb = free_vram / (1024**3)

        # Select precision
        precision = self.select_precision(total_vram_gb / self.num_gpus)

        # Calculate batch size based on free VRAM
        batch_size = self.calculate_batch_size(free_vram_gb / self.num_gpus, precision)

        # Calculate concurrency
        max_concurrent = self.calculate_concurrency(batch_size, total_vram_gb)

        # Calculate frame sampling
        frame_sample_rate = self.calculate_frame_sample_rate(total_vram_gb, video_length_seconds)

        return AdaptiveConfig(
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent,
            num_gpus=self.num_gpus,
            vram_per_gpu=[total // (1024**3) for total, _ in gpu_memory],
            total_vram_gb=total_vram_gb,
            precision=precision,
            frame_sample_rate=frame_sample_rate
        )

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# Quick test
if __name__ == "__main__":
    detector = VRAMDetector()

    # Test for different video lengths
    for duration in [300, 1800, 3600, 5400]:
        print(f"\n{'='*60}")
        print(f"Video Length: {duration//60} minutes")
        print(f"{'='*60}")
        config = detector.get_adaptive_config(duration)
        print(config)

        # Calculate processing time estimate
        fps = 30
        total_frames = duration * fps
        sampled_frames = total_frames // config.frame_sample_rate
        batches = sampled_frames // config.batch_size

        # Assume 1.2s per batch (optimistic with batching)
        processing_time = batches * 1.2 / config.max_concurrent_batches

        print(f"\nEstimated Processing:")
        print(f"  Total Frames: {total_frames:,}")
        print(f"  Sampled Frames: {sampled_frames:,}")
        print(f"  Batches: {batches:,}")
        print(f"  Est. Time: {processing_time/60:.1f} minutes ({processing_time/duration*100:.1f}% of video length)")
