"""
Adaptive Video Processor with Qwen3-VL-8B
Auto-scales based on available VRAM and processes videos efficiently
"""

import torch
import cv2
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

from vram_detector import VRAMDetector, AdaptiveConfig
from frame_sampler import FrameSampler, FrameInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveVideoProcessor:
    """
    Adaptive video processor that scales based on available VRAM
    Efficiently processes long videos with parallel batch processing
    """

    # Prompts for different analysis modes
    PROMPTS = {
        "screen_share": """You are analyzing a screen recording or screen share from a video call.
Provide comprehensive analysis including:
1. Scene Description: What type of content is being shared (app demo, presentation, code, etc.)
2. Layout: Describe screen layout and visible element positions
3. Text Extraction: List ALL visible text (UI labels, buttons, menus, chat messages, code)
4. UI Elements: Identify interactive elements with approximate positions (top-left, center, etc.)
5. Activity: What is the user doing or demonstrating
6. Context: Purpose or goal of this screen share

Format as structured JSON.""",

        "ui_detection": """Analyze this screen and extract detailed information about UI elements.
For each UI element, provide:
- Type (button, input, menu, etc.)
- Text/label
- Approximate position (coordinates or description)
- State (enabled, disabled, selected)

Also extract all visible text and describe the overall layout.""",

        "meeting_analysis": """Analyze this video call/meeting screenshot.
Identify:
- Number of participants visible
- Screen sharing status
- Visible UI elements (chat, controls)
- Any text in chat or on screen
- Current activity or topic being discussed""",

        "app_demo": """This appears to be an application demonstration. Analyze:
1. What application or feature is being shown
2. Current screen/page name
3. All visible UI elements and their positions
4. Text content (labels, instructions, data)
5. User actions being demonstrated
6. Key features being highlighted"""
    }

    def __init__(self, mode: str = "screen_share"):
        """
        Initialize processor with adaptive configuration

        Args:
            mode: Analysis mode (screen_share, ui_detection, meeting_analysis, app_demo)
        """
        self.mode = mode
        self.prompt = self.PROMPTS.get(mode, self.PROMPTS["screen_share"])

        # Detect VRAM and get adaptive config
        logger.info("Detecting GPU configuration...")
        detector = VRAMDetector()
        self.config = detector.get_adaptive_config(video_length_seconds=3600)  # Default 1 hour
        logger.info(f"\n{self.config}")

        # Load model
        logger.info("Loading Qwen3-VL-8B-Instruct model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Select dtype based on config
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'int8': torch.int8,
            'int4': torch.int8  # Will use load_in_4bit
        }
        self.dtype = dtype_map.get(self.config.precision, torch.bfloat16)

        # Load with optimizations
        load_kwargs = {
            'device_map': 'auto',
            'trust_remote_code': True,
            'cache_dir': '/models'
        }

        if self.config.precision in ['bfloat16', 'float16']:
            load_kwargs['torch_dtype'] = self.dtype
        elif self.config.precision == 'int8':
            load_kwargs['load_in_8bit'] = True
        elif self.config.precision == 'int4':
            load_kwargs['load_in_4bit'] = True

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen3-VL-8B-Instruct',
            **load_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen3-VL-8B-Instruct',
            trust_remote_code=True,
            cache_dir='/models'
        )

        # Initialize frame sampler
        self.sampler = FrameSampler(
            sample_rate=self.config.frame_sample_rate,
            detect_scene_changes=True
        )

        # Thread safety
        self.model_lock = Lock()

        logger.info("Processor initialized and ready!")

    def update_config_for_video(self, video_path: str):
        """Update configuration based on specific video length"""
        video_info = self.sampler.get_video_info(video_path)
        duration = int(video_info['duration_seconds'])

        logger.info(f"Video duration: {duration/60:.1f} minutes")

        # Recalculate config for this video length
        detector = VRAMDetector()
        self.config = detector.get_adaptive_config(video_length_seconds=duration)

        # Update sampler
        self.sampler.sample_rate = self.config.frame_sample_rate

        logger.info(f"Updated configuration:\n{self.config}")

    def analyze_frame_batch(self, frames: List[FrameInfo]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of frames using Qwen3-VL-8B

        Args:
            frames: List of FrameInfo objects

        Returns:
            List of analysis results
        """
        if not frames:
            return []

        # Convert frames to PIL Images
        images = []
        for frame_info in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_info.image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            images.append(pil_image)

        results = []

        # Process each frame (batching at model level)
        for i, (frame_info, image) in enumerate(zip(frames, images)):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate with model lock (thread-safe)
            with self.model_lock:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=0.7
                    )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Parse JSON if possible
            try:
                analysis = json.loads(output_text)
            except json.JSONDecodeError:
                analysis = {"analysis": output_text}

            results.append({
                'frame_number': frame_info.frame_number,
                'timestamp': frame_info.timestamp,
                'is_keyframe': frame_info.is_keyframe,
                'scene_change_score': frame_info.scene_change_score,
                'analysis': analysis
            })

            logger.info(f"Analyzed frame {frame_info.frame_number} ({frame_info.timestamp:.1f}s)")

        return results

    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process entire video with parallel batch processing

        Args:
            video_path: Path to video file
            output_path: Path to save results JSON

        Returns:
            Complete analysis results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Update config for this video
        self.update_config_for_video(str(video_path))

        # Get video info
        video_info = self.sampler.get_video_info(str(video_path))

        logger.info(f"\nProcessing video: {video_path.name}")
        logger.info(f"  Duration: {video_info['duration_minutes']:.1f} minutes")
        logger.info(f"  Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"  Total Frames: {video_info['total_frames']:,}")

        # Estimate processing time
        estimated_sampled_frames = video_info['total_frames'] // self.config.frame_sample_rate
        estimated_batches = estimated_sampled_frames // self.config.batch_size
        # Assume 1.2s per batch / concurrent batches
        estimated_time_seconds = (estimated_batches * 1.2) / self.config.max_concurrent_batches

        logger.info(f"\nEstimated Processing:")
        logger.info(f"  Sampled Frames: ~{estimated_sampled_frames:,}")
        logger.info(f"  Batches: ~{estimated_batches:,}")
        logger.info(f"  Est. Time: {estimated_time_seconds/60:.1f} minutes")

        # Process in parallel batches
        all_results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_batches) as executor:
            futures = []

            # Submit batches
            for batch in self.sampler.batch_iterator(str(video_path), self.config.batch_size):
                future = executor.submit(self.analyze_frame_batch, batch)
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)

                    # Progress update
                    elapsed = time.time() - start_time
                    progress = len(all_results) / estimated_sampled_frames * 100
                    logger.info(f"Progress: {progress:.1f}% ({len(all_results)}/{estimated_sampled_frames} frames, {elapsed/60:.1f}m elapsed)")

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")

        # Sort results by frame number
        all_results.sort(key=lambda x: x['frame_number'])

        total_time = time.time() - start_time

        # Create final output
        output = {
            'video_info': video_info,
            'processing_config': {
                'batch_size': self.config.batch_size,
                'concurrent_batches': self.config.max_concurrent_batches,
                'frame_sample_rate': self.config.frame_sample_rate,
                'precision': self.config.precision,
                'mode': self.mode
            },
            'processing_stats': {
                'total_frames_analyzed': len(all_results),
                'processing_time_seconds': total_time,
                'processing_time_minutes': total_time / 60,
                'frames_per_second': len(all_results) / total_time,
                'efficiency_ratio': total_time / video_info['duration_seconds']
            },
            'results': all_results
        }

        logger.info(f"\nProcessing Complete!")
        logger.info(f"  Analyzed: {len(all_results):,} frames")
        logger.info(f"  Time: {total_time/60:.1f} minutes")
        logger.info(f"  Speed: {len(all_results)/total_time:.2f} frames/second")
        logger.info(f"  Efficiency: {total_time/video_info['duration_seconds']*100:.1f}% of video length")

        # Save results
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_analysis.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

        return output

# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python adaptive_processor.py <video_path> [mode] [output_path]")
        print("Modes: screen_share (default), ui_detection, meeting_analysis, app_demo")
        sys.exit(1)

    video_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "screen_share"
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    processor = AdaptiveVideoProcessor(mode=mode)
    processor.process_video(video_path, output_path)
