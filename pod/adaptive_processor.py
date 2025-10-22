"""
Adaptive Video Processor with Qwen3-VL-8B
Auto-scales based on available VRAM and processes videos efficiently
"""

import torch
import cv2
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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

# Optional import for video chunking (backward compatible)
try:
    from video_chunker import VideoChunker, VideoChunk
    CHUNKING_AVAILABLE = True
except ImportError:
    logger.warning("video_chunker not available. Chunking features disabled.")
    CHUNKING_AVAILABLE = False
    VideoChunker = None
    VideoChunk = None

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

    def __init__(
        self,
        mode: str = "screen_share",
        chunk_duration: float = 60.0,
        use_chunking: bool = True
    ):
        """
        Initialize processor with adaptive configuration

        Args:
            mode: Analysis mode (screen_share, ui_detection, meeting_analysis, app_demo)
            chunk_duration: Duration of each video chunk in seconds (default: 60s)
            use_chunking: Whether to use video chunking for long videos
        """
        self.mode = mode
        self.prompt = self.PROMPTS.get(mode, self.PROMPTS["screen_share"])
        self.chunk_duration = chunk_duration
        self.use_chunking = use_chunking

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

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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

        # Initialize video chunker (if available)
        if CHUNKING_AVAILABLE and self.use_chunking:
            self.chunker = VideoChunker(
                chunk_duration=self.chunk_duration,
                overlap=2.0,
                use_scene_detection=True,
                min_chunk_duration=10.0,
                max_chunk_duration=120.0
            )
        else:
            self.chunker = None
            if self.use_chunking:
                logger.warning("Chunking requested but video_chunker not available. Will use direct analysis only.")
                self.use_chunking = False

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

            # Prepare inputs using new Qwen3-VL API
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

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

    def analyze_video_chunk(
        self,
        video_path: str,
        chunk: VideoChunk = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Analyze a video or video chunk using Qwen3-VL's native video support

        Args:
            video_path: Path to video file (or extracted chunk file)
            chunk: Optional VideoChunk for context
            progress_callback: Optional callback function for progress updates

        Returns:
            Video analysis result
        """
        chunk_info = f"chunk {chunk.chunk_id+1}/{chunk.total_chunks}" if chunk else "video"
        logger.info(f"Analyzing {chunk_info} with Qwen3-VL (file: {Path(video_path).name})...")

        if progress_callback and chunk:
            progress_callback({
                'progress': (chunk.chunk_id / chunk.total_chunks) * 100,
                'message': f'Processing chunk {chunk.chunk_id+1}/{chunk.total_chunks} ({chunk.start_time:.1f}s-{chunk.end_time:.1f}s)...',
                'chunk_id': chunk.chunk_id,
                'total_chunks': chunk.total_chunks
            })

        # Enhanced prompt with temporal context for chunks
        prompt = self.prompt
        if chunk:
            temporal_context = f"\n\nTemporal Context: This is segment {chunk.chunk_id+1} of {chunk.total_chunks}, covering {chunk.start_time:.1f}s to {chunk.end_time:.1f}s of the video ({chunk.duration:.1f}s duration)."
            prompt = prompt + temporal_context

        # Prepare video message using Qwen3-VL video support
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process video
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate with model lock (thread-safe)
        with self.model_lock:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # More tokens for chunk analysis
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

        # Add chunk metadata
        result = {
            'analysis': analysis
        }

        if chunk:
            result['chunk_info'] = {
                'chunk_id': chunk.chunk_id,
                'total_chunks': chunk.total_chunks,
                'start_time': chunk.start_time,
                'end_time': chunk.end_time,
                'duration': chunk.duration
            }

        logger.info(f"Analysis of {chunk_info} complete!")
        return result

    def aggregate_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple video chunks into coherent summary

        Args:
            chunk_results: List of analysis results from each chunk

        Returns:
            Aggregated analysis with timeline and overall summary
        """
        logger.info(f"Aggregating {len(chunk_results)} chunk results...")

        # Build timeline
        timeline = []
        all_text = []

        for result in chunk_results:
            chunk_info = result.get('chunk_info', {})
            analysis = result.get('analysis', {})

            timeline.append({
                'chunk_id': chunk_info.get('chunk_id', 0),
                'start_time': chunk_info.get('start_time', 0),
                'end_time': chunk_info.get('end_time', 0),
                'duration': chunk_info.get('duration', 0),
                'analysis': analysis
            })

            # Collect text for overall summary
            if isinstance(analysis, dict):
                all_text.append(json.dumps(analysis))
            else:
                all_text.append(str(analysis))

        # Sort by chunk_id
        timeline.sort(key=lambda x: x['chunk_id'])

        aggregated = {
            'timeline': timeline,
            'total_chunks': len(chunk_results),
            'aggregation_method': 'temporal_sequencing'
        }

        logger.info("Aggregation complete!")
        return aggregated

    def process_video(self, video_path: str, output_path: str = None, progress_callback=None) -> Dict[str, Any]:
        """
        Process entire video with intelligent chunking and parallel batch processing

        Args:
            video_path: Path to video file
            output_path: Path to save results JSON
            progress_callback: Optional callback function for progress updates

        Returns:
            Complete analysis results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = time.time()

        if progress_callback:
            progress_callback({
                'progress': 2.0,
                'message': 'Loading video information...'
            })

        # Get video info
        video_info = self.sampler.get_video_info(str(video_path))
        duration_seconds = video_info['duration_seconds']

        logger.info(f"\nProcessing video: {video_path.name}")
        logger.info(f"  Duration: {video_info['duration_minutes']:.1f} minutes ({duration_seconds:.1f}s)")
        logger.info(f"  Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"  Total Frames: {video_info['total_frames']:,}")

        # Decide on processing strategy
        use_chunking = (
            self.use_chunking and
            CHUNKING_AVAILABLE and
            self.chunker is not None and
            duration_seconds > self.chunk_duration * 1.5
        )

        if use_chunking:
            logger.info(f"\nUsing CHUNKED video analysis (chunks of ~{self.chunk_duration}s)")
            logger.info("  Benefits: Better depth, quality, and understanding per segment")

            if progress_callback:
                progress_callback({
                    'progress': 5.0,
                    'message': 'Creating video chunks...'
                })

            # Create chunks
            chunks = self.chunker.create_chunks(str(video_path))
            total_chunks = len(chunks)

            logger.info(f"  Created {total_chunks} chunks for optimal processing")

            if progress_callback:
                progress_callback({
                    'progress': 10.0,
                    'message': f'Processing {total_chunks} chunks in parallel...'
                })

            # Create temp directory for chunk files
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="video_chunks_")
            logger.info(f"Created temp directory for chunks: {temp_dir}")

            try:
                # Extract all chunks first (fast with ffmpeg copy mode)
                chunk_files = []
                for i, chunk in enumerate(chunks):
                    chunk_file = Path(temp_dir) / f"chunk_{chunk.chunk_id:03d}.mp4"
                    self.chunker.extract_chunk_video(chunk, str(chunk_file))
                    chunk_files.append((chunk, str(chunk_file)))
                    logger.info(f"Extracted chunk {i+1}/{total_chunks}: {chunk_file.name}")

                if progress_callback:
                    progress_callback({
                        'progress': 15.0,
                        'message': f'All {total_chunks} chunks extracted, starting parallel analysis...'
                    })

                # Process chunks in parallel batches
                chunk_results = []
                max_concurrent = self.config.max_concurrent_batches

                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    # Submit all chunks with their extracted files
                    future_to_chunk = {
                        executor.submit(self.analyze_video_chunk, chunk_file, chunk, progress_callback): chunk
                        for chunk, chunk_file in chunk_files
                    }

                    # Collect results as they complete
                    for i, future in enumerate(as_completed(future_to_chunk)):
                        chunk = future_to_chunk[future]
                        try:
                            result = future.result()
                            chunk_results.append(result)

                        # Progress update WITH chunk result data
                        progress = 10.0 + (i + 1) / total_chunks * 80.0  # 10% to 90%
                        elapsed = time.time() - start_time

                        if progress_callback:
                            progress_callback({
                                'progress': progress,
                                'message': f'Completed chunk {i+1}/{total_chunks} ({elapsed:.1f}s elapsed)',
                                'chunks_completed': i + 1,
                                'total_chunks': total_chunks,
                                # Stream the chunk result immediately!
                                'chunk_result': {
                                    'chunk_id': chunk.chunk_id,
                                    'start_time': chunk.start_time,
                                    'end_time': chunk.end_time,
                                    'duration': chunk.duration,
                                    'analysis': result.get('analysis', {})
                                }
                            })

                        logger.info(f"Chunk {chunk.chunk_id+1}/{total_chunks} completed ({elapsed:.1f}s elapsed)")

                    except Exception as e:
                        logger.error(f"Chunk {chunk.chunk_id} failed: {e}")

            finally:
                # Cleanup temporary chunk files
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir: {e}")

            # Aggregate results
            if progress_callback:
                progress_callback({
                    'progress': 90.0,
                    'message': 'Aggregating results from all chunks...'
                })

            aggregated_analysis = self.aggregate_chunk_results(chunk_results)

            total_time = time.time() - start_time

            # Create final output
            output = {
                'video_info': video_info,
                'processing_config': {
                    'method': 'chunked_video_analysis',
                    'chunk_duration': self.chunk_duration,
                    'total_chunks': total_chunks,
                    'max_concurrent_batches': max_concurrent,
                    'precision': self.config.precision,
                    'mode': self.mode
                },
                'processing_stats': {
                    'processing_time_seconds': total_time,
                    'processing_time_minutes': total_time / 60,
                    'efficiency_ratio': total_time / duration_seconds,
                    'avg_time_per_chunk': total_time / total_chunks if total_chunks > 0 else 0
                },
                'results': aggregated_analysis
            }

        else:
            # Short video - process directly without chunking
            logger.info(f"\nUsing DIRECT video analysis (video < {self.chunk_duration*1.5}s)")

            if progress_callback:
                progress_callback({
                    'progress': 10.0,
                    'message': 'Processing short video directly...'
                })

            try:
                analysis_result = self.analyze_video_chunk(str(video_path), None, progress_callback)

                total_time = time.time() - start_time

                # Create final output
                output = {
                    'video_info': video_info,
                    'processing_config': {
                        'method': 'direct_video_analysis',
                        'precision': self.config.precision,
                        'mode': self.mode
                    },
                    'processing_stats': {
                        'processing_time_seconds': total_time,
                        'processing_time_minutes': total_time / 60,
                        'efficiency_ratio': total_time / duration_seconds
                    },
                    'results': analysis_result
                }

            except Exception as e:
                logger.warning(f"Direct video analysis failed: {e}. Falling back to frame-by-frame analysis.")

                if progress_callback:
                    progress_callback({
                        'progress': 10.0,
                        'message': 'Falling back to frame-by-frame analysis...'
                    })

                # Fallback to frame-by-frame
                return self._process_video_frames(video_path, output_path, progress_callback, video_info, start_time)

        logger.info(f"\nProcessing Complete!")
        logger.info(f"  Time: {total_time/60:.1f} minutes")
        logger.info(f"  Efficiency: {total_time/duration_seconds*100:.1f}% of video length")

        # Save results
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_analysis.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")

        if progress_callback:
            progress_callback({
                'progress': 100.0,
                'message': 'Processing complete!'
            })

        return output

    def _process_video_frames(self, video_path: Path, output_path: str, progress_callback, video_info: Dict, start_time: float) -> Dict[str, Any]:
        """
        Fallback method: Process video by extracting and analyzing frames

        Args:
            video_path: Path to video file
            output_path: Path to save results JSON
            progress_callback: Optional callback function for progress updates
            video_info: Video metadata
            start_time: Processing start time

        Returns:
            Complete analysis results
        """
        # Update config for this video
        self.update_config_for_video(str(video_path))

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

                    if progress_callback:
                        progress_callback({
                            'progress': min(95.0, progress),  # Cap at 95% until fully done
                            'message': f'Analyzed {len(all_results)}/{estimated_sampled_frames} frames',
                            'frames_processed': len(all_results),
                            'total_frames': estimated_sampled_frames
                        })

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")

        # Sort results by frame number
        all_results.sort(key=lambda x: x['frame_number'])

        total_time = time.time() - start_time

        # Create final output
        output = {
            'video_info': video_info,
            'processing_config': {
                'method': 'frame_by_frame',
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
