"""
Process video file using RunPod serverless endpoints - PARALLEL VERSION
Optimized for multiple workers with async batch processing
"""
import os
import sys
import time
import json
import base64
import requests
import cv2
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import threading

# Fix Windows console encoding for emojis and disable buffering
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Check for .env file and load it
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

# RunPod Configuration
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
YOLO_ENDPOINT_ID = os.getenv('RUNPOD_YOLO_ENDPOINT_ID')
WHISPER_ENDPOINT_ID = os.getenv('RUNPOD_WHISPER_ENDPOINT_ID')

# Validate configuration
if not RUNPOD_API_KEY:
    print("ERROR: RUNPOD_API_KEY not set in environment variables")
    sys.exit(1)

if not YOLO_ENDPOINT_ID:
    print("ERROR: RUNPOD_YOLO_ENDPOINT_ID not set")
    sys.exit(1)

if not WHISPER_ENDPOINT_ID:
    print("ERROR: RUNPOD_WHISPER_ENDPOINT_ID not set")
    sys.exit(1)


class RunPodClient:
    """Client for calling RunPod serverless endpoints with parallel support"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def run_sync(self, endpoint_id: str, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """
        Run inference synchronously on a RunPod endpoint

        Args:
            endpoint_id: RunPod endpoint ID
            input_data: Input data for the endpoint
            timeout: Maximum time to wait for response (seconds)

        Returns:
            Response data from endpoint
        """
        url = f"{self.base_url}/{endpoint_id}/runsync"

        payload = {
            "input": input_data
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()

            result = response.json()

            # Check for errors in response
            if result.get('status') == 'error':
                raise Exception(f"RunPod error: {result.get('error', 'Unknown error')}")

            # Extract output
            if 'output' in result:
                return result['output']
            else:
                return result

        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")


class ParallelVideoProcessor:
    """Process video files using RunPod endpoints with parallel workers"""

    def __init__(self, yolo_endpoint_id: str, whisper_endpoint_id: str, api_key: str,
                 max_workers: int = 5, batch_size: int = 16):
        self.client = RunPodClient(api_key)
        self.yolo_endpoint = yolo_endpoint_id
        self.whisper_endpoint = whisper_endpoint_id
        self.max_workers = max_workers  # Number of parallel requests
        self.batch_size = batch_size    # Frames per batch (for 24GB VRAM)

    def extract_frames(self, video_path: str, fps: int = 1) -> List[Dict[str, Any]]:
        """Extract frames from video"""
        print(f"\n📹 Extracting frames from video (sampling at {fps} fps)...", flush=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        print(f"   Video info: {video_fps:.2f} fps, {total_frames} frames, {duration:.2f}s duration", flush=True)

        # Calculate frame interval
        frame_interval = int(video_fps / fps)

        frames = []
        frame_count = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract every Nth frame
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Higher quality
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                frames.append({
                    'timestamp': timestamp,
                    'frame_number': frame_count,
                    'image_b64': frame_b64
                })
                extracted += 1

            frame_count += 1

        cap.release()
        print(f"   ✓ Extracted {extracted} frames", flush=True)

        return frames

    def process_batch(self, batch_frames: List[Dict[str, Any]], batch_num: int, total_batches: int) -> Dict[str, Any]:
        """
        Process a batch of frames with YOLO endpoint

        Args:
            batch_frames: List of frames to process
            batch_num: Current batch number
            total_batches: Total number of batches

        Returns:
            Batch results
        """
        try:
            start_time = time.time()

            # Prepare batch request with optimized settings
            batch_input = {
                'images': [frame['image_b64'] for frame in batch_frames],
                'conf': 0.15,    # Lower confidence for more detections
                'iou': 0.5,      # Lower IOU for more detail
                'imgsz': 1280    # Higher resolution
            }

            # Call YOLO endpoint with batch
            response = self.client.run_sync(
                self.yolo_endpoint,
                batch_input,
                timeout=120
            )

            batch_time = time.time() - start_time

            # Extract results
            all_detections = response.get('detections', [])
            all_counts = response.get('count', [])
            throughput = response.get('throughput', 0)

            # Format results
            results = []
            for i, frame in enumerate(batch_frames):
                results.append({
                    'timestamp': frame['timestamp'],
                    'frame_number': frame['frame_number'],
                    'detections': all_detections[i] if i < len(all_detections) else [],
                    'count': all_counts[i] if i < len(all_counts) else 0
                })

            total_objs = sum(all_counts) if isinstance(all_counts, list) else 0
            print(f"   ✓ Batch {batch_num}/{total_batches}: {len(batch_frames)} frames, "
                  f"{total_objs} total objects, {batch_time:.2f}s ({throughput:.1f} fps)", flush=True)

            return {
                'success': True,
                'results': results,
                'batch_time': batch_time,
                'throughput': throughput
            }

        except Exception as e:
            print(f"   ✗ Batch {batch_num} failed: {str(e)}", flush=True)
            # Return error results for each frame
            return {
                'success': False,
                'results': [{
                    'timestamp': frame['timestamp'],
                    'frame_number': frame['frame_number'],
                    'error': str(e)
                } for frame in batch_frames],
                'error': str(e)
            }

    def process_frames_with_yolo_parallel(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process frames with YOLO endpoint using parallel workers

        Args:
            frames: List of frame data

        Returns:
            List of detection results
        """
        print(f"\n🎯 Processing {len(frames)} frames with YOLO endpoint...", flush=True)
        print(f"   Using {self.max_workers} parallel workers, batch size: {self.batch_size}", flush=True)

        # Split frames into batches
        batches = []
        for i in range(0, len(frames), self.batch_size):
            batches.append(frames[i:i + self.batch_size])

        total_batches = len(batches)
        print(f"   Split into {total_batches} batches", flush=True)

        # Process batches in parallel
        all_results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self.process_batch, batch, i+1, total_batches): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()
                    all_results.extend(batch_result['results'])
                except Exception as e:
                    print(f"   ✗ Batch {batch_idx + 1} exception: {str(e)}", flush=True)

        total_time = time.time() - start_time
        avg_fps = len(frames) / total_time if total_time > 0 else 0

        print(f"\n   ✅ YOLO processing complete!", flush=True)
        print(f"   Total time: {total_time:.2f}s, Average: {avg_fps:.1f} fps", flush=True)

        # Sort results by frame number to maintain order
        all_results.sort(key=lambda x: x['frame_number'])

        return all_results

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video and save as WAV"""
        print(f"\n🎵 Extracting audio from video...", flush=True)

        # Create temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            temp_audio.name
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✓ Audio extracted to {temp_audio.name}", flush=True)

            # Read and encode audio
            with open(temp_audio.name, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')

            return audio_b64

        except subprocess.CalledProcessError as e:
            raise Exception(f"ffmpeg failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("ffmpeg not found")

    def process_audio_with_whisper(self, audio_b64: str) -> Dict[str, Any]:
        """Process audio with Whisper endpoint"""
        print(f"\n🎤 Processing audio with Whisper endpoint...", flush=True)

        try:
            response = self.client.run_sync(
                self.whisper_endpoint,
                {
                    'audio': audio_b64,
                    'vad_filter': True,
                    'word_timestamps': False,
                    'beam_size': 5  # Higher beam size for better accuracy
                },
                timeout=300
            )

            print(f"   ✓ Transcription complete!", flush=True)
            print(f"   Language: {response.get('language', 'unknown')}", flush=True)
            print(f"   Duration: {response.get('duration', 0):.2f}s", flush=True)
            print(f"   Processing time: {response.get('processing_time', 0):.2f}s", flush=True)
            print(f"   Segments: {len(response.get('transcription', []))}", flush=True)

            return response

        except Exception as e:
            print(f"   ✗ Transcription failed: {str(e)}", flush=True)
            return {'error': str(e)}

    def process_video(self, video_path: str, output_dir: str = 'output', fps: int = 1):
        """Process complete video file"""
        print(f"\n{'='*60}", flush=True)
        print(f"🚀 Processing video: {video_path}", flush=True)
        print(f"{'='*60}", flush=True)

        start_time = time.time()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem

        # Extract frames
        frames = self.extract_frames(video_path, fps=fps)

        # Process with YOLO (parallel)
        detection_results = self.process_frames_with_yolo_parallel(frames)

        # Extract and process audio
        audio_b64 = self.extract_audio(video_path)
        transcription_results = self.process_audio_with_whisper(audio_b64)

        # Save results
        results = {
            'video': video_path,
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': time.time() - start_time,
            'config': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'fps': fps,
                'yolo_conf': 0.15,
                'yolo_iou': 0.5,
                'yolo_imgsz': 1280
            },
            'detections': detection_results,
            'transcription': transcription_results
        }

        output_file = os.path.join(output_dir, f'{video_name}_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}", flush=True)
        print(f"✅ Processing complete!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Total time: {time.time() - start_time:.2f}s", flush=True)
        print(f"Results saved to: {output_file}", flush=True)
        print(f"\nSummary:", flush=True)
        print(f"  - Frames processed: {len(detection_results)}", flush=True)
        print(f"  - Transcription segments: {len(transcription_results.get('transcription', []))}", flush=True)

        return results


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python process_video_parallel.py <video_file.mp4> [fps] [max_workers] [batch_size] [output_dir]")
        print("\nExample:")
        print("  python process_video_parallel.py video.mp4")
        print("  python process_video_parallel.py video.mp4 2 5 16")
        print("  python process_video_parallel.py video.mp4 1 3 12 results/")
        print("\nDefaults:")
        print("  fps: 1")
        print("  max_workers: 5 (concurrent requests)")
        print("  batch_size: 16 (frames per batch)")
        print("  output_dir: output/")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    output_dir = sys.argv[5] if len(sys.argv) > 5 else 'output'

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Create processor
    processor = ParallelVideoProcessor(
        yolo_endpoint_id=YOLO_ENDPOINT_ID,
        whisper_endpoint_id=WHISPER_ENDPOINT_ID,
        api_key=RUNPOD_API_KEY,
        max_workers=max_workers,
        batch_size=batch_size
    )

    # Process video
    try:
        processor.process_video(video_path, output_dir=output_dir, fps=fps)
    except Exception as e:
        print(f"\n❌ Error processing video: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
