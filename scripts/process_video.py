"""
Process video file using RunPod serverless endpoints
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

# Fix Windows console encoding for emojis and disable buffering
if sys.platform == 'win32':
    import codecs
    # Use line buffering mode for immediate output
    class LineBufferedWriter(codecs.StreamWriter):
        def write(self, data):
            result = super().write(data)
            self.stream.flush()
            return result

    # Wrap stdout/stderr with UTF-8 encoding and auto-flush
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
    print("Please create a .env file with: RUNPOD_API_KEY=your-api-key")
    sys.exit(1)

if not YOLO_ENDPOINT_ID:
    print("ERROR: RUNPOD_YOLO_ENDPOINT_ID not set in environment variables")
    print("Please add to .env: RUNPOD_YOLO_ENDPOINT_ID=your-endpoint-id")
    sys.exit(1)

if not WHISPER_ENDPOINT_ID:
    print("ERROR: RUNPOD_WHISPER_ENDPOINT_ID not set in environment variables")
    print("Please add to .env: RUNPOD_WHISPER_ENDPOINT_ID=your-endpoint-id")
    sys.exit(1)


class RunPodClient:
    """Client for calling RunPod serverless endpoints"""

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


class VideoProcessor:
    """Process video files using RunPod endpoints"""

    def __init__(self, yolo_endpoint_id: str, whisper_endpoint_id: str, api_key: str):
        self.client = RunPodClient(api_key)
        self.yolo_endpoint = yolo_endpoint_id
        self.whisper_endpoint = whisper_endpoint_id

    def extract_frames(self, video_path: str, fps: int = 1) -> List[Dict[str, Any]]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            fps: Frames per second to extract (default: 1 frame/sec)

        Returns:
            List of frame data with timestamps
        """
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
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video and save as WAV

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file (base64 encoded)
        """
        print(f"\n🎵 Extracting audio from video...")

        # Create temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate (Whisper standard)
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            temp_audio.name
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✓ Audio extracted to {temp_audio.name}")

            # Read and encode audio
            with open(temp_audio.name, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')

            return audio_b64

        except subprocess.CalledProcessError as e:
            raise Exception(f"ffmpeg failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("ffmpeg not found. Please install ffmpeg: https://ffmpeg.org/download.html")

    def process_frames_with_yolo(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process frames with YOLO endpoint

        Args:
            frames: List of frame data

        Returns:
            List of detection results
        """
        print(f"\n🎯 Processing {len(frames)} frames with YOLO endpoint...", flush=True)

        results = []

        for i, frame in enumerate(frames):
            try:
                # Call YOLO endpoint
                response = self.client.run_sync(
                    self.yolo_endpoint,
                    {
                        'image': frame['image_b64'],
                        'conf': 0.25
                    },
                    timeout=60
                )

                results.append({
                    'timestamp': frame['timestamp'],
                    'frame_number': frame['frame_number'],
                    'detections': response.get('detections', []),
                    'count': response.get('count', 0),
                    'inference_time': response.get('inference_time', 0)
                })

                print(f"   Frame {i+1}/{len(frames)}: {response.get('count', 0)} objects detected " +
                      f"({response.get('inference_time', 0):.3f}s)", flush=True)

            except Exception as e:
                print(f"   ✗ Frame {i+1} failed: {str(e)}")
                results.append({
                    'timestamp': frame['timestamp'],
                    'frame_number': frame['frame_number'],
                    'error': str(e)
                })

        return results

    def process_audio_with_whisper(self, audio_b64: str) -> Dict[str, Any]:
        """
        Process audio with Whisper endpoint

        Args:
            audio_b64: Base64 encoded audio

        Returns:
            Transcription results
        """
        print(f"\n🎤 Processing audio with Whisper endpoint...")

        try:
            response = self.client.run_sync(
                self.whisper_endpoint,
                {
                    'audio': audio_b64,
                    'vad_filter': True,
                    'word_timestamps': False
                },
                timeout=300
            )

            print(f"   ✓ Transcription complete!")
            print(f"   Language: {response.get('language', 'unknown')}")
            print(f"   Duration: {response.get('duration', 0):.2f}s")
            print(f"   Processing time: {response.get('processing_time', 0):.2f}s")
            print(f"   Segments: {len(response.get('transcription', []))}")

            return response

        except Exception as e:
            print(f"   ✗ Transcription failed: {str(e)}")
            return {'error': str(e)}

    def process_video(self, video_path: str, output_dir: str = 'output', fps: int = 1):
        """
        Process complete video file

        Args:
            video_path: Path to video file
            output_dir: Directory to save results
            fps: Frames per second to process
        """
        print(f"\n{'='*60}")
        print(f"🚀 Processing video: {video_path}")
        print(f"{'='*60}")

        start_time = time.time()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem

        # Extract frames
        frames = self.extract_frames(video_path, fps=fps)

        # Process with YOLO
        detection_results = self.process_frames_with_yolo(frames)

        # Extract and process audio
        audio_b64 = self.extract_audio(video_path)
        transcription_results = self.process_audio_with_whisper(audio_b64)

        # Save results
        results = {
            'video': video_path,
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': time.time() - start_time,
            'detections': detection_results,
            'transcription': transcription_results
        }

        output_file = os.path.join(output_dir, f'{video_name}_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"{'='*60}")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Results saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  - Frames processed: {len(detection_results)}")
        print(f"  - Transcription segments: {len(transcription_results.get('transcription', []))}")

        return results


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_file.mp4> [fps] [output_dir]")
        print("\nExample:")
        print("  python process_video.py video.mp4")
        print("  python process_video.py video.mp4 2")
        print("  python process_video.py video.mp4 1 results/")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output'

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Create processor
    processor = VideoProcessor(
        yolo_endpoint_id=YOLO_ENDPOINT_ID,
        whisper_endpoint_id=WHISPER_ENDPOINT_ID,
        api_key=RUNPOD_API_KEY
    )

    # Process video
    try:
        processor.process_video(video_path, output_dir=output_dir, fps=fps)
    except Exception as e:
        print(f"\n❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
