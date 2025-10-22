"""
Client for submitting videos to RunPod Video Processing Pod
"""

import requests
import time
import json
import sys
from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime

class PodClient:
    """Client for interacting with video processing pod"""

    def __init__(self, pod_url: str):
        """
        Args:
            pod_url: Base URL of the pod (e.g., http://your-pod-id.runpod.net)
        """
        self.base_url = pod_url.rstrip('/')
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Check pod health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_config(self) -> dict:
        """Get processor configuration"""
        response = self.session.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()

    def submit_video(
        self,
        video_path: str,
        mode: str = "screen_share",
        chunk_duration: Optional[float] = None
    ) -> dict:
        """
        Submit video for processing

        Args:
            video_path: Path to video file
            mode: Analysis mode (screen_share, ui_detection, meeting_analysis, app_demo)
            chunk_duration: Optional chunk duration in seconds (default: 60s)
                          Use 30-45s for detailed analysis, 90-120s for overview

        Returns:
            Job information
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"Uploading video: {video_path.name}")
        print(f"Mode: {mode}")
        if chunk_duration:
            print(f"Chunk Duration: {chunk_duration}s")

        with open(video_path, 'rb') as f:
            files = {'video': (video_path.name, f, 'video/mp4')}
            data = {'mode': mode}

            if chunk_duration is not None:
                data['chunk_duration'] = str(chunk_duration)

            response = self.session.post(
                f"{self.base_url}/process",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for upload + queueing
            )

        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> dict:
        """Get job status"""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_result(self, job_id: str, output_path: Optional[str] = None) -> dict:
        """
        Get processing result

        Args:
            job_id: Job ID
            output_path: Path to save result JSON

        Returns:
            Processing result
        """
        response = self.session.get(f"{self.base_url}/results/{job_id}")
        response.raise_for_status()

        result = response.json()

        # Save to file if specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Result saved to: {output_path}")

        return result

    def stream_progress(self, job_id: str) -> dict:
        """
        Stream real-time progress updates using SSE

        Args:
            job_id: Job ID

        Returns:
            Final job status
        """
        print(f"\nStreaming progress for job {job_id}...")
        print("(Press Ctrl+C to stop streaming - job will continue processing)\n")

        start_time = time.time()

        try:
            response = self.session.get(
                f"{self.base_url}/jobs/{job_id}/stream",
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=None  # No timeout for streaming
            )
            response.raise_for_status()

            # Process SSE stream
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')

                # Skip heartbeat
                if line.startswith(':'):
                    continue

                # Parse SSE data
                if line.startswith('data: '):
                    data_json = line[6:]  # Remove 'data: ' prefix
                    try:
                        data = json.loads(data_json)

                        # Display progress
                        if 'message' in data:
                            elapsed = time.time() - start_time
                            progress = data.get('progress', 0)
                            print(f"[{elapsed:.1f}s] {progress:.1f}% - {data['message']}")

                        # Display chunk result if available (INCREMENTAL STREAMING!)
                        if 'chunk_result' in data:
                            chunk_result = data['chunk_result']
                            print(f"\n{'='*60}")
                            print(f"CHUNK {chunk_result['chunk_id']+1} RESULT ({chunk_result['start_time']:.1f}s - {chunk_result['end_time']:.1f}s):")
                            print(f"{'='*60}")

                            # Display analysis (first 500 chars)
                            analysis = chunk_result.get('analysis', {})
                            if isinstance(analysis, dict):
                                for key, value in analysis.items():
                                    value_str = str(value)[:500]
                                    print(f"{key}: {value_str}")
                                    if len(str(value)) > 500:
                                        print("  ... (truncated)")
                            else:
                                analysis_str = str(analysis)[:500]
                                print(analysis_str)
                                if len(str(analysis)) > 500:
                                    print("... (truncated)")

                            print(f"{'='*60}\n")

                        # Check if completed
                        if data.get('completed'):
                            elapsed = time.time() - start_time
                            status = data.get('status', 'unknown')
                            print(f"\nJob {status} in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
                            return self.get_job_status(job_id)

                    except json.JSONDecodeError:
                        continue

        except KeyboardInterrupt:
            print("\n\nStopped streaming (job continues processing)")
            print(f"Check status later with: job_id={job_id}")
            return self.get_job_status(job_id)

        except Exception as e:
            print(f"\nStreaming error: {e}")
            print("Falling back to polling...")
            return self.wait_for_completion_polling(job_id)

    def wait_for_completion_polling(self, job_id: str, poll_interval: int = 10) -> dict:
        """
        Wait for job to complete using polling (fallback method)

        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks

        Returns:
            Final job status
        """
        print(f"\nPolling for job {job_id} status...")
        print("(Press Ctrl+C to stop polling - job will continue processing)\n")

        start_time = time.time()
        last_status = None

        try:
            while True:
                status = self.get_job_status(job_id)
                current_status = status['status']

                # Print status updates
                if current_status != last_status:
                    elapsed = time.time() - start_time
                    progress = status.get('progress', 0)
                    print(f"[{elapsed/60:.1f}m] Status: {current_status} ({progress:.1f}%)")

                    if current_status == "processing":
                        if status.get('started_at'):
                            print(f"  Started: {status['started_at']}")

                    last_status = current_status

                # Check if done
                if current_status in ["completed", "failed"]:
                    elapsed = time.time() - start_time
                    print(f"\nJob {current_status} in {elapsed/60:.1f} minutes")

                    if current_status == "failed":
                        print(f"Error: {status.get('error', 'Unknown error')}")

                    return status

                # Wait before next check
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nStopped polling (job continues processing)")
            print(f"Check status later with: job_id={job_id}")
            return self.get_job_status(job_id)

    def wait_for_completion(self, job_id: str, poll_interval: int = 10, use_streaming: bool = True) -> dict:
        """
        Wait for job to complete

        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks (for polling mode)
            use_streaming: Use SSE streaming for real-time updates (default: True)

        Returns:
            Final job status
        """
        if use_streaming:
            try:
                return self.stream_progress(job_id)
            except Exception as e:
                print(f"Streaming not available: {e}")
                print("Falling back to polling mode...")
                return self.wait_for_completion_polling(job_id, poll_interval)
        else:
            return self.wait_for_completion_polling(job_id, poll_interval)

    def process_and_wait(
        self,
        video_path: str,
        mode: str = "screen_share",
        output_path: Optional[str] = None,
        poll_interval: int = 10,
        use_streaming: bool = True,
        chunk_duration: Optional[float] = None
    ) -> dict:
        """
        Submit video and wait for completion

        Args:
            video_path: Path to video file
            mode: Analysis mode
            output_path: Path to save result JSON
            poll_interval: Seconds between status checks
            use_streaming: Use SSE streaming for real-time updates
            chunk_duration: Optional chunk duration in seconds

        Returns:
            Processing result
        """
        # Check pod health
        print("Checking pod health...")
        health = self.health_check()
        print(f"Pod status: {health['status']}")
        print(f"GPUs available: {health['gpu_available']}")

        if health['gpu_available']:
            for i, gpu in enumerate(health['gpu_info']['devices']):
                print(f"  GPU {i}: {gpu['name']} ({gpu['memory_allocated_gb']:.1f}GB / {gpu['memory_reserved_gb']:.1f}GB)")

        # Get config
        config = self.get_config()
        print(f"\nProcessor Configuration:")
        if 'chunking' in config:
            print(f"  Video Chunking: {'Enabled' if config['chunking']['enabled'] else 'Disabled'}")
            print(f"  Chunk Duration: {config['chunking']['chunk_duration']}s")
            print(f"  Scene Detection: {'Enabled' if config['chunking']['scene_detection'] else 'Disabled'}")
        print(f"  Batch Size: {config['config']['batch_size']}")
        print(f"  Concurrent Batches: {config['config']['max_concurrent_batches']}")
        print(f"  VRAM: {config['config']['total_vram_gb']:.1f} GB")
        print(f"  Precision: {config['config']['precision']}")

        # Submit video
        print(f"\n{'='*60}")
        job_info = self.submit_video(video_path, mode, chunk_duration)
        job_id = job_info['job_id']

        print(f"\nJob submitted!")
        print(f"Job ID: {job_id}")
        print(f"{'='*60}")

        # Wait for completion
        final_status = self.wait_for_completion(job_id, poll_interval, use_streaming)

        # Get result if completed
        if final_status['status'] == 'completed':
            print("\nDownloading result...")

            if output_path is None:
                video_path = Path(video_path)
                output_path = video_path.parent / f"{video_path.stem}_analysis.json"

            result = self.get_result(job_id, str(output_path))

            # Print summary
            print(f"\n{'='*60}")
            print("Processing Summary")
            print(f"{'='*60}")

            stats = result.get('processing_stats', {})
            print(f"Frames Analyzed: {stats.get('total_frames_analyzed', 0):,}")
            print(f"Processing Time: {stats.get('processing_time_minutes', 0):.1f} minutes")
            print(f"Speed: {stats.get('frames_per_second', 0):.2f} frames/second")
            print(f"Efficiency: {stats.get('efficiency_ratio', 0)*100:.1f}% of video length")

            return result

        return final_status

def main():
    parser = argparse.ArgumentParser(description="Submit video to processing pod")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--pod-url", required=True, help="Pod URL (e.g., http://your-pod-id.runpod.net)")
    parser.add_argument(
        "--mode",
        choices=["screen_share", "ui_detection", "meeting_analysis", "app_demo"],
        default="screen_share",
        help="Analysis mode"
    )
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--poll-interval", type=int, default=10, help="Status check interval (seconds)")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    parser.add_argument("--no-streaming", action="store_true", help="Use polling instead of streaming for progress updates")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        help="Chunk duration in seconds (default: 60s). Use 30-45s for detailed analysis, 90-120s for overview"
    )

    args = parser.parse_args()

    # Create client
    client = PodClient(args.pod_url)

    try:
        if args.no_wait:
            # Just submit
            job_info = client.submit_video(args.video_path, args.mode, args.chunk_duration)
            print(json.dumps(job_info, indent=2))
        else:
            # Submit and wait
            client.process_and_wait(
                args.video_path,
                mode=args.mode,
                output_path=args.output,
                poll_interval=args.poll_interval,
                use_streaming=not args.no_streaming,
                chunk_duration=args.chunk_duration
            )

    except requests.exceptions.RequestException as e:
        print(f"\nError communicating with pod: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
