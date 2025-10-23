"""
Example usage scripts for Qwen3-VL RunPod endpoint.
Demonstrates various ways to interact with the deployed serverless function.
"""

import requests
import time
import json


# Configuration
RUNPOD_ENDPOINT_ID = "your-endpoint-id-here"
RUNPOD_API_KEY = "your-api-key-here"
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"


def example_basic_video_analysis():
    """Basic video analysis with URL."""
    print("\n" + "="*80)
    print("Example 1: Basic Video Analysis")
    print("="*80)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "video_url": "https://example.com/sample_video.mp4",
            "prompt": "Describe what happens in this video."
        }
    }

    print(f"\nSending request to: {RUNPOD_API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(RUNPOD_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Success!")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"Response: {response.text}")


def example_detailed_analysis_with_parameters():
    """Advanced video analysis with custom parameters."""
    print("\n" + "="*80)
    print("Example 2: Detailed Analysis with Parameters")
    print("="*80)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "video_url": "https://example.com/long_video.mp4",
            "prompt": """Analyze this video and provide:
            1. A description of the main events
            2. Key objects or people visible
            3. Any text or signage shown
            4. The overall setting and atmosphere
            """,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "chunk_duration": 45.0,  # 45 second chunks
            "aggregate": True,
            "aggregation_prompt": "Provide a comprehensive summary of the entire video, highlighting the main events in chronological order:"
        }
    }

    print(f"\nSending detailed analysis request...")
    response = requests.post(RUNPOD_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Success!")

        # Extract key information
        output = result.get("output", {})
        if output.get("success"):
            print(f"\nVideo Duration: {output['video_info']['duration']:.2f}s")
            print(f"Number of Chunks: {output['num_chunks']}")
            print(f"Processing Time: {output['total_processing_time']:.2f}s")

            print(f"\n--- Aggregated Summary ---")
            print(output.get("aggregated_response", "No summary available"))

            print(f"\n--- Individual Chunk Results ---")
            for chunk in output.get("chunk_results", []):
                print(f"\nChunk {chunk['chunk_index']} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):")
                print(f"  {chunk['response']}")

    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"Response: {response.text}")


def example_async_processing():
    """Async job submission and polling."""
    print("\n" + "="*80)
    print("Example 3: Async Processing with Job Polling")
    print("="*80)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    # Submit async job
    async_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"

    payload = {
        "input": {
            "video_url": "https://example.com/very_long_video.mp4",
            "prompt": "Analyze this video and describe all key events.",
            "max_tokens": 512
        }
    }

    print(f"\nSubmitting async job...")
    response = requests.post(async_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"\n✗ Error submitting job: {response.status_code}")
        return

    job_data = response.json()
    job_id = job_data.get("id")
    print(f"✓ Job submitted: {job_id}")

    # Poll for results
    status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"

    print("\nPolling for results...")
    max_attempts = 60
    attempt = 0

    while attempt < max_attempts:
        time.sleep(5)  # Wait 5 seconds between polls
        attempt += 1

        status_response = requests.get(status_url, headers=headers)

        if status_response.status_code != 200:
            print(f"\n✗ Error checking status: {status_response.status_code}")
            break

        status_data = status_response.json()
        job_status = status_data.get("status")

        print(f"  Attempt {attempt}/{max_attempts}: Status = {job_status}")

        if job_status == "COMPLETED":
            print(f"\n✓ Job completed!")
            result = status_data.get("output", {})
            print(f"Result: {json.dumps(result, indent=2)}")
            break

        elif job_status == "FAILED":
            print(f"\n✗ Job failed!")
            print(f"Error: {status_data.get('error')}")
            break

        elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
            continue

    if attempt >= max_attempts:
        print(f"\n⚠ Timeout: Job did not complete within {max_attempts * 5}s")


def example_batch_processing():
    """Process multiple videos in batch."""
    print("\n" + "="*80)
    print("Example 4: Batch Video Processing")
    print("="*80)

    videos = [
        "https://example.com/video1.mp4",
        "https://example.com/video2.mp4",
        "https://example.com/video3.mp4",
    ]

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    async_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"

    job_ids = []

    # Submit all jobs
    print(f"\nSubmitting {len(videos)} videos for processing...")
    for i, video_url in enumerate(videos):
        payload = {
            "input": {
                "video_url": video_url,
                "prompt": "Describe what happens in this video."
            }
        }

        response = requests.post(async_url, json=payload, headers=headers)

        if response.status_code == 200:
            job_id = response.json().get("id")
            job_ids.append((i, job_id, video_url))
            print(f"  ✓ Video {i+1}: Job {job_id}")
        else:
            print(f"  ✗ Video {i+1}: Failed to submit")

    # Wait for all results
    print(f"\nWaiting for results...")
    results = {}

    for idx, job_id, video_url in job_ids:
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"

        # Poll until complete
        max_attempts = 60
        for attempt in range(max_attempts):
            time.sleep(3)

            status_response = requests.get(status_url, headers=headers)
            if status_response.status_code != 200:
                break

            status_data = status_response.json()
            job_status = status_data.get("status")

            if job_status == "COMPLETED":
                results[idx] = status_data.get("output")
                print(f"  ✓ Video {idx+1} completed")
                break

            elif job_status == "FAILED":
                results[idx] = {"error": status_data.get("error")}
                print(f"  ✗ Video {idx+1} failed")
                break

    print(f"\n{'='*80}")
    print(f"Batch Processing Complete: {len(results)}/{len(videos)} videos processed")
    print(f"{'='*80}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Qwen3-VL RunPod Endpoint - Usage Examples")
    print("="*80)

    # Check configuration
    if RUNPOD_ENDPOINT_ID == "your-endpoint-id-here":
        print("\n⚠️  Please configure RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY first!")
        print("\nUpdate the variables at the top of this file:")
        print("  RUNPOD_ENDPOINT_ID = 'your-endpoint-id-here'")
        print("  RUNPOD_API_KEY = 'your-api-key-here'")
        return

    print("\nAvailable Examples:")
    print("1. Basic video analysis")
    print("2. Detailed analysis with parameters")
    print("3. Async processing with job polling")
    print("4. Batch video processing")
    print("5. Run all examples")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        example_basic_video_analysis()
    elif choice == "2":
        example_detailed_analysis_with_parameters()
    elif choice == "3":
        example_async_processing()
    elif choice == "4":
        example_batch_processing()
    elif choice == "5":
        example_basic_video_analysis()
        example_detailed_analysis_with_parameters()
        example_async_processing()
        example_batch_processing()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
