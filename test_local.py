"""
Local testing script for Qwen3-VL video processing.
Run this to test the system without deploying to RunPod.
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from handler import process_video_job, initialize_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_with_video_url():
    """Test with a video URL."""
    logger.info("Testing with video URL...")

    # Initialize services
    initialize_services()

    # Test input
    job_input = {
        "video_url": "https://download.samplelib.com/mp4/sample-5s.mp4",  # 5 second sample
        "prompt": "Describe what you see in this video.",
        "max_tokens": 256,
        "aggregate": True
    }

    # Process
    result = process_video_job(job_input)

    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nSuccess: {result.get('success')}")

    if result.get('success'):
        print(f"\nVideo Info:")
        for key, value in result.get('video_info', {}).items():
            print(f"  {key}: {value}")

        print(f"\nProcessing Summary:")
        print(f"  Total Chunks: {result.get('num_chunks')}")
        print(f"  Successful: {result.get('total_successful_chunks')}")
        print(f"  Total Time: {result.get('total_processing_time'):.2f}s")

        print(f"\nChunk Results:")
        for chunk in result.get('chunk_results', []):
            print(f"\n  Chunk {chunk['chunk_index']} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):")
            print(f"    Success: {chunk['success']}")
            print(f"    Time: {chunk['processing_time']:.2f}s")
            if chunk['response']:
                print(f"    Response: {chunk['response'][:100]}...")

        if result.get('aggregated_response'):
            print(f"\nAggregated Summary:")
            print(f"  {result['aggregated_response']}")

    else:
        print(f"\nError: {result.get('error')}")
        if result.get('traceback'):
            print(f"\nTraceback:\n{result['traceback']}")

    print("\n" + "="*80)


def test_with_local_video():
    """Test with a local video file."""
    video_path = input("Enter path to local video file: ").strip()

    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        return

    logger.info(f"Testing with local video: {video_path}")

    # Initialize services
    initialize_services()

    # Test input
    job_input = {
        "video_path": video_path,
        "prompt": "Describe what you see in this video in detail.",
        "max_tokens": 512,
        "aggregate": True
    }

    # Process
    result = process_video_job(job_input)

    # Print results (same as above)
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nSuccess: {result.get('success')}")

    if result.get('success'):
        print(f"\nVideo Info:")
        for key, value in result.get('video_info', {}).items():
            print(f"  {key}: {value}")

        print(f"\nProcessing Summary:")
        print(f"  Total Chunks: {result.get('num_chunks')}")
        print(f"  Successful: {result.get('total_successful_chunks')}")
        print(f"  Total Time: {result.get('total_processing_time'):.2f}s")

        if result.get('aggregated_response'):
            print(f"\nAggregated Summary:")
            print(f"  {result['aggregated_response']}")

    else:
        print(f"\nError: {result.get('error')}")

    print("\n" + "="*80)


def test_video_validation():
    """Test video validation only."""
    from video_processor import VideoProcessor

    video_path = input("Enter path to video file to validate: ").strip()

    processor = VideoProcessor()
    is_valid, message = processor.validate_video(video_path)

    print(f"\nValidation Result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Message: {message}")

    if is_valid:
        from frame_extractor import FrameExtractor
        extractor = FrameExtractor()
        video_info = extractor.get_video_info(video_path)

        print("\nVideo Information:")
        for key, value in video_info.items():
            print(f"  {key}: {value}")

        # Calculate chunks
        chunks = processor.calculate_optimal_chunks(video_info)
        print(f"\nCalculated Chunks: {len(chunks)}")
        for i, (start, end) in enumerate(chunks):
            print(f"  Chunk {i}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")


def main():
    """Main test menu."""
    print("\n" + "="*80)
    print("Qwen3-VL Video Processing - Local Test")
    print("="*80)
    print("\nTest Options:")
    print("1. Test with sample video URL")
    print("2. Test with local video file")
    print("3. Test video validation only")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        test_with_video_url()
    elif choice == "2":
        test_with_local_video()
    elif choice == "3":
        test_video_validation()
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return


if __name__ == "__main__":
    # Check environment
    if not os.environ.get("MODEL_NAME"):
        print("⚠️  Warning: MODEL_NAME not set, using default")
        os.environ["MODEL_NAME"] = "Qwen/Qwen3-VL-8B-Instruct-FP8"

    if not os.environ.get("HF_TOKEN"):
        print("⚠️  Warning: HF_TOKEN not set - model download may fail")
        hf_token = input("Enter HuggingFace token (or press Enter to continue): ").strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
