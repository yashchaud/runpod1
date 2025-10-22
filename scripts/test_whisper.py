"""
Test Whisper endpoint to diagnose issues
"""
import os
import sys
import base64
import requests
import tempfile
import subprocess
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
WHISPER_ENDPOINT_ID = os.getenv('RUNPOD_WHISPER_ENDPOINT_ID')

def test_endpoint_status():
    """Check if endpoint is accessible"""
    url = f"https://api.runpod.ai/v2/{WHISPER_ENDPOINT_ID}/status"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

    print(f"Testing Whisper endpoint: {WHISPER_ENDPOINT_ID}")
    print(f"URL: {url}")
    print()

    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def create_test_audio():
    """Create a simple test audio file"""
    # Create 3-second silence WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()

    # Use ffmpeg to create test audio
    cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
        '-t', '3', '-acodec', 'pcm_s16le',
        '-y', temp_file.name
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        with open(temp_file.name, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(temp_file.name)
        return audio_b64
    except Exception as e:
        print(f"Failed to create test audio: {e}")
        return None

def test_whisper_request(audio_b64):
    """Test actual Whisper request"""
    url = f"https://api.runpod.ai/v2/{WHISPER_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "audio": audio_b64,
            "vad_filter": True
        }
    }

    print("\nTesting Whisper transcription...")
    print(f"Audio size: {len(audio_b64)} bytes (base64)")
    print()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*60)
    print("Whisper Endpoint Diagnostics")
    print("="*60)
    print()

    # Test 1: Endpoint status
    print("Test 1: Checking endpoint status...")
    if not test_endpoint_status():
        print("\n❌ Endpoint is not accessible or not deployed")
        print("\nPossible fixes:")
        print("1. Check if endpoint is deployed in RunPod dashboard")
        print("2. Verify RUNPOD_WHISPER_ENDPOINT_ID in .env file")
        print("3. Check if endpoint has workers available")
        return

    print("\n✅ Endpoint is accessible")

    # Test 2: Create test audio
    print("\nTest 2: Creating test audio...")
    audio_b64 = create_test_audio()
    if not audio_b64:
        print("\n❌ Failed to create test audio")
        print("Make sure ffmpeg is installed")
        return

    print("✅ Test audio created")

    # Test 3: Test transcription
    print("\nTest 3: Testing transcription...")
    if test_whisper_request(audio_b64):
        print("\n✅ Whisper endpoint is working!")
    else:
        print("\n❌ Whisper request failed")
        print("\nPossible fixes:")
        print("1. Endpoint may not have the updated handler")
        print("2. Rebuild and redeploy Docker image")
        print("3. Check RunPod logs for errors")

if __name__ == '__main__':
    main()
