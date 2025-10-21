"""
RunPod Serverless Handler for Faster-Whisper Transcription
"""
import runpod
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import numpy as np

# Load model at startup
print("Loading Faster-Whisper large-v3-turbo model...")
model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
    download_root="/models"
)
print("Model loaded successfully!")

def handler(event):
    """
    Handler for Faster-Whisper transcription

    Input format:
    {
        "input": {
            "audio": "base64_encoded_audio",
            "language": "en",  # optional, auto-detect if not provided
            "task": "transcribe",  # or "translate"
            "vad_filter": true,
            "word_timestamps": false
        }
    }

    Output format:
    {
        "transcription": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "confidence": 0.95
            },
            ...
        ],
        "language": "en",
        "duration": 10.5
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Decode audio
        audio_b64 = input_data.get('audio')
        if not audio_b64:
            return {"error": "No audio provided"}

        audio_bytes = base64.b64decode(audio_b64)
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Get parameters
        language = input_data.get('language', None)
        task = input_data.get('task', 'transcribe')
        vad_filter = input_data.get('vad_filter', True)
        word_timestamps = input_data.get('word_timestamps', False)

        # Transcribe
        segments, info = model.transcribe(
            audio,
            language=language,
            task=task,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            beam_size=5
        )

        # Format output
        transcription = []
        for segment in segments:
            transcription.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob
            })

        processing_time = time.time() - start_time

        return {
            'transcription': transcription,
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'processing_time': processing_time
        }

    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
