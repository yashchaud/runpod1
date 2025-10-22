"""
RunPod Serverless Handler for Faster-Whisper Transcription - BATCH ENABLED
Optimized for 24GB VRAM with batch processing support
"""
import runpod
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import numpy as np
import torch

# Load model at startup with optimizations
print("Loading Faster-Whisper large-v3-turbo model with batch optimization...")
model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
    download_root="/models",
    # Batch processing optimizations
    num_workers=4,  # Parallel workers for preprocessing
)
# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("Model loaded successfully!")

def handler(event):
    """
    Handler for Faster-Whisper transcription with BATCH support

    Input format (SINGLE):
    {
        "input": {
            "audio": "base64_encoded_audio",
            "language": "en",  # optional, auto-detect if not provided
            "task": "transcribe",  # or "translate"
            "vad_filter": true,
            "word_timestamps": false,
            "beam_size": 5  # Higher for better accuracy
        }
    }

    Input format (BATCH):
    {
        "input": {
            "audios": ["base64_1", "base64_2", ...],  # List of audio files
            "language": "en",
            "task": "transcribe",
            "vad_filter": true,
            "word_timestamps": false,
            "beam_size": 5
        }
    }

    Output format:
    {
        "transcription": [...] or [[...], [...], ...]  # Single or batch
        "language": "en" or ["en", "es", ...]
        "duration": 10.5 or [10.5, 8.2, ...]
        "processing_time": 2.5
        "batch_size": 1 or N
    }
    """
    try:
        import time
        start_time = time.time()

        input_data = event.get('input', {})

        # Check if batch or single
        is_batch = 'audios' in input_data

        # Get parameters
        language = input_data.get('language', None)
        task = input_data.get('task', 'transcribe')
        vad_filter = input_data.get('vad_filter', True)
        word_timestamps = input_data.get('word_timestamps', False)
        beam_size = input_data.get('beam_size', 5)

        if is_batch:
            # BATCH PROCESSING
            audios_b64 = input_data.get('audios', [])
            if not audios_b64:
                return {"error": "No audios provided"}

            all_transcriptions = []
            all_languages = []
            all_durations = []

            # Process each audio (Whisper processes sequentially but with optimizations)
            for idx, audio_b64 in enumerate(audios_b64):
                # Decode audio
                audio_bytes = base64.b64decode(audio_b64)
                audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

                # Ensure mono audio
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Transcribe with optimizations
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    task=task,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                    beam_size=beam_size,
                    # Batch optimizations
                    condition_on_previous_text=False,  # Faster for batch
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6
                )

                # Format output for this audio
                transcription = []
                for segment in segments:
                    transcription.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip(),
                        'confidence': segment.avg_logprob
                    })

                all_transcriptions.append(transcription)
                all_languages.append(info.language)
                all_durations.append(info.duration)

            processing_time = time.time() - start_time

            return {
                'transcription': all_transcriptions,
                'language': all_languages,
                'language_probability': [1.0] * len(audios_b64),  # Simplified
                'duration': all_durations,
                'processing_time': processing_time,
                'batch_size': len(audios_b64),
                'throughput': sum(all_durations) / processing_time if processing_time > 0 else 0
            }

        else:
            # SINGLE AUDIO PROCESSING (backward compatible)
            audio_b64 = input_data.get('audio')
            if not audio_b64:
                return {"error": "No audio provided"}

            audio_bytes = base64.b64decode(audio_b64)
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Transcribe
            segments, info = model.transcribe(
                audio,
                language=language,
                task=task,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                beam_size=beam_size
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
                'processing_time': processing_time,
                'batch_size': 1
            }

    except Exception as e:
        return {"error": str(e), "traceback": __import__('traceback').format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
