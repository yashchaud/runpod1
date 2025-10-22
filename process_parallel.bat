@echo off
REM Quick command to process a video with YOLO and Whisper - PARALLEL VERSION
REM Usage: process_parallel.bat <video_file.mp4> [fps] [max_workers] [batch_size] [output_dir]
REM Example: process_parallel.bat video.mp4
REM Example: process_parallel.bat video.mp4 2 5 16 results

REM Defaults:
REM   fps: 1
REM   max_workers: 5 (concurrent requests to RunPod)
REM   batch_size: 16 (frames per batch - optimized for 24GB VRAM)
REM   output_dir: output/

REM Run with unbuffered output (-u flag)
python -u scripts/process_video_parallel.py %*
