@echo off
REM Quick command to process a video with YOLO and Whisper
REM Usage: process.bat <video_file.mp4> [fps] [output_dir]
REM Example: process.bat video.mp4
REM Example: process.bat video.mp4 2 results

REM Run with unbuffered output (-u flag)
python -u scripts/process_video.py %*
