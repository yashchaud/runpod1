#!/usr/bin/env python
"""
Quick runner script for parallel video processing
Usage: python run.py <video_file.mp4> [fps] [max_workers] [batch_size]
"""
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import and run the parallel processor
from process_video_parallel import main

if __name__ == '__main__':
    main()
