#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify Phase 0 & 1 setup"""
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_imports():
    print("Testing Python package imports...")
    packages = [
        ('requests', 'requests'),
        ('runpod', 'runpod'),
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('numpy', 'numpy'),
        ('redis', 'redis'),
        ('psycopg2', 'psycopg2-binary'),
    ]

    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
            failed.append(package_name)

    if failed:
        print(f"\n⚠ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_env():
    print("\nChecking environment variables...")
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('RUNPOD_API_KEY')
        if api_key and api_key != 'your_runpod_api_key_here':
            print(f"  ✓ RUNPOD_API_KEY is set")
            return True
        else:
            print(f"  ⚠ RUNPOD_API_KEY not set in .env file")
            print(f"    Please edit .env and add your RunPod API key")
            return False
    except ImportError:
        print("  ⚠ python-dotenv not installed")
        return False

def test_directory_structure():
    print("\nChecking directory structure...")
    required_dirs = [
        'config',
        'services/ingestion',
        'services/detection',
        'services/audio',
        'services/vlm',
        'data/videos',
        'data/frames',
        'logs',
        'scripts',
        'utils',
        'tests'
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} missing")
            all_exist = False

    return all_exist

if __name__ == '__main__':
    print("=" * 50)
    print("PHASE 0 & 1 VERIFICATION TEST")
    print("=" * 50)
    print()

    dirs_ok = test_directory_structure()
    imports_ok = test_imports()
    env_ok = test_env()

    print()
    print("=" * 50)
    if dirs_ok and imports_ok:
        print("✓ Phase 1 setup complete!")
        if env_ok:
            print("✓ RunPod API key configured!")
            print("  Next: Run Phase 2 (Database setup)")
        else:
            print("⚠ Phase 0: Add your RunPod API key to .env")
            print("  1. Go to https://www.runpod.io/")
            print("  2. Create account and get API key")
            print("  3. Edit .env file and replace 'your_runpod_api_key_here'")
    else:
        print("⚠ Some issues need to be fixed")
        sys.exit(1)
    print("=" * 50)
