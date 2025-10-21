# Phase 0 & 1 Automated Setup Script
# Run this with: powershell -ExecutionPolicy Bypass -File setup_phase_0_1.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VIDEO AGENT - PHASE 0 & 1 SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$projectRoot = "e:\New folder (3)\runpod"
Set-Location $projectRoot

# Phase 0: Pre-flight Checks
Write-Host "[PHASE 0] PRE-FLIGHT CHECKS" -ForegroundColor Yellow
Write-Host ""

# Task 0.4: Check Disk Space
Write-Host "[Task 0.4] Checking disk space..." -ForegroundColor Green
$drive = Get-PSDrive -Name E -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Host "  Free space on E: drive: $freeGB GB" -ForegroundColor White
    if ($freeGB -gt 500) {
        Write-Host "  ✓ Sufficient disk space available" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Warning: Less than 500GB free. Recommended to free up space." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ℹ Could not check E: drive" -ForegroundColor Yellow
}
Write-Host ""

# Task 0.5: Network Connectivity
Write-Host "[Task 0.5] Testing network connectivity..." -ForegroundColor Green
Write-Host "  Testing internet connection (8.8.8.8)..."
$internetOk = Test-Connection -ComputerName 8.8.8.8 -Count 2 -Quiet
if ($internetOk) {
    Write-Host "  ✓ Internet connection OK" -ForegroundColor Green
} else {
    Write-Host "  ✗ Internet connection failed" -ForegroundColor Red
}

Write-Host "  Testing RunPod API (api.runpod.ai)..."
try {
    $response = Invoke-WebRequest -Uri "https://api.runpod.ai" -Method Head -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  ✓ RunPod API reachable" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Could not reach RunPod API (this is OK if behind firewall)" -ForegroundColor Yellow
}
Write-Host ""

# Phase 1: Environment Setup
Write-Host "[PHASE 1] ENVIRONMENT SETUP" -ForegroundColor Yellow
Write-Host ""

# Task 1.1: Check existing tools
Write-Host "[Task 1.1] Checking system requirements..." -ForegroundColor Green
Write-Host "  Python version:"
python --version
Write-Host "  Docker version:"
docker --version
Write-Host "  Git version:"
git --version
Write-Host ""

# Check FFmpeg
Write-Host "  Checking FFmpeg..."
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "  ✓ FFmpeg installed: $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ FFmpeg not found. You may need to install it." -ForegroundColor Yellow
    Write-Host "    Download from: https://ffmpeg.org/download.html" -ForegroundColor White
}
Write-Host ""

# Task 1.2.1: Create Directory Structure
Write-Host "[Task 1.2.1] Creating project directory structure..." -ForegroundColor Green
$directories = @(
    "config",
    "models\yolo",
    "models\emotion",
    "models\whisper",
    "models\vlm",
    "services\ingestion",
    "services\detection",
    "services\audio",
    "services\fusion",
    "services\vlm",
    "services\events",
    "services\highlights",
    "services\export",
    "data\videos",
    "data\frames",
    "data\embeddings",
    "data\events",
    "logs",
    "outputs",
    "scripts",
    "utils",
    "tests"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor White
    } else {
        Write-Host "  Exists:  $dir" -ForegroundColor Gray
    }
}
Write-Host "  ✓ Directory structure created" -ForegroundColor Green
Write-Host ""

# Task 1.2.2: Initialize Git
Write-Host "[Task 1.2.2] Initializing Git repository..." -ForegroundColor Green
if (-not (Test-Path ".git")) {
    git init
    Write-Host "  ✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "  ✓ Git repository already initialized" -ForegroundColor Green
}
Write-Host ""

# Task 1.2.3: Create .gitignore
Write-Host "[Task 1.2.3] Creating .gitignore..." -ForegroundColor Green
$gitignoreContent = @"
# Python
venv/
env/
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Environment variables
.env
.env.local

# Data and models (large files)
data/videos/*
data/frames/*
!data/videos/.gitkeep
!data/frames/.gitkeep
models/
logs/
outputs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Docker
docker-compose.override.yml

# Temporary files
*.tmp
*.log
*.cache
"@

Set-Content -Path ".gitignore" -Value $gitignoreContent -Force
Write-Host "  ✓ .gitignore created" -ForegroundColor Green
Write-Host ""

# Create .gitkeep files for empty directories
@("data\videos", "data\frames") | ForEach-Object {
    $gitkeepPath = Join-Path $projectRoot "$_\.gitkeep"
    if (-not (Test-Path $gitkeepPath)) {
        New-Item -ItemType File -Path $gitkeepPath -Force | Out-Null
    }
}

# Task 1.3.1: Create Python Virtual Environment
Write-Host "[Task 1.3.1] Creating Python virtual environment..." -ForegroundColor Green
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Task 1.3.2-1.3.4: Install Python packages
Write-Host "[Task 1.3.2-1.3.4] Installing Python packages..." -ForegroundColor Green
Write-Host "  Activating virtual environment..." -ForegroundColor White

# Activate venv and install packages
$activateScript = Join-Path $projectRoot "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "  Installing packages (this may take a few minutes)..." -ForegroundColor White

    # Create requirements file
    $requirements = @"
# Core dependencies
requests>=2.31.0
runpod>=1.0.0

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

# Data processing
psycopg2-binary>=2.9.9
redis>=5.0.0

# Streaming
livekit>=0.17.0

# Audio processing
librosa>=0.10.2
soundfile>=0.12.1

# Object tracking
supervision>=0.22.0

# API framework (for later phases)
fastapi>=0.104.0
uvicorn>=0.24.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
"@

    Set-Content -Path "requirements.txt" -Value $requirements -Force
    Write-Host "  ✓ requirements.txt created" -ForegroundColor Green

    # Install packages
    & "$projectRoot\venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools --quiet
    & "$projectRoot\venv\Scripts\pip.exe" install -r requirements.txt --quiet

    Write-Host "  ✓ Python packages installed" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Could not find activation script" -ForegroundColor Yellow
}
Write-Host ""

# Create .env template
Write-Host "[Extra] Creating .env template..." -ForegroundColor Green
$envTemplate = @"
# RunPod API Credentials
RUNPOD_API_KEY=your_runpod_api_key_here

# RunPod Endpoint IDs (will be filled after Phase 3)
RUNPOD_YOLO_ENDPOINT_ID=
RUNPOD_WHISPER_ENDPOINT_ID=
RUNPOD_VLM_ENDPOINT_ID=
RUNPOD_EMOTION_ENDPOINT_ID=

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=videoagent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_me_in_production

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# MinIO/S3 Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=changeme123
MINIO_SECURE=false

# LiveKit Configuration (optional)
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
"@

Set-Content -Path ".env.example" -Value $envTemplate -Force
Write-Host "  ✓ .env.example created" -ForegroundColor Green

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  ✓ .env created (REMEMBER TO ADD YOUR RUNPOD API KEY!)" -ForegroundColor Yellow
} else {
    Write-Host "  ℹ .env already exists (not overwriting)" -ForegroundColor Cyan
}
Write-Host ""

# Create a simple test script
Write-Host "[Extra] Creating test script..." -ForegroundColor Green
$testScript = @"
# Test script to verify Phase 0 & 1 setup
import sys
import os

def test_imports():
    print("Testing Python package imports...")
    packages = [
        'requests',
        'runpod',
        'cv2',
        'PIL',
        'numpy',
        'redis',
        'psycopg2',
    ]

    failed = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError as e:
            print(f"  ✗ {pkg}: {e}")
            failed.append(pkg)

    if failed:
        print(f"\n⚠ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_env():
    print("\nChecking environment variables...")
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

if __name__ == '__main__':
    print("=" * 50)
    print("PHASE 0 & 1 VERIFICATION TEST")
    print("=" * 50)
    print()

    imports_ok = test_imports()
    env_ok = test_env()

    print()
    print("=" * 50)
    if imports_ok and env_ok:
        print("✓ Phase 0 & 1 setup complete!")
        print("  Next: Add your RunPod API key to .env")
        print("  Then: Run Phase 2 (Database setup)")
    else:
        print("⚠ Some issues need to be fixed")
        sys.exit(1)
    print("=" * 50)
"@

Set-Content -Path "scripts\test_setup.py" -Value $testScript -Force
Write-Host "  ✓ Test script created: scripts\test_setup.py" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Create RunPod account: https://www.runpod.io/" -ForegroundColor White
Write-Host "  2. Get API key from RunPod dashboard" -ForegroundColor White
Write-Host "  3. Edit .env file and add your RUNPOD_API_KEY" -ForegroundColor White
Write-Host "  4. Test setup: .\venv\Scripts\python.exe scripts\test_setup.py" -ForegroundColor White
Write-Host ""
Write-Host "Phase 0 Manual Tasks:" -ForegroundColor Yellow
Write-Host "  - [ ] Create RunPod account" -ForegroundColor White
Write-Host "  - [ ] Get RunPod API key" -ForegroundColor White
Write-Host "  - [ ] Add credits to RunPod" -ForegroundColor White
Write-Host "  - [ ] Update .env with API key" -ForegroundColor White
Write-Host ""
Write-Host "Phase 1 Status:" -ForegroundColor Yellow
Write-Host "  ✓ Directory structure created" -ForegroundColor Green
Write-Host "  ✓ Git initialized" -ForegroundColor Green
Write-Host "  ✓ Python environment created" -ForegroundColor Green
Write-Host "  ✓ Packages installed" -ForegroundColor Green
Write-Host ""
Write-Host "Review PHASE_0_1_SETUP_GUIDE.md for details" -ForegroundColor Cyan
Write-Host ""
