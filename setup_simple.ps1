# Simple Phase 0 & 1 Setup Script
$projectRoot = "e:\New folder (3)\runpod"
Set-Location $projectRoot

Write-Host "Creating directory structure..." -ForegroundColor Green

$directories = @(
    "config",
    "services\ingestion",
    "services\detection",
    "services\audio",
    "services\vlm",
    "data\videos",
    "data\frames",
    "logs",
    "scripts",
    "utils",
    "tests"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "Created: $dir"
    }
}

Write-Host "Creating Python virtual environment..." -ForegroundColor Green
if (-not (Test-Path "venv")) {
    python -m venv venv
}

Write-Host "Installing packages..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\venv\Scripts\pip.exe" install requests runpod opencv-python pillow numpy psycopg2-binary redis python-dotenv --quiet

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Next: Create RunPod account and get API key" -ForegroundColor Yellow
