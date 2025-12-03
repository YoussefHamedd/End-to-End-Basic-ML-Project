# Windows Setup Script for End-to-End ML Project
# Run this with: .\setup_windows.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   ML Project Setup for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.8+ from python.org" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# Check Git
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "  ✓ Found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Git not found. Install from git-scm.com" -ForegroundColor Yellow
}

# Initialize DVC if not already done
if (Test-Path ".dvc") {
    Write-Host "  ✓ DVC already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing DVC..." -ForegroundColor Yellow
    dvc init
    Write-Host "  ✓ DVC initialized" -ForegroundColor Green
}

# Create artifacts directory
Write-Host "Creating artifacts directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "artifacts/transformers" | Out-Null
Write-Host "  ✓ Artifacts directory ready" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Start MLflow: python start_mlflow_for_ngrok.py"
Write-Host "  2. In new terminal, start ngrok: ngrok http 5000"
Write-Host "  3. Train on Kaggle with ngrok URL"
Write-Host "  4. Download model from Kaggle"
Write-Host "  5. Integrate: python kaggle_training\integrate_kaggle_model.py --kaggle-model-path <path> --model-name roberta_fakenews_model --symlink"
Write-Host ""
