# Windows Quick Start Guide

Complete guide to run this MLOps project on Windows from start to finish.

## Prerequisites

1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation

2. **Git** - Download from [git-scm.com](https://git-scm.com/download/win)

3. **ngrok** (for Kaggle integration) - Download from [ngrok.com](https://ngrok.com/download)
   - Extract to a folder in your PATH (e.g., `C:\ngrok\`)

## Step 1: Initial Setup

Open **PowerShell** (not CMD):

```powershell
# Clone or navigate to project
cd C:\path\to\End-to-End-Basic-ML-Project

# Run automated setup
.\setup_windows.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 2: Start MLflow Server

In PowerShell:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate

# Start MLflow (configured for ngrok)
python start_mlflow_for_ngrok.py
```

Keep this terminal open! You should see:
```
🚀 Starting MLflow server for ngrok access...
```

## Step 3: Start ngrok (in New Terminal)

Open a **NEW PowerShell** window:

```powershell
# Start ngrok
ngrok http 5000
```

You'll see something like:
```
Forwarding: https://xxxx-xxx-xxx.ngrok-free.app -> http://localhost:5000
```

**Copy the ngrok URL** - you'll need it for Kaggle!

Test it: Open the URL in your browser - you should see the MLflow UI.

## Step 4: Train on Kaggle

1. **Upload your dataset** to Kaggle
2. **Create a new notebook** on Kaggle
3. **Add your training script** (from `kaggle_training/train_roberta_fakenews.py`)
4. **Set the MLflow tracking URI**:
   ```python
   import os
   os.environ["MLFLOW_TRACKING_URI"] = "https://YOUR-NGROK-URL.ngrok-free.app"
   ```
5. **Run the notebook** with GPU accelerator
6. **Watch metrics** flow into your local MLflow UI!

## Step 5: Download Model from Kaggle

After training completes:

1. In Kaggle, navigate to **Output** tab
2. Download the `roberta-large` folder
3. Save to your **Downloads** folder:
   ```
   C:\Users\YourName\Downloads\roberta-large\
   ```

## Step 6: Integrate Model into Project

Open a **NEW PowerShell** terminal (keep MLflow running):

```powershell
cd C:\path\to\End-to-End-Basic-ML-Project
.\venv\Scripts\Activate

# Option 1: Use Windows junction (saves space, recommended)
python kaggle_training\integrate_kaggle_model.py `
  --kaggle-model-path "C:\Users\YourName\Downloads\roberta-large" `
  --model-name roberta_fakenews_model `
  --symlink

# Option 2: Move files (saves space but empties download folder)
python kaggle_training\integrate_kaggle_model.py `
  --kaggle-model-path "C:\Users\YourName\Downloads\roberta-large" `
  --model-name roberta_fakenews_model `
  --move

# Option 3: Copy files (safest but uses more space)
python kaggle_training\integrate_kaggle_model.py `
  --kaggle-model-path "C:\Users\YourName\Downloads\roberta-large" `
  --model-name roberta_fakenews_model
```

**Note**: The script automatically skips checkpoint folders to save space!

## Step 7: Test the Model

```powershell
# Test fake news detection
python -c "from src.Pipelines.predict_pipeline_fakenews import FakeNewsPredictionPipeline; p = FakeNewsPredictionPipeline(); result = p.predict_single('Breaking News', 'Scientists discover chocolate cures everything'); print(f'Prediction: {result[\"prediction\"]}, Confidence: {result[\"confidence\"]:.2%}')"
```

## Step 8: Version with DVC

```powershell
# Track model with DVC
dvc add artifacts\transformers\roberta_fakenews_model

# Commit to Git
git add artifacts\transformers\roberta_fakenews_model.dvc .gitignore
git commit -m "Add RoBERTa fake news detection model"
git push
```

## Step 9: Run Flask Application

```powershell
# Start Flask app
python app.py
```

Open browser: `http://localhost:8080`

## Step 10: Run with Docker (Optional)

```powershell
# Build and run all services
docker-compose up --build

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Common Issues & Solutions

### 1. "No space left on device"

**Solution**: Use `--symlink` or `--move` option:
```powershell
python kaggle_training\integrate_kaggle_model.py `
  --kaggle-model-path "C:\Users\YourName\Downloads\roberta-large" `
  --model-name roberta_fakenews_model `
  --symlink
```

### 2. "Invalid Host header" with ngrok

**Solution**: The `start_mlflow_for_ngrok.py` script already handles this!

### 3. PowerShell execution policy error

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Python not in PATH

**Solution**: Reinstall Python and check "Add Python to PATH" during installation.

### 5. Virtual environment activation fails

**Solution**:
```powershell
# Use full path
C:\path\to\project\venv\Scripts\Activate.ps1
```

### 6. ngrok session expired

**Solution**: Free ngrok URLs expire. Just restart ngrok:
```powershell
# Kill ngrok (Ctrl+C) and restart
ngrok http 5000
```
Update the URL in your Kaggle notebook.

### 7. MLflow not showing Kaggle experiments

**Solution**: Check:
- MLflow server is running
- ngrok is running
- ngrok URL is correct in Kaggle
- Open ngrok URL in browser - should show MLflow UI
- Check Kaggle output for connection errors

---

## File Locations (Windows Paths)

- **Project**: `C:\Users\YourName\Desktop\End-to-End-Basic-ML-Project`
- **Virtual Environment**: `C:\...\End-to-End-Basic-ML-Project\venv`
- **Models**: `C:\...\End-to-End-Basic-ML-Project\artifacts\transformers`
- **MLflow Data**: `C:\...\End-to-End-Basic-ML-Project\mlruns`
- **DVC Cache**: `C:\...\End-to-End-Basic-ML-Project\.dvc\cache`

---

## Quick Commands Reference

```powershell
# Activate venv
.\venv\Scripts\Activate

# Start MLflow
python start_mlflow_for_ngrok.py

# Start ngrok
ngrok http 5000

# Integrate model (junction/symlink)
python kaggle_training\integrate_kaggle_model.py --kaggle-model-path "<path>" --model-name roberta_fakenews_model --symlink

# Test model
python -c "from src.Pipelines.predict_pipeline_fakenews import FakeNewsPredictionPipeline; p = FakeNewsPredictionPipeline(); print(p.predict_single('title', 'text'))"

# Version with DVC
dvc add artifacts\transformers\roberta_fakenews_model
git add *.dvc .gitignore
git commit -m "message"
git push

# Run Flask app
python app.py

# Run with Docker
docker-compose up --build
```

---

## MLOps Workflow Summary

1. **Setup** → `.\setup_windows.ps1`
2. **Start MLflow** → `python start_mlflow_for_ngrok.py`
3. **Start ngrok** → `ngrok http 5000`
4. **Train on Kaggle** → Use ngrok URL for tracking
5. **Download** → Save model from Kaggle output
6. **Integrate** → `python kaggle_training\integrate_kaggle_model.py --symlink`
7. **Test** → Verify predictions work
8. **Version** → `dvc add` + `git commit`
9. **Deploy** → `docker-compose up` or `python app.py`
10. **Repeat** → Train new models, integrate, version, deploy!

---

## Need Help?

- MLflow UI: `http://localhost:5000`
- Flask App: `http://localhost:8080`
- Documentation: See `KAGGLE_INTEGRATION.md`, `MLFLOW_KAGGLE_SETUP.md`
- Issues: Check the error message and refer to "Common Issues" above

Happy ML-ing on Windows! 🚀
