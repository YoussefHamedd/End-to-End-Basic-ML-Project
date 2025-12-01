# 🔬 MLflow Tracking from Kaggle

This guide shows you how to track your Kaggle experiments in MLflow, so all your training runs are logged to your MLflow server.

---

## 🎯 Three Options for MLflow Tracking

### Option 1: Use ngrok (Recommended - Track to Local MLflow)

This lets Kaggle send experiments to your local MLflow server.

#### Step 1: Start MLflow Locally

```bash
# On your Mac
cd ~/Desktop/End-to-End-Basic-ML-Project/End-to-End-Basic-ML-Project
source venv/bin/activate

# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000
```

Keep this terminal open!

#### Step 2: Expose MLflow with ngrok

**Install ngrok** (one-time):
```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

**Start ngrok** (in a new terminal):
```bash
ngrok http 5000
```

You'll see output like:
```
Forwarding  https://abc123def456.ngrok.io -> http://localhost:5000
```

**Copy the HTTPS URL** (e.g., `https://abc123def456.ngrok.io`)

#### Step 3: Configure in Kaggle Notebook

At the **top** of your Kaggle notebook:

```python
import os

# Set MLflow tracking URI to your ngrok URL
os.environ["MLFLOW_TRACKING_URI"] = "https://abc123def456.ngrok.io"

# Now run your training script
# It will automatically log to your local MLflow!
```

---

### Option 2: Use Deployed MLflow Server

If you have MLflow deployed somewhere (AWS, GCP, Heroku, etc.):

```python
import os

# Point to your deployed MLflow server
os.environ["MLFLOW_TRACKING_URI"] = "https://your-mlflow-server.com"
```

---

### Option 3: File-Based Tracking (No Server Needed)

Track experiments to files, download later:

```python
import os

# Track to local files (default)
os.environ["MLFLOW_TRACKING_URI"] = "file:///kaggle/working/mlruns"

# After training, download the mlruns folder from Kaggle output
# Then view locally with: mlflow ui --backend-store-uri ./mlruns
```

---

## 📝 Complete Kaggle Notebook Setup

Here's the **complete setup** for your Kaggle notebook:

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q transformers datasets torch mlflow
```

### Cell 2: Configure MLflow (Choose Your Option)

**Option A: With ngrok (recommended)**
```python
import os

# REPLACE with your ngrok URL
os.environ["MLFLOW_TRACKING_URI"] = "https://your-ngrok-url.ngrok.io"

print(f"✓ MLflow will track to: {os.environ['MLFLOW_TRACKING_URI']}")
```

**Option B: File-based (no setup needed)**
```python
import os

# Track to local files
os.environ["MLFLOW_TRACKING_URI"] = "file:///kaggle/working/mlruns"

print(f"✓ MLflow will track to local files")
```

### Cell 3: Training Code

```python
# Your complete training code here
# (paste from train_roberta_fakenews.py)
```

---

## 🔍 Viewing Experiments

### If Using ngrok or Deployed Server:

1. Open **http://localhost:5000** (if using ngrok)
2. You'll see your Kaggle runs in real-time!
3. Click on runs to see metrics, parameters, artifacts

### If Using File-Based Tracking:

1. After training, download `mlruns` folder from Kaggle output
2. On your Mac:
   ```bash
   cd ~/Downloads
   mlflow ui --backend-store-uri ./mlruns
   open http://localhost:5000
   ```

---

## 📊 What Gets Logged?

Your training automatically logs:

### Parameters
- Model name (roberta-large)
- Task type (fake_news_detection)
- Dataset (WELFake)
- Max length (256)
- Batch size (8)
- Learning rate (2e-5)
- Epochs (3)
- Train/Val/Test samples

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Test loss

### Artifacts
- Model (PyTorch format)
- Training arguments
- Tokenizer files
- Metadata

---

## 🎯 Complete Example Workflow

### On Your Mac (Terminal 1):

```bash
# Start MLflow
cd ~/Desktop/End-to-End-Basic-ML-Project/End-to-End-Basic-ML-Project
source venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
```

### On Your Mac (Terminal 2):

```bash
# Start ngrok
ngrok http 5000

# Copy the https URL shown
```

### In Kaggle Notebook:

```python
# Cell 1: Install
!pip install -q transformers datasets torch mlflow

# Cell 2: Configure MLflow
import os
os.environ["MLFLOW_TRACKING_URI"] = "https://YOUR-NGROK-URL.ngrok.io"

# Cell 3: Train (paste your training code)
# ... training happens ...

# Cell 4: Check if logged
import mlflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
experiment = mlflow.get_experiment_by_name("kaggle_fake_news_detection")
print(f"Experiment ID: {experiment.experiment_id}")
```

### Back on Your Mac:

1. Open **http://localhost:5000**
2. See your Kaggle training run appear!
3. View metrics, compare runs, download artifacts

---

## 🔄 After Training

Once training completes on Kaggle:

1. **Model saved** to `/kaggle/working/roberta_fakenews_model/`
2. **Logged to MLflow** automatically
3. **Download model** from Kaggle output
4. **Integrate locally**:
   ```bash
   python kaggle_training/integrate_kaggle_model.py \
       --kaggle-model-path ~/Downloads/roberta_fakenews_model \
       --model-name roberta_fakenews_model \
       --register
   ```

The integration script will link your local model to the MLflow run!

---

## 🆘 Troubleshooting

### ngrok URL keeps changing
ngrok free tier generates new URLs each time. Options:
- Copy new URL to Kaggle each time
- Use ngrok paid plan for static URLs
- Use deployed MLflow server

### "Connection refused" error
- Check MLflow is running: `curl http://localhost:5000/health`
- Check ngrok is running: Look for "Forwarding" line
- Make sure firewall allows connections

### Experiments not showing up
- Check tracking URI is correct: `echo $MLFLOW_TRACKING_URI`
- Look for MLflow logs in Kaggle output
- Try file-based tracking first to verify code works

### MLflow not installed on Kaggle
Add to first cell:
```python
!pip install mlflow
```

---

## 🎓 Best Practices

1. **Always set tracking URI** at the top of your notebook
2. **Use descriptive run names**: Include model, dataset, key params
3. **Log custom metrics** that matter for your task
4. **Tag important runs**: Add tags in MLflow UI
5. **Compare runs** before downloading models
6. **Document experiments** with notes in MLflow

---

## 🚀 Advanced: Multiple Experiments

Track different experiments separately:

```python
# Experiment 1: Baseline
os.environ["MLFLOW_TRACKING_URI"] = "https://your-ngrok-url.ngrok.io"
EXPERIMENT_NAME = "fake_news_baseline"
# ... train roberta-base ...

# Experiment 2: Large model
EXPERIMENT_NAME = "fake_news_large"
# ... train roberta-large ...

# Compare in MLflow UI
```

---

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ngrok Documentation](https://ngrok.com/docs)
- [MLflow on Kaggle Tutorial](https://www.kaggle.com/code/mlflow-tracking)

---

## ✅ Quick Checklist

Before training on Kaggle:

- [ ] MLflow running locally (`mlflow ui`)
- [ ] ngrok exposing MLflow (`ngrok http 5000`)
- [ ] ngrok URL copied
- [ ] Kaggle notebook has `MLFLOW_TRACKING_URI` set
- [ ] `mlflow` installed in Kaggle (`!pip install mlflow`)
- [ ] Training code includes MLflow logging

After training:

- [ ] Check MLflow UI for new run
- [ ] Verify metrics logged
- [ ] Download model from Kaggle
- [ ] Integrate with integration script
- [ ] Model linked to MLflow run

---

Now your Kaggle training is fully tracked in MLflow! 🎉
