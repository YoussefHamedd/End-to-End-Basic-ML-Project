# Prefect Orchestration Setup Guide

**Prefect** is a modern workflow orchestration tool that works natively on **Windows, Mac, and Linux**. Unlike Airflow, Prefect is Python-native and doesn't require complex setup.

---

## 🎯 Why Prefect?

✅ **Windows Compatible** - Works natively on Windows (unlike Airflow)
✅ **Simple Setup** - Just `pip install prefect`
✅ **Modern Python** - Clean, Pythonic API with decorators
✅ **Built-in UI** - Beautiful web interface for monitoring
✅ **Cloud or Local** - Run locally or use Prefect Cloud (free tier)
✅ **Retry Logic** - Automatic retries on task failures
✅ **Better Error Handling** - Clear error messages and debugging

---

## ⚡ Quick Start (5 minutes)

### 1. Install Prefect

```bash
# Already in requirements.txt
pip install prefect>=2.14.0

# Or install standalone
pip install prefect
```

### 2. Start Prefect Server (Optional - for UI)

```bash
# Terminal 1: Start Prefect server
prefect server start

# Opens UI at: http://localhost:4200
```

This gives you a beautiful dashboard to monitor your workflows!

### 3. Run Your ML Pipeline with Prefect

```bash
# Terminal 2: Run the pipeline
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# Or run full training
python prefect_flows/ml_pipeline_flow.py --epochs 3
```

**That's it!** Your pipeline runs with:
- ✅ Task orchestration
- ✅ Automatic retries
- ✅ Error handling
- ✅ Monitoring in UI
- ✅ MLflow tracking

---

## 🚀 Complete Usage

### Option 1: Run Directly (No UI needed)

```bash
# Quick test (1000 samples, 1 epoch) - 5 min
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# Full training
python prefect_flows/ml_pipeline_flow.py --epochs 3 --model roberta-base
```

The flow runs and logs everything to console. No UI needed!

### Option 2: Run with Prefect UI (Recommended)

```bash
# Terminal 1: Start Prefect server + MLflow
prefect server start
mlflow ui --port 5000

# Terminal 2: Run pipeline
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# View results:
# - Prefect UI: http://localhost:4200
# - MLflow UI: http://localhost:5000
```

### Option 3: Schedule Periodic Runs (Advanced)

```bash
# Create a deployment
prefect deployment build prefect_flows/ml_pipeline_flow.py:fake_news_ml_pipeline \
  --name "Daily Fake News Training" \
  --cron "0 2 * * *"  # Run at 2 AM daily

# Apply deployment
prefect deployment apply fake_news_ml_pipeline-deployment.yaml

# Start agent
prefect agent start -q default
```

---

## 📊 Pipeline Architecture

The Prefect flow orchestrates these tasks:

```
1. Data Ingestion
   ↓
2. Data Validation
   ↓
3. Data Preparation
   ↓
4. Model Training (RoBERTa + MLflow)
   ↓
   Final F1 Score
```

Each task has:
- **Retries**: Automatically retries on failure
- **Timeouts**: Prevents hanging tasks
- **Logging**: Detailed logs for debugging
- **Dependencies**: Ensures correct execution order

---

## 🛠️ Available Parameters

```bash
python prefect_flows/ml_pipeline_flow.py --help
```

**Parameters**:
- `--data`: Path to dataset (default: `data/fake_news.csv`)
- `--sample`: Sample size for testing (default: None = full dataset)
- `--model`: Model to use (`roberta-base` or `roberta-large`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--mlflow-uri`: MLflow tracking URI (default: `http://localhost:5000`)

**Examples**:

```bash
# Quick test
python prefect_flows/ml_pipeline_flow.py --sample 500 --epochs 1

# Medium training
python prefect_flows/ml_pipeline_flow.py --sample 2000 --epochs 2 --batch-size 16

# Full training with roberta-large
python prefect_flows/ml_pipeline_flow.py --model roberta-large --epochs 3
```

---

## 🎨 Prefect UI Features

When you run `prefect server start`, you get:

1. **Flow Runs**: See all your pipeline executions
2. **Task Runs**: Drill down into individual tasks
3. **Logs**: Real-time logs for each task
4. **Duration**: See how long each task takes
5. **Retry History**: Track failed tasks and retries
6. **State Changes**: Visual timeline of task states

Navigate to `http://localhost:4200` to explore!

---

## 🔄 Task Details

### Task 1: Data Ingestion
- **Retries**: 2 attempts with 5s delay
- **What it does**: Loads CSV, samples if needed, splits train/test
- **Output**: Paths to train.csv and test.csv

### Task 2: Data Validation
- **Retries**: 1 attempt
- **What it does**: Validates schema, quality, labels
- **Fails if**: Missing columns, invalid labels, too few samples

### Task 3: Data Preparation
- **Retries**: 1 attempt
- **What it does**: Cleans text, removes URLs, combines title+text
- **Output**: Prepared train/test CSV files

### Task 4: Model Training
- **Retries**: 1 attempt
- **Timeout**: 2 hours (7200 seconds)
- **What it does**: Trains RoBERTa, logs to MLflow
- **Output**: F1 score, saved model

---

## 🐛 Debugging Failed Runs

If a task fails:

1. **Check Prefect UI**: See detailed error logs
2. **Check MLflow**: View partial training metrics
3. **Check logs/**: Application logs directory
4. **Retry**: Prefect automatically retries failed tasks

Common issues:

```bash
# Issue: Dataset not found
# Fix: Place CSV at data/fake_news.csv

# Issue: Out of memory
# Fix: Reduce batch size or use sample
python prefect_flows/ml_pipeline_flow.py --sample 1000 --batch-size 4

# Issue: MLflow not running
# Fix: Start MLflow in another terminal
mlflow ui --port 5000
```

---

## 🆚 Prefect vs Airflow

| Feature | Prefect | Airflow |
|---------|---------|---------|
| **Windows** | ✅ Native | ❌ Requires WSL/Docker |
| **Setup** | `pip install prefect` | Complex (PostgreSQL, Redis) |
| **Python API** | Clean decorators | DAG objects |
| **UI** | Built-in, modern | Requires webserver |
| **Retry Logic** | Automatic | Manual configuration |
| **Best For** | Local + Cloud | Production Kubernetes |

**For this project**: Prefect is perfect because it's simple, Windows-compatible, and powerful enough for academic MLOps.

---

## 📖 Using with DVC and MLflow

Prefect works seamlessly with other tools:

```bash
# Terminal 1: Prefect server
prefect server start

# Terminal 2: MLflow server
mlflow ui --port 5000

# Terminal 3: Run pipeline
python prefect_flows/ml_pipeline_flow.py --epochs 3

# After training, version your model with DVC
dvc add artifacts/roberta_fakenews
git add artifacts/roberta_fakenews.dvc
git commit -m "Model v1 - Prefect orchestrated training"
```

---

## 🎓 Academic Project Integration

### Week 6: Orchestration (Checkpoint 2)

✅ **What you'll demonstrate**:
1. Prefect flow orchestrating complete pipeline
2. Task dependencies and error handling
3. Monitoring in Prefect UI
4. Integration with MLflow tracking

✅ **Deliverables**:
- `prefect_flows/ml_pipeline_flow.py` - Flow definition
- Screenshots of Prefect UI showing successful runs
- Documentation of retry logic and error handling

### Week 7-8: Production Ready

✅ **Next steps**:
- Schedule periodic retraining
- Deploy to Prefect Cloud (free)
- Add data drift detection
- Integrate with CI/CD

---

## 🌐 Prefect Cloud (Optional)

Want to monitor remotely?

```bash
# 1. Sign up (free): https://app.prefect.cloud

# 2. Login
prefect cloud login

# 3. Run your flow
python prefect_flows/ml_pipeline_flow.py

# View at: https://app.prefect.cloud
```

Free tier includes:
- 20,000 task runs/month
- Unlimited flows
- Cloud monitoring
- Team collaboration

---

## 🧪 Testing the Flow

Quick test to verify everything works:

```bash
# 1. Make sure you have a dataset
ls data/fake_news.csv

# 2. Start MLflow (optional but recommended)
mlflow ui --port 5000 &

# 3. Run quick test
python prefect_flows/ml_pipeline_flow.py --sample 500 --epochs 1

# Expected output:
# ✅ All tasks complete
# ✅ F1 Score printed
# ✅ Model saved to artifacts/
```

---

## 🔧 Troubleshooting

### Error: "No module named 'prefect'"
```bash
pip install prefect>=2.14.0
```

### Error: "Cannot connect to Prefect server"
```bash
# You don't need the server to run flows!
# Just run directly:
python prefect_flows/ml_pipeline_flow.py
```

### Error: "Task timeout"
```bash
# Increase timeout in ml_pipeline_flow.py:
@task(name="Model Training", timeout_seconds=14400)  # 4 hours
```

---

## 📚 Learn More

- **Prefect Docs**: https://docs.prefect.io/
- **Prefect Tutorials**: https://docs.prefect.io/tutorials/
- **Prefect Cloud**: https://app.prefect.cloud

---

## ✅ Quick Commands Summary

```bash
# Run pipeline without UI
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# Run with UI monitoring
prefect server start  # Terminal 1
python prefect_flows/ml_pipeline_flow.py --epochs 3  # Terminal 2

# View Prefect UI
# http://localhost:4200

# View MLflow experiments
mlflow ui --port 5000
# http://localhost:5000
```

**Simple, powerful, Windows-friendly orchestration!** 🎉
