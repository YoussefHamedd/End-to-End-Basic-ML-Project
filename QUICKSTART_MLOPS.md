# 🚀 MLOps Pipeline - Quick Start Guide

Your complete MLOps infrastructure is ready! Here's how to use it:

## 📦 What's Set Up

✅ **DVC** - Data & model versioning
✅ **MLflow** - Experiment tracking
✅ **Git** - Code versioning
✅ **Docker** - Containerization
✅ **GitHub Actions** - CI/CD automation
✅ **Monitoring** - Prometheus + Grafana

## 🎯 The MLOps Cycle

```
1. New Data → 2. Train Model → 3. Track in MLflow → 4. Version with DVC → 5. Commit to Git
```

---

## ⚡ Quick Commands

### Run Complete Pipeline

```bash
# Activate environment
source venv/bin/activate

# Run the full pipeline
python run_pipeline.py
```

This does:
- ✅ Data ingestion
- ✅ Data transformation
- ✅ Model training with hyperparameter tuning
- ✅ MLflow experiment tracking
- ✅ Saves artifacts to `artifacts/`

### Using Docker (with MLflow UI)

```bash
# Start everything (MLflow, Prometheus, Grafana, App)
docker-compose up -d

# In another terminal, run training
docker exec ml_application python run_pipeline.py

# View experiments
open http://localhost:5000
```

---

## 📝 Step-by-Step: Train a New Model

### 1. Start MLflow Server (Optional - if not using Docker)

```bash
# Terminal 1
source venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
```

### 2. Run Training Pipeline

```bash
# Terminal 2
source venv/bin/activate
python run_pipeline.py
```

You'll see:
```
🚀🚀🚀 ... COMPLETE MLOPS PIPELINE RUNNER ...

[1/3] Starting Data Ingestion...
✓ Data Ingestion Complete

[2/3] Starting Data Transformation...
✓ Data Transformation Complete

[3/3] Starting Model Training (with MLflow tracking)...
✓ Model Training Complete
  - Best Model R2 Score: 0.8745

✅ Pipeline completed successfully!
```

### 3. View Results in MLflow

```bash
# Open MLflow UI
open http://localhost:5000
```

You can:
- View all training runs
- Compare metrics (R2, MSE, MAE)
- See which model performed best
- Download model artifacts

### 4. Version Your Model with DVC

```bash
# Track artifacts with DVC
dvc add artifacts/model.pkl
dvc add artifacts/proprocessor.pkl

# Commit to git
git add artifacts/*.dvc .dvc/config
git commit -m "Train new model - R2: 0.8745"
git push
```

---

## 🔄 Working with New Data

### Scenario: You have a new dataset

```bash
# 1. Replace the data file
cp /path/to/new_data.csv data/stud.csv

# 2. Retrain the model
python run_pipeline.py

# 3. View experiments in MLflow
open http://localhost:5000

# 4. If better, version it
dvc add artifacts/model.pkl artifacts/proprocessor.pkl
git add artifacts/*.dvc
git commit -m "Retrain on new dataset"
git push
```

### Scenario: Modify model hyperparameters

```bash
# 1. Edit hyperparameters
nano src/components/model_trainer.py
# Modify the 'params' dictionary

# 2. Retrain
python run_pipeline.py

# 3. Compare in MLflow
# Go to http://localhost:5000 and compare runs

# 4. If improved, commit
git add src/components/model_trainer.py
git commit -m "Optimize hyperparameters"
dvc add artifacts/model.pkl
git add artifacts/model.pkl.dvc
git commit -m "Update model with optimized params"
git push
```

---

## 📊 MLflow Features

### View All Experiments

```bash
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000
```

### Compare Multiple Runs

1. Go to MLflow UI
2. Check boxes for runs you want to compare
3. Click "Compare"
4. See side-by-side metrics, parameters, artifacts

### Logged Information

For each run, MLflow tracks:
- **Parameters**: `train_samples`, `test_samples`, `n_features`, `best_model_name`
- **Metrics**: `r2_score`, `mse`, `mae` for all models
- **Artifacts**: Trained model, metrics.json
- **Code**: Git commit hash (if available)

---

## 🔧 Helper Scripts

```bash
# Test infrastructure
python test_pipeline.py

# Train model (with script)
bash scripts/train_model.sh

# Start MLflow server
bash scripts/start_mlflow.sh
```

---

## 🎓 Example Workflow

### Day 1: Initial Model

```bash
# Clone and setup
git clone https://github.com/YoussefHamedd/End-to-End-Basic-ML-Project.git
cd End-to-End-Basic-ML-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train initial model
python run_pipeline.py

# View results
mlflow ui --host 0.0.0.0 --port 5000  # Then open browser
```

### Day 2: Improve Model

```bash
# Try different hyperparameters
nano src/components/model_trainer.py

# Train again
python run_pipeline.py

# Compare in MLflow UI - which is better?
```

### Day 3: New Data Arrives

```bash
# Replace data
cp new_data.csv data/stud.csv

# Retrain
python run_pipeline.py

# Version everything
dvc add data/stud.csv artifacts/model.pkl
git add .
git commit -m "Update model with new data"
git push
```

---

## 🐳 Docker Workflow

```bash
# Start all services
docker-compose up -d

# Run training inside container
docker exec ml_application python run_pipeline.py

# View logs
docker-compose logs -f ml-app

# Access services:
# - ML App: http://localhost:8080
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000

# Stop all
docker-compose down
```

---

## 📈 Monitoring

### Prometheus Metrics

```bash
# Application metrics
curl http://localhost:8080/metrics

# View in Prometheus
open http://localhost:9090
```

Key metrics:
- `model_predictions_total` - Total predictions made
- `model_prediction_duration_seconds` - Prediction latency
- `model_errors_total` - Error count

### Grafana Dashboards

```bash
# Login: admin/admin
open http://localhost:3000

# Add Prometheus datasource: http://prometheus:9090
# Create dashboards for your metrics
```

---

## 🔍 Troubleshooting

### MLflow not connecting?

```bash
# Check if running
curl http://localhost:5000/health

# Restart
pkill -f mlflow
mlflow ui --host 0.0.0.0 --port 5000 &
```

### DVC issues?

```bash
# Check status
dvc status

# Reinitialize if needed
dvc repair
```

### Training fails?

```bash
# Check logs
cat logs/*.log

# Clean and retry
rm -rf artifacts/*
python run_pipeline.py
```

---

## 📚 File Structure

```
.
├── run_pipeline.py           # 🚀 Main pipeline runner
├── test_pipeline.py          # 🧪 Infrastructure test
├── scripts/
│   ├── train_model.sh        # Training helper script
│   └── start_mlflow.sh       # MLflow server starter
├── src/
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py  # With MLflow tracking!
│   └── Pipelines/            # Prediction pipeline
├── data/
│   └── stud.csv              # Your dataset (swap this for new data)
├── artifacts/                # Generated models & preprocessors
├── dvc.yaml                  # DVC pipeline definition
├── docker-compose.yml        # Multi-service setup
└── MLOPS_WORKFLOW.md         # Detailed workflow guide
```

---

## 🎯 Key Takeaways

1. **Train**: `python run_pipeline.py`
2. **Track**: Automatic in MLflow
3. **Version**: `dvc add artifacts/model.pkl`
4. **Commit**: `git commit -m "Update model"`
5. **Deploy**: Use Docker images

Your MLOps pipeline is production-ready! 🚀

For detailed workflow examples, see: **MLOPS_WORKFLOW.md**
