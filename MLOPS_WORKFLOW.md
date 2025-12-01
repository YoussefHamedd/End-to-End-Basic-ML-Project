# MLOps Workflow Guide

This guide shows you how to use the complete MLOps pipeline for training, tracking, and versioning your ML models.

## 🎯 The Complete Cycle

```
New Data → Train Model → MLflow Tracking → Save Artifacts → DVC Version → Git Commit
```

## 📋 Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Using the Runner Script (Recommended)

```bash
# Run the complete pipeline in one command
python run_pipeline.py
```

This automatically:
- ✅ Ingests data from `data/stud.csv`
- ✅ Transforms and prepares data
- ✅ Trains models with hyperparameter tuning
- ✅ Logs everything to MLflow
- ✅ Saves artifacts to `artifacts/`

### Option 2: Using Helper Scripts

```bash
# 1. Start MLflow server (in one terminal)
bash scripts/start_mlflow.sh

# 2. Run training (in another terminal)
bash scripts/train_model.sh
```

### Option 3: Step-by-Step

```bash
# 1. Start MLflow server (background)
mlflow ui --host 0.0.0.0 --port 5000 &

# 2. Run pipeline
python run_pipeline.py

# 3. View experiments
open http://localhost:5000
```

## 🔄 Working with New Data

### Scenario: You have a new dataset

```bash
# 1. Replace your data file
cp /path/to/new_data.csv data/stud.csv

# 2. Track new data with DVC
dvc add data/stud.csv

# 3. Commit data changes
git add data/stud.csv.dvc data/.gitignore
git commit -m "Update dataset"

# 4. Train model with new data
python run_pipeline.py

# 5. Version the new model artifacts
dvc add artifacts/model.pkl
dvc add artifacts/proprocessor.pkl

# 6. Commit model changes
git add artifacts/*.dvc .dvc/config
git commit -m "Train model on new dataset"
git push
```

## 🧪 Experiment Tracking with MLflow

### View All Experiments

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Open browser: http://localhost:5000
```

### Compare Model Runs

1. Go to http://localhost:5000
2. Select multiple runs
3. Click "Compare"
4. View metrics, parameters, and artifacts side-by-side

### Register Best Model

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Register model
mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="student_performance_model"
)
```

## 📦 DVC Workflow

### Track Data Files

```bash
# Track your raw data
dvc add data/stud.csv

# This creates: data/stud.csv.dvc
git add data/stud.csv.dvc data/.gitignore
git commit -m "Track data with DVC"
```

### Track Model Artifacts

```bash
# Track model and preprocessor
dvc add artifacts/model.pkl
dvc add artifacts/proprocessor.pkl

# Commit the .dvc files
git add artifacts/*.dvc
git commit -m "Version model artifacts"
```

### Pull Data/Models (Team Collaboration)

```bash
# Pull latest data and models
dvc pull

# Your artifacts are now up to date!
```

### Push Data/Models to Remote

```bash
# Configure remote storage (S3, GCS, etc.)
dvc remote add -d mystorage s3://mybucket/dvcstore

# Push artifacts
dvc push

# Now team members can pull your artifacts!
```

## 🔄 Complete Workflow Example

### New Team Member Onboarding

```bash
# 1. Clone repository
git clone https://github.com/YoussefHamedd/End-to-End-Basic-ML-Project.git
cd End-to-End-Basic-ML-Project

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Pull data and models
dvc pull

# 4. Run the app
python app.py
```

### Making Model Improvements

```bash
# 1. Create experiment branch
git checkout -b experiment/improve-model

# 2. Modify hyperparameters in src/components/model_trainer.py
# Edit params dictionary

# 3. Train new model
python run_pipeline.py

# 4. Compare in MLflow UI
open http://localhost:5000

# 5. If better, version and commit
dvc add artifacts/model.pkl
git add artifacts/model.pkl.dvc src/components/model_trainer.py
git commit -m "Improve model performance"
git push origin experiment/improve-model

# 6. Create PR for review
```

## 📊 Monitoring Training

### Real-time Logs

```bash
# Watch logs during training
tail -f logs/*.log
```

### Check MLflow Runs Programmatically

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.get_experiment_by_name("student_performance_prediction")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
print(runs[['run_id', 'metrics.best_model_r2_score', 'params.best_model_name']])
```

## 🎯 DVC Pipeline Automation

### Run DVC Pipeline

```bash
# Execute the complete DVC pipeline
dvc repro

# View pipeline DAG
dvc dag
```

### Update Pipeline

Edit `dvc.yaml` to modify pipeline stages:

```yaml
stages:
  train:
    cmd: python run_pipeline.py
    deps:
      - data/stud.csv
      - src/components/
    outs:
      - artifacts/model.pkl
      - artifacts/proprocessor.pkl
    metrics:
      - metrics.json:
          cache: false
```

## 🔐 Best Practices

1. **Always version your data** with DVC before training
2. **Track all experiments** with MLflow
3. **Document hyperparameters** in git commits
4. **Version control artifacts** after successful training
5. **Compare models** before deploying
6. **Use branches** for experiments
7. **Tag production models** in MLflow Model Registry

## 🆘 Troubleshooting

### MLflow Connection Issues

```bash
# Check if MLflow server is running
curl http://localhost:5000/health

# Restart MLflow server
pkill -f mlflow
mlflow ui --host 0.0.0.0 --port 5000 &
```

### DVC Issues

```bash
# Check DVC status
dvc status

# Repair DVC
dvc repair

# Re-add files if needed
dvc add data/stud.csv --force
```

### Pipeline Failures

```bash
# Check logs
cat logs/*.log

# Run with verbose output
python run_pipeline.py --verbose

# Clean artifacts and retry
rm -rf artifacts/*
python run_pipeline.py
```

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLOps Best Practices](https://ml-ops.org/)

---

## 🎓 Example: Complete New Model Training

```bash
# Step-by-step example
echo "1. Prepare new data"
cp new_student_data.csv data/stud.csv

echo "2. Version data"
dvc add data/stud.csv
git add data/stud.csv.dvc
git commit -m "Add new dataset"

echo "3. Start MLflow"
mlflow ui --host 0.0.0.0 --port 5000 &

echo "4. Train model"
python run_pipeline.py

echo "5. Check results in MLflow UI"
open http://localhost:5000

echo "6. Version new model"
dvc add artifacts/model.pkl artifacts/proprocessor.pkl
git add artifacts/*.dvc
git commit -m "Train model v2.0 - improved R2 score"

echo "7. Push everything"
git push
dvc push  # if remote configured

echo "✅ Complete!"
```

That's the complete MLOps workflow! 🚀
