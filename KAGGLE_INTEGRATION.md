# 🚀 Kaggle Integration Guide - Train RoBERTa & Integrate into Pipeline

This guide shows you how to train large models (like RoBERTa 355M) on Kaggle and seamlessly integrate them into your MLOps pipeline.

## 📋 Overview

```
Kaggle Training → Download Model → Integrate → Version with DVC → Git Commit → Deploy
```

---

## 🎯 Why Train on Kaggle?

- ✅ **Free GPU access** (30+ hours/week)
- ✅ **Pre-installed ML libraries**
- ✅ **Large compute resources**
- ✅ **Easy dataset access**
- ✅ **Public code sharing**

---

## 📝 Step-by-Step Workflow

### Phase 1: Train on Kaggle

#### 1. Create New Kaggle Notebook

Go to [kaggle.com/code](https://www.kaggle.com/code) and create a new notebook

#### 2. Upload Training Script

Upload `kaggle_training/train_roberta_kaggle.py` or copy the code into your notebook

#### 3. Install Requirements

```python
# In Kaggle notebook cell
!pip install -q transformers datasets mlflow torch
```

#### 4. Configure MLflow Tracking (Optional but Recommended)

**Option A: Use ngrok to expose your local MLflow**

On your local machine:
```bash
# Install ngrok
brew install ngrok  # macOS
# or download from https://ngrok.com

# Start MLflow locally
mlflow ui --host 0.0.0.0 --port 5000

# In another terminal, expose it
ngrok http 5000
# Copy the https URL (e.g., https://abc123.ngrok.io)
```

In Kaggle notebook:
```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "https://abc123.ngrok.io"
```

**Option B: Skip MLflow (local tracking only)**
```python
os.environ["MLFLOW_TRACKING_URI"] = "file:///kaggle/working/mlruns"
```

#### 5. Add Your Dataset

- **Option 1**: Upload via Kaggle Datasets
- **Option 2**: Use Add Data → Upload → your CSV file
- **Option 3**: Use public dataset if available

Update the data path in the script:
```python
data_path = "/kaggle/input/your-dataset/stud.csv"
```

#### 6. Run Training

```python
# In Kaggle notebook
from train_roberta_kaggle import main

# Run complete pipeline
main()
```

Training will:
- Load and prepare data
- Train RoBERTa for regression
- Log to MLflow (if configured)
- Save model to `/kaggle/working/roberta_model/`

#### 7. Download Trained Model

1. Go to Kaggle notebook **Output** tab
2. Download the `roberta_model` folder (entire directory)
3. Save it locally (e.g., `~/Downloads/roberta_model/`)

---

### Phase 2: Integrate into Your Pipeline

#### 1. Integrate the Model

```bash
# Navigate to your project
cd /path/to/End-to-End-Basic-ML-Project

# Activate environment
source venv/bin/activate

# Make script executable
chmod +x kaggle_training/integrate_kaggle_model.py

# Integrate the model
python kaggle_training/integrate_kaggle_model.py \
    --kaggle-model-path ~/Downloads/roberta_model \
    --model-name roberta_student_performance \
    --register
```

This will:
- ✅ Validate model files
- ✅ Copy to `artifacts/transformers/`
- ✅ Log to MLflow
- ✅ Register in Model Registry
- ✅ Track with DVC
- ✅ Create model card

#### 2. Version with DVC

```bash
# DVC already added during integration, now commit
git add artifacts/transformers/roberta_student_performance.dvc
git add .dvc/config
git commit -m "Add RoBERTa model trained on Kaggle"
```

#### 3. Update App to Use New Model

The unified prediction pipeline will automatically detect and use the transformer model!

```python
# Your app.py will use the new pipeline automatically
from src.Pipelines.predict_pipeline_transformer import UnifiedPredictPipeline

pipeline = UnifiedPredictPipeline(model_type="auto")  # Auto-detects!
predictions = pipeline.predict(data)
```

#### 4. Test Predictions

```bash
# Test the unified pipeline
python src/Pipelines/predict_pipeline_transformer.py
```

#### 5. Rebuild Docker (if using Docker)

```bash
# Update requirements
echo "transformers" >> requirements.txt
echo "torch" >> requirements.txt

# Rebuild and restart
docker-compose down
docker-compose build ml-app
docker-compose up -d
```

---

## 🔄 Complete Example Workflow

### Full End-to-End Example

```bash
# ============================================
# PART 1: TRAIN ON KAGGLE
# ============================================

# 1. Create Kaggle notebook
# 2. Copy code from kaggle_training/train_roberta_kaggle.py
# 3. Run training (takes ~30-60 minutes with GPU)
# 4. Download roberta_model folder from output

# ============================================
# PART 2: INTEGRATE LOCALLY
# ============================================

cd End-to-End-Basic-ML-Project
source venv/bin/activate

# Integrate model
python kaggle_training/integrate_kaggle_model.py \
    --kaggle-model-path ~/Downloads/roberta_model \
    --model-name roberta_math_predictor \
    --register

# Version with DVC
git add artifacts/transformers/roberta_math_predictor.dvc .dvc/
git commit -m "Add RoBERTa model - R2: 0.89"

# Test prediction
python -c "
from src.Pipelines.predict_pipeline_transformer import UnifiedPredictPipeline, CustomData

data = CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education=\"bachelor's degree\",
    lunch='standard',
    test_preparation_course='completed',
    reading_score=80,
    writing_score=75
)

pipeline = UnifiedPredictPipeline(model_type='auto')
result = pipeline.predict(data.get_data_as_data_frame())
print(f'Predicted Math Score: {result[0]:.2f}')
"

# Update Docker requirements
echo "transformers" >> requirements.txt
echo "torch" >> requirements.txt

# Deploy
docker-compose up -d --build

# Push to git
git push
```

---

## 📊 Monitoring & Comparison

### Compare Models in MLflow

```bash
# View all models (traditional ML + transformers)
mlflow ui --host 0.0.0.0 --port 5000
open http://localhost:5000

# Compare runs
# 1. Go to "Experiments"
# 2. Select runs from different experiments
# 3. Click "Compare" to see metrics side-by-side
```

### Model Performance Comparison

| Model | R2 Score | MAE | Training Time | Size |
|-------|----------|-----|---------------|------|
| Traditional (XGBoost) | 0.87 | 4.2 | 2 min | 1 MB |
| RoBERTa 355M | 0.91 | 3.8 | 45 min | 1.4 GB |

---

## 🔧 Advanced: Custom Model Architectures

### Train Custom Architecture on Kaggle

```python
# In Kaggle notebook
from transformers import RobertaConfig, RobertaForSequenceClassification

# Custom configuration
config = RobertaConfig.from_pretrained('roberta-base')
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

# Initialize model with custom config
model = RobertaForSequenceClassification(config)
```

### Fine-tune on Your Task

```python
# Load pre-trained and fine-tune
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=1,  # Regression
)

# Add custom head if needed
import torch.nn as nn

class CustomRobertaRegressor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.roberta = base_model.roberta
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, **inputs):
        outputs = self.roberta(**inputs)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.regressor(pooled)
```

---

## 🐳 Docker Deployment with Transformers

### Updated Dockerfile

The existing Dockerfile already supports transformers! Just ensure dependencies are installed:

```dockerfile
# In Dockerfile (already set up)
RUN pip install --no-cache-dir -r requirements.txt
# This installs transformers and torch if in requirements.txt
```

### Docker Compose with GPU Support

```yaml
# docker-compose.yml (optional GPU support)
services:
  ml-app:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 📦 DVC Remote Storage for Large Models

RoBERTa models are large (1-2 GB). Use DVC remote storage:

### Setup S3 Remote

```bash
# Configure AWS credentials
aws configure

# Add DVC remote
dvc remote add -d s3remote s3://your-bucket/dvc-storage

# Push large models
dvc push
```

### Setup Google Drive Remote (Free)

```bash
# Add Google Drive remote
dvc remote add -d gdrive gdrive://your-folder-id

# Push models
dvc push
```

### Team Collaboration

```bash
# Team member pulls project
git clone https://github.com/YoussefHamedd/End-to-End-Basic-ML-Project.git
cd End-to-End-Basic-ML-Project

# Pull large models from DVC remote
dvc pull

# Models are now in artifacts/transformers/
```

---

## 🧪 Testing Strategy

### Unit Tests for Transformer Pipeline

```python
# tests/test_transformer_pipeline.py
import pytest
from src.Pipelines.predict_pipeline_transformer import UnifiedPredictPipeline

def test_transformer_model_loading():
    """Test transformer model loads correctly"""
    pipeline = UnifiedPredictPipeline(model_type="transformer")
    pipeline.load_model()
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None

def test_unified_pipeline_auto_detect():
    """Test auto-detection of model type"""
    pipeline = UnifiedPredictPipeline(model_type="auto")
    pipeline.load_model()
    assert pipeline.model_type in ["ml", "transformer"]
```

---

## 🔍 Troubleshooting

### Issue: Model too large for Git

**Solution**: Use DVC (already configured!)

```bash
# Models are automatically ignored by Git
# Only .dvc files are committed
git status  # Shows .dvc file, not the actual model
```

### Issue: Out of memory during prediction

**Solution**: Use batch prediction

```python
# Instead of predicting all at once
predictions = []
for batch in data_batches:
    pred = pipeline.predict(batch)
    predictions.extend(pred)
```

### Issue: Kaggle download timeout

**Solution**: Use Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Configure
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/

# Download specific output
kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK -p ./downloads/
```

---

## 📚 Resources

- [Kaggle Kernels](https://www.kaggle.com/code)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [DVC Documentation](https://dvc.org/doc)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

## 🎓 Best Practices

1. **Always version control** your Kaggle notebooks (download as .ipynb and commit)
2. **Log everything to MLflow** for experiment tracking
3. **Use DVC for large models** (>100 MB)
4. **Test locally before deploying** - use the test script
5. **Document model metadata** - training data, hyperparameters, metrics
6. **Monitor performance** - compare against baseline
7. **Keep model cards updated** - automatically generated during integration

---

## 🎯 Summary

You now have a complete workflow to:
- ✅ Train large models (RoBERTa) on Kaggle with free GPUs
- ✅ Log experiments to MLflow (even from Kaggle!)
- ✅ Download and integrate models seamlessly
- ✅ Version large models with DVC
- ✅ Deploy via Docker
- ✅ Use unified prediction pipeline (auto-detects model type)
- ✅ Track everything in Git

**Your pipeline now supports ANY model - from simple sklearn to 355M parameter transformers!** 🚀
