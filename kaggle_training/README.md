# 🚀 Kaggle Training Integration

This directory contains everything you need to train models on Kaggle and integrate them into your MLOps pipeline.

## 📁 Files

- **`train_roberta_kaggle.py`** - Complete training script for Kaggle notebooks (RoBERTa 355M)
- **`integrate_kaggle_model.py`** - Integration script to bring Kaggle models into your pipeline
- **`README.md`** - This file

## ⚡ Quick Start

### 1. Train on Kaggle

```python
# In Kaggle notebook:
# 1. Upload train_roberta_kaggle.py
# 2. Install dependencies
!pip install -q transformers datasets mlflow torch

# 3. Run training
from train_roberta_kaggle import main
main()

# 4. Download the output folder (roberta_model)
```

### 2. Integrate into Pipeline

```bash
# Back on your local machine:
cd End-to-End-Basic-ML-Project

# Integrate the downloaded model
python kaggle_training/integrate_kaggle_model.py \
    --kaggle-model-path ~/Downloads/roberta_model \
    --model-name my_roberta_model \
    --register

# The model is now in artifacts/transformers/ and ready to use!
```

### 3. Use in Predictions

```python
from src.Pipelines.predict_pipeline_transformer import UnifiedPredictPipeline

# Auto-detects and uses your new transformer model!
pipeline = UnifiedPredictPipeline(model_type="auto")
predictions = pipeline.predict(your_data)
```

## 📚 Full Documentation

See **[KAGGLE_INTEGRATION.md](../KAGGLE_INTEGRATION.md)** for:
- Detailed step-by-step guide
- MLflow integration from Kaggle
- DVC versioning for large models
- Custom architectures
- Troubleshooting
- Best practices

## 🎯 Why Train on Kaggle?

- ✅ Free GPU access (P100, T4)
- ✅ 30+ hours/week of compute
- ✅ Pre-installed ML libraries
- ✅ Perfect for large models like RoBERTa

## 🔄 Workflow Summary

```
Kaggle (Train) → Download → Integrate → DVC Version → Git Commit → Deploy
```

## 💡 Tips

1. **Use ngrok** to connect Kaggle to your local MLflow server
2. **Version large models** with DVC (not Git)
3. **Test locally** before deploying
4. **Document everything** - model cards are auto-generated

## 🆘 Need Help?

Check the main documentation: [KAGGLE_INTEGRATION.md](../KAGGLE_INTEGRATION.md)
