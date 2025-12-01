"""
Train RoBERTa on Kaggle with MLflow Tracking

This script is designed to run on Kaggle notebooks and log experiments
to your MLflow server. It saves the model so you can integrate it back
into your main pipeline.

Setup in Kaggle:
1. Install required packages
2. Set MLflow tracking URI (your server or ngrok tunnel)
3. Train model
4. Model is saved to Kaggle output and logged to MLflow
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ==================== Configuration ====================

# Set these in Kaggle notebook
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "roberta_student_performance"
MODEL_NAME = "roberta-base"  # or "roberta-large"
OUTPUT_DIR = "/kaggle/working/roberta_model"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


# ==================== Data Preparation ====================

def prepare_data(data_path="/kaggle/input/student-performance/stud.csv"):
    """Load and prepare data for RoBERTa"""
    print("📊 Loading data...")
    df = pd.read_csv(data_path)

    # Create text representation of features
    def create_text_input(row):
        text = f"""Student profile:
Gender: {row['gender']},
Ethnicity: {row['race/ethnicity']},
Parent Education: {row['parental level of education']},
Lunch: {row['lunch']},
Test Prep: {row['test preparation course']},
Reading Score: {row['reading score']},
Writing Score: {row['writing score']}"""
        return text

    df["text"] = df.apply(create_text_input, axis=1)
    df["label"] = df["math score"]  # Target variable

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"✓ Train samples: {len(train_df)}")
    print(f"✓ Test samples: {len(test_df)}")

    return train_df, test_df


def tokenize_data(train_df, test_df):
    """Tokenize text data for RoBERTa"""
    print("🔤 Tokenizing data...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    print("✓ Tokenization complete")
    return train_dataset, test_dataset, tokenizer


# ==================== Model Training ====================

def compute_metrics(eval_pred):
    """Compute regression metrics"""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {"mse": mse, "mae": mae, "r2": r2}


def train_model(train_dataset, test_dataset):
    """Train RoBERTa model"""
    print(f"🤖 Initializing {MODEL_NAME}...")

    # Load model for regression (num_labels=1)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="r2",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        report_to="none",  # We'll use MLflow for logging
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training...")
    train_result = trainer.train()

    print("📊 Evaluating model...")
    eval_result = trainer.evaluate()

    return trainer, model, train_result, eval_result


# ==================== MLflow Integration ====================

def log_to_mlflow(trainer, model, tokenizer, train_result, eval_result, train_df, test_df):
    """Log experiment to MLflow"""
    print("📝 Logging to MLflow...")

    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"roberta_{MODEL_NAME}"):
        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))

        # Log metrics
        mlflow.log_metric("train_loss", train_result.training_loss)
        mlflow.log_metric("eval_loss", eval_result["eval_loss"])
        mlflow.log_metric("eval_mse", eval_result["eval_mse"])
        mlflow.log_metric("eval_mae", eval_result["eval_mae"])
        mlflow.log_metric("eval_r2", eval_result["eval_r2"])

        # Log model (PyTorch format)
        mlflow.pytorch.log_model(model, "model")

        # Save tokenizer separately
        tokenizer_path = f"{OUTPUT_DIR}/tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")

        # Log training arguments
        mlflow.log_dict(training_args.to_dict(), "training_args.json")

        print(f"✓ Logged to MLflow run: {mlflow.active_run().info.run_id}")

    return mlflow.active_run().info.run_id


# ==================== Save Model ====================

def save_model_for_download(trainer, tokenizer):
    """Save model in format ready for download"""
    print("💾 Saving model for download...")

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metadata
    metadata = {
        "model_type": "roberta",
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "task": "regression",
        "target": "math_score",
    }

    import json

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("📦 Download this folder from Kaggle output!")


# ==================== Main Pipeline ====================

def main():
    """Run complete training pipeline"""
    print("\n" + "="*70)
    print("  🚀 RoBERTa Training Pipeline (Kaggle)")
    print("="*70 + "\n")

    try:
        # 1. Prepare data
        train_df, test_df = prepare_data()

        # 2. Tokenize
        train_dataset, test_dataset, tokenizer = tokenize_data(train_df, test_df)

        # 3. Train model
        trainer, model, train_result, eval_result = train_model(
            train_dataset, test_dataset
        )

        # 4. Log to MLflow
        run_id = log_to_mlflow(
            trainer, model, tokenizer, train_result, eval_result, train_df, test_df
        )

        # 5. Save for download
        save_model_for_download(trainer, tokenizer)

        # Print summary
        print("\n" + "="*70)
        print("  ✅ Training Complete!")
        print("="*70)
        print(f"R2 Score: {eval_result['eval_r2']:.4f}")
        print(f"MAE: {eval_result['eval_mae']:.4f}")
        print(f"MSE: {eval_result['eval_mse']:.4f}")
        print(f"\nMLflow Run ID: {run_id}")
        print(f"Model saved to: {OUTPUT_DIR}")
        print("\n📥 Download the model folder from Kaggle output!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise


# ==================== Kaggle Notebook Setup ====================

"""
# Kaggle Notebook Setup Instructions

## 1. Install packages
!pip install -q transformers datasets mlflow torch

## 2. Import and configure
import os

# Option A: Use ngrok to expose your local MLflow (recommended)
# Run locally: ngrok http 5000
# Then use the ngrok URL
os.environ["MLFLOW_TRACKING_URI"] = "https://your-ngrok-url.ngrok.io"

# Option B: Use a deployed MLflow server
# os.environ["MLFLOW_TRACKING_URI"] = "https://your-mlflow-server.com"

# Option C: Skip MLflow tracking (just train and save)
# os.environ["MLFLOW_TRACKING_URI"] = "file:///kaggle/working/mlruns"

## 3. Run training
main()

## 4. Download output
# Go to Kaggle output section and download the roberta_model folder
# It will contain: pytorch_model.bin, config.json, tokenizer files, metadata.json
"""


if __name__ == "__main__":
    main()
