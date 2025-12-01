"""
Complete Kaggle Training Script for Fake News Detection with RoBERTa-Large

This script trains RoBERTa-large on the WELFake dataset and saves it for integration.
Includes MLflow tracking for experiment management.
"""

import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os

# ==================== Configuration ====================
MODEL_NAME = "roberta-large"
OUTPUT_DIR = "/kaggle/working/roberta_fakenews_model"
MAX_LENGTH = 256
BATCH_SIZE = 8  # Smaller batch size for roberta-large
EPOCHS = 3
LEARNING_RATE = 2e-5

# MLflow Configuration
# Option 1: Use ngrok URL (recommended for local MLflow)
# MLFLOW_TRACKING_URI = "https://your-ngrok-url.ngrok.io"
# Option 2: Use deployed MLflow server
# MLFLOW_TRACKING_URI = "https://your-mlflow-server.com"
# Option 3: Local file tracking (no server needed)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///kaggle/working/mlruns")
EXPERIMENT_NAME = "kaggle_fake_news_detection"

os.environ["WANDB_DISABLED"] = "true"

# ==================== Load and Prepare Data ====================
print("📊 Loading WELFake dataset...")
df = pd.read_csv("/kaggle/input/welfake-dataset/WELFake_Dataset.csv")

df = df.rename(columns={
    "title": "title",
    "text": "text",
    "label": "label"
})

print("Dataset shape:", df.shape)
print("Label distribution:\n", df.label.value_counts(normalize=True))

# ==================== Clean Text ====================
def clean_text(t):
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

print("🧹 Cleaning text...")
df["text_clean"] = df["text"].apply(clean_text)
df["title_clean"] = df["title"].apply(clean_text)

# Filter and combine
df = df[df["text_clean"].str.len() > 10]
df = df.dropna(subset=["label"])
df["combined"] = df["title_clean"] + " [SEP] " + df["text_clean"]

print(f"✓ Cleaned data: {len(df)} samples")

# ==================== Split Data ====================
print("🔀 Splitting data...")
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["combined"].tolist(),
    df["label"].tolist(),
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

print(f"✓ Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# ==================== Dataset Class ====================
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ==================== Initialize Model ====================
print(f"🤖 Loading {MODEL_NAME}...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # Binary classification
)

# ==================== Create Datasets ====================
print("📦 Creating datasets...")
train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

# ==================== Metrics ====================
def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==================== Training Arguments ====================
print("⚙️  Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    fp16=True,  # Mixed precision for faster training
)

# ==================== Trainer ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ==================== Train ====================
print("\n" + "="*70)
print("  🚀 Starting Training")
print("="*70)

train_result = trainer.train()

print("\n✅ Training completed!")

# ==================== Evaluate on Test Set ====================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)

print("\nTest Results:")
print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall: {test_results['eval_recall']:.4f}")
print(f"  F1 Score: {test_results['eval_f1']:.4f}")

# ==================== MLflow Tracking ====================
print("\n📊 Logging to MLflow...")

try:
    import mlflow
    import mlflow.pytorch

    # Set tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run(run_name=f"roberta_large_fakenews_e{EPOCHS}"):

        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("task", "fake_news_detection")
        mlflow.log_param("dataset", "WELFake")
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_samples", len(train_texts))
        mlflow.log_param("val_samples", len(val_texts))
        mlflow.log_param("test_samples", len(test_texts))

        # Log metrics
        mlflow.log_metric("accuracy", test_results['eval_accuracy'])
        mlflow.log_metric("precision", test_results['eval_precision'])
        mlflow.log_metric("recall", test_results['eval_recall'])
        mlflow.log_metric("f1_score", test_results['eval_f1'])
        mlflow.log_metric("test_loss", test_results['eval_loss'])

        # Log model (PyTorch format)
        mlflow.pytorch.log_model(model, "model")

        # Log training args as artifact
        training_args_dict = training_args.to_dict()
        mlflow.log_dict(training_args_dict, "training_args.json")

        mlflow_run_id = mlflow.active_run().info.run_id
        print(f"✓ MLflow Run ID: {mlflow_run_id}")
        print(f"✓ MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

except ImportError:
    print("⚠️  MLflow not installed. Skipping experiment tracking.")
    print("   Install with: !pip install mlflow")
    mlflow_run_id = None
except Exception as e:
    print(f"⚠️  MLflow tracking failed: {str(e)}")
    print("   Continuing without MLflow tracking...")
    mlflow_run_id = None

# ==================== Save Model ====================
print("\n💾 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
metadata = {
    "model_type": "roberta",
    "model_name": MODEL_NAME,
    "task": "fake_news_detection",
    "num_labels": 2,
    "label_names": ["Real News", "Fake News"],
    "max_length": MAX_LENGTH,
    "dataset": "WELFake",
    "train_samples": len(train_texts),
    "val_samples": len(val_texts),
    "test_samples": len(test_texts),
    "metrics": {
        "accuracy": float(test_results['eval_accuracy']),
        "precision": float(test_results['eval_precision']),
        "recall": float(test_results['eval_recall']),
        "f1": float(test_results['eval_f1']),
    },
    "mlflow_run_id": mlflow_run_id,
    "mlflow_tracking_uri": MLFLOW_TRACKING_URI
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Save label mapping
label_map = {"0": "Real News", "1": "Fake News"}
with open(f"{OUTPUT_DIR}/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"✓ Model saved to: {OUTPUT_DIR}")

# ==================== Summary ====================
print("\n" + "="*70)
print("  ✅ Training Pipeline Complete!")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Task: Binary Classification (Fake News Detection)")
print(f"\nTest Performance:")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"\n📥 Download: {OUTPUT_DIR}")
print("="*70 + "\n")
