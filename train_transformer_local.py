"""
Script d'entraînement local pour transformers (optionnel)
ATTENTION: Ce script est OPTIONNEL et NON requis pour le projet académique.
Le projet académique se concentre sur scikit-learn dans run_pipeline.py
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
import mlflow
import mlflow.pytorch

# ==================== Configuration ====================
MODEL_NAME = "roberta-base"  # Utiliser base au lieu de large (plus léger)
OUTPUT_DIR = "artifacts/transformers/roberta_fakenews"
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
BATCH_SIZE = 8  # Réduire si mémoire insuffisante
EPOCHS = 1  # Pour test rapide
SAMPLE_SIZE = 1000  # Sous-échantillon pour test rapide

# MLflow Configuration LOCAL
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "local_fake_news_detection"

os.environ["WANDB_DISABLED"] = "true"

# ==================== Load and Prepare Data ====================
print("📊 Loading dataset...")

# IMPORTANT: Adapte le chemin vers TON dataset local
# Option 1: CSV local
data_path = "data/fake_news_dataset.csv"  # Change ce chemin!

if not os.path.exists(data_path):
    print(f"❌ Dataset not found at {data_path}")
    print("Please download a fake news dataset or use a different dataset.")
    print("\nAlternative: Use a public dataset from HuggingFace:")
    print("from datasets import load_dataset")
    print("dataset = load_dataset('GonzaloA/fake_news')")
    exit(1)

df = pd.read_csv(data_path)

# Adapter les noms de colonnes selon ton dataset
# df = df.rename(columns={
#     "title": "title",
#     "text": "text",
#     "label": "label"
# })

# Sous-échantillon pour test rapide
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")

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
df["combined"] = df["text_clean"]

# Filter
df = df[df["text_clean"].str.len() > 10]
df = df.dropna(subset=["label"])
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
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
    num_labels=2
)

# ==================== Create Datasets ====================
print("📦 Creating datasets...")
train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

# ==================== Metrics ====================
def compute_metrics(eval_pred):
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
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_total_limit=1,
    report_to="none",
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
print("  🚀 Starting Training (LOCAL)")
print("="*70)

train_result = trainer.train()
print("\n✅ Training completed!")

# ==================== Evaluate ====================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)

print("\nTest Results:")
print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall: {test_results['eval_recall']:.4f}")
print(f"  F1 Score: {test_results['eval_f1']:.4f}")

# ==================== MLflow Tracking ====================
print("\n📊 Logging to MLflow (LOCAL)...")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"roberta_local_e{EPOCHS}"):
        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("task", "fake_news_detection")
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("train_samples", len(train_texts))

        # Log metrics
        mlflow.log_metric("accuracy", test_results['eval_accuracy'])
        mlflow.log_metric("precision", test_results['eval_precision'])
        mlflow.log_metric("recall", test_results['eval_recall'])
        mlflow.log_metric("f1_score", test_results['eval_f1'])

        mlflow_run_id = mlflow.active_run().info.run_id
        print(f"✓ MLflow Run ID: {mlflow_run_id}")

except Exception as e:
    print(f"⚠️  MLflow tracking failed: {str(e)}")
    mlflow_run_id = None

# ==================== Save Model ====================
print("\n💾 Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
metadata = {
    "model_type": "roberta",
    "model_name": MODEL_NAME,
    "task": "fake_news_detection",
    "max_length": MAX_LENGTH,
    "train_samples": len(train_texts),
    "metrics": {
        "accuracy": float(test_results['eval_accuracy']),
        "precision": float(test_results['eval_precision']),
        "recall": float(test_results['eval_recall']),
        "f1": float(test_results['eval_f1']),
    },
    "mlflow_run_id": mlflow_run_id,
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved to: {OUTPUT_DIR}")

# ==================== Summary ====================
print("\n" + "="*70)
print("  ✅ Training Complete (LOCAL)!")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"\nTest Performance:")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"\nModel saved: {OUTPUT_DIR}")
print(f"View in MLflow: http://localhost:5000")
print("="*70 + "\n")
