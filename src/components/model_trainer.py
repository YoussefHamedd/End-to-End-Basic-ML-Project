"""
Model Trainer for RoBERTa Fake News Detection
Trains transformer models with MLflow tracking
"""
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.exceptions import CustomException
from src.logger import logging

# MLflow imports
import mlflow
import mlflow.pytorch


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news text"""

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


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "roberta_fakenews")


class ModelTrainerRoBERTa:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
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

    def initiate_model_trainer(
        self,
        train_path,
        test_path,
        model_name="roberta-base",
        max_length=128,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
        mlflow_tracking_uri="http://localhost:5000",
        experiment_name="fake_news_detection"
    ):
        """
        Train RoBERTa model for fake news detection

        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            model_name: HuggingFace model name (default: roberta-base)
            max_length: Max sequence length
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            mlflow_tracking_uri: MLflow server URI
            experiment_name: MLflow experiment name

        Returns:
            test_metrics: Dict with test performance metrics
        """
        try:
            logging.info("="*70)
            logging.info("Starting RoBERTa Model Training")
            logging.info("="*70)

            # Load data
            logging.info(f"Loading data from {train_path} and {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_texts = train_df['text'].tolist()
            train_labels = train_df['label'].tolist()
            test_texts = test_df['text'].tolist()
            test_labels = test_df['label'].tolist()

            logging.info(f"Train samples: {len(train_texts)}")
            logging.info(f"Test samples: {len(test_texts)}")

            # Initialize tokenizer and model
            logging.info(f"Loading {model_name}...")
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )

            # Create datasets
            logging.info("Creating datasets...")
            train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, max_length)
            test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer, max_length)

            # Training arguments
            output_dir = self.model_trainer_config.trained_model_path
            os.makedirs(output_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=1,
                report_to="none",
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=self.compute_metrics,
            )

            # MLflow tracking
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{model_name}_epochs_{epochs}"):
                logging.info("🚀 Starting training...")

                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("task", "fake_news_detection")
                mlflow.log_param("max_length", max_length)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("train_samples", len(train_texts))
                mlflow.log_param("test_samples", len(test_texts))

                # Train
                train_result = trainer.train()
                logging.info("✅ Training completed!")

                # Evaluate on test set
                logging.info("📊 Evaluating on test set...")
                test_results = trainer.evaluate(test_dataset)

                # Log metrics
                mlflow.log_metric("accuracy", test_results['eval_accuracy'])
                mlflow.log_metric("precision", test_results['eval_precision'])
                mlflow.log_metric("recall", test_results['eval_recall'])
                mlflow.log_metric("f1_score", test_results['eval_f1'])
                mlflow.log_metric("test_loss", test_results['eval_loss'])

                # Log model
                logging.info("Logging model to MLflow...")
                mlflow.pytorch.log_model(model, "model")

                mlflow_run_id = mlflow.active_run().info.run_id
                logging.info(f"✓ MLflow Run ID: {mlflow_run_id}")

            # Save model locally
            logging.info(f"💾 Saving model to {output_dir}...")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save metadata
            metadata = {
                "model_type": "roberta",
                "model_name": model_name,
                "task": "fake_news_detection",
                "num_labels": 2,
                "label_names": ["Real News", "Fake News"],
                "max_length": max_length,
                "train_samples": len(train_texts),
                "test_samples": len(test_texts),
                "metrics": {
                    "accuracy": float(test_results['eval_accuracy']),
                    "precision": float(test_results['eval_precision']),
                    "recall": float(test_results['eval_recall']),
                    "f1": float(test_results['eval_f1']),
                },
                "mlflow_run_id": mlflow_run_id,
                "mlflow_tracking_uri": mlflow_tracking_uri
            }

            with open(f"{output_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save label mapping
            label_map = {"0": "Real News", "1": "Fake News"}
            with open(f"{output_dir}/label_map.json", "w") as f:
                json.dump(label_map, f, indent=2)

            logging.info("="*70)
            logging.info("✅ Training Pipeline Complete!")
            logging.info("="*70)
            logging.info(f"Model: {model_name}")
            logging.info(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
            logging.info(f"Precision: {test_results['eval_precision']:.4f}")
            logging.info(f"Recall:    {test_results['eval_recall']:.4f}")
            logging.info(f"F1 Score:  {test_results['eval_f1']:.4f}")
            logging.info(f"Model saved: {output_dir}")
            logging.info("="*70)

            return test_results['eval_f1']  # Return F1 as main metric

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion_fakenews import DataIngestionFakeNews

    # Test the pipeline
    print("Testing Fake News Training Pipeline...")

    # 1. Ingest data
    ingestion = DataIngestionFakeNews()
    train_data, test_data = ingestion.initiate_data_ingestion(
        data_path='data/fake_news.csv',
        sample_size=500  # Small sample for testing
    )

    # 2. Train model
    trainer = ModelTrainerRoBERTa()
    f1_score = trainer.initiate_model_trainer(
        train_path=train_data,
        test_path=test_data,
        model_name="roberta-base",
        epochs=1,  # Quick test
        batch_size=8
    )

    print(f"\n✅ Training completed! F1 Score: {f1_score:.4f}")
