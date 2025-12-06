"""
Prefect Flow for Fake News Detection ML Pipeline
Orchestrates: Data Ingestion → Validation → Preparation → Training → MLflow Tracking
"""
import os
import sys
import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from datetime import timedelta

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import logging
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestionFakeNews
from src.components.data_validation import DataValidation
from src.components.data_preparation import DataPreparation
from src.components.model_trainer import ModelTrainerRoBERTa


@task(name="Data Ingestion", retries=2, retry_delay_seconds=5)
def ingest_data_task(data_path: str, sample_size: int = None):
    """
    Task: Ingest fake news data and split into train/test

    Args:
        data_path: Path to fake news CSV dataset
        sample_size: Optional sample size for testing

    Returns:
        Tuple of (train_path, test_path)
    """
    logging.info("🔄 Task: Data Ingestion started")

    data_ingestion = DataIngestionFakeNews()
    train_path, test_path = data_ingestion.initiate_data_ingestion(
        data_path=data_path,
        sample_size=sample_size
    )

    logging.info(f"✅ Task: Data Ingestion completed")
    logging.info(f"  - Train: {train_path}")
    logging.info(f"  - Test: {test_path}")

    return train_path, test_path


@task(name="Data Validation", retries=1)
def validate_data_task(train_path: str, test_path: str):
    """
    Task: Validate train and test data quality

    Args:
        train_path: Path to training data
        test_path: Path to test data

    Returns:
        Dict with validation results
    """
    logging.info("🔄 Task: Data Validation started")

    validator = DataValidation()

    # Validate train data
    train_df = pd.read_csv(train_path)
    train_results = validator.validate_all(train_df)

    # Validate test data
    test_df = pd.read_csv(test_path)
    test_results = validator.validate_all(test_df)

    if not train_results['is_valid'] or not test_results['is_valid']:
        raise ValueError(f"Data validation failed!\nTrain: {train_results['errors']}\nTest: {test_results['errors']}")

    logging.info("✅ Task: Data Validation passed")

    return {
        'train_valid': train_results['is_valid'],
        'test_valid': test_results['is_valid'],
        'train_samples': train_results['sample_count'],
        'test_samples': test_results['sample_count']
    }


@task(name="Data Preparation", retries=1)
def prepare_data_task(train_path: str, test_path: str):
    """
    Task: Clean and prepare text data

    Args:
        train_path: Path to training data
        test_path: Path to test data

    Returns:
        Tuple of (prepared_train_path, prepared_test_path)
    """
    logging.info("🔄 Task: Data Preparation started")

    prep = DataPreparation()

    # Prepare train data
    train_df = pd.read_csv(train_path)
    train_clean = prep.prepare_dataset(train_df)

    # Prepare test data
    test_df = pd.read_csv(test_path)
    test_clean = prep.prepare_dataset(test_df)

    # Save prepared data
    prepared_train_path = train_path.replace('.csv', '_prepared.csv')
    prepared_test_path = test_path.replace('.csv', '_prepared.csv')

    train_clean.to_csv(prepared_train_path, index=False)
    test_clean.to_csv(prepared_test_path, index=False)

    logging.info("✅ Task: Data Preparation completed")
    logging.info(f"  - Train samples: {len(train_clean)}")
    logging.info(f"  - Test samples: {len(test_clean)}")

    return prepared_train_path, prepared_test_path


@task(name="Model Training", retries=1, timeout_seconds=7200)  # 2 hour timeout
def train_model_task(
    train_path: str,
    test_path: str,
    model_name: str = 'roberta-base',
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    mlflow_uri: str = 'http://localhost:5000'
):
    """
    Task: Train RoBERTa model with MLflow tracking

    Args:
        train_path: Path to prepared training data
        test_path: Path to prepared test data
        model_name: HuggingFace model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        mlflow_uri: MLflow tracking URI

    Returns:
        F1 score on test set
    """
    logging.info("🔄 Task: Model Training started")

    trainer = ModelTrainerRoBERTa()
    f1_score = trainer.initiate_model_trainer(
        train_path=train_path,
        test_path=test_path,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name="fake_news_detection_prefect"
    )

    logging.info(f"✅ Task: Model Training completed - F1 Score: {f1_score:.4f}")

    return f1_score


@flow(
    name="Fake News Detection ML Pipeline",
    description="Complete MLOps pipeline with Prefect orchestration",
    task_runner=SequentialTaskRunner(),
    retries=1,
    retry_delay_seconds=10
)
def fake_news_ml_pipeline(
    data_path: str = 'data/fake_news.csv',
    sample_size: int = None,
    model_name: str = 'roberta-base',
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    mlflow_uri: str = 'http://localhost:5000'
):
    """
    Main Prefect Flow: Orchestrate the complete ML pipeline

    Args:
        data_path: Path to fake news CSV dataset
        sample_size: Limit dataset size (None = full dataset)
        model_name: HuggingFace model name (roberta-base, roberta-large)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        mlflow_uri: MLflow tracking URI

    Returns:
        f1_score: Final F1 score on test set
    """

    print("\n" + "="*70)
    print("  🚀 PREFECT FLOW: FAKE NEWS DETECTION ML PIPELINE")
    print("="*70)
    print(f"Dataset: {data_path}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"MLflow: {mlflow_uri}")
    if sample_size:
        print(f"Sample Size: {sample_size} (testing mode)")
    print("="*70 + "\n")

    # Task 1: Data Ingestion
    train_path, test_path = ingest_data_task(data_path, sample_size)

    # Task 2: Data Validation
    validation_results = validate_data_task(train_path, test_path)
    print(f"\n📊 Validation: {validation_results}")

    # Task 3: Data Preparation
    prepared_train_path, prepared_test_path = prepare_data_task(train_path, test_path)

    # Task 4: Model Training with MLflow
    f1_score = train_model_task(
        train_path=prepared_train_path,
        test_path=prepared_test_path,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlflow_uri=mlflow_uri
    )

    # Summary
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Model saved to: artifacts/roberta_fakenews/")
    print(f"\nNext steps:")
    print("  1. View experiments: mlflow ui --port 5000")
    print("  2. View Prefect UI: prefect server start")
    print("  3. Deploy API: python app.py")
    print("="*70 + "\n")

    return f1_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prefect Flow: Fake News Detection')
    parser.add_argument('--data', type=str, default='data/fake_news.csv',
                        help='Path to fake news dataset CSV')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for quick testing')
    parser.add_argument('--model', type=str, default='roberta-base',
                        choices=['roberta-base', 'roberta-large'],
                        help='Model to use')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5000',
                        help='MLflow tracking URI')

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Dataset not found at {args.data}")
        print("\nPlease download a fake news dataset and place it at data/fake_news.csv")
        sys.exit(1)

    # Run Prefect flow
    f1_score = fake_news_ml_pipeline(
        data_path=args.data,
        sample_size=args.sample,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mlflow_uri=args.mlflow_uri
    )

    print(f"\n✅ Final F1 Score: {f1_score:.4f}")
