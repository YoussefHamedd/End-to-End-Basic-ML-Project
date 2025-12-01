#!/usr/bin/env python3
"""
Quick Pipeline Test Script

Tests the complete MLOps pipeline without full training.
Useful for verifying the infrastructure is working.
"""

import os
import sys
import mlflow
from src.logger import logging


def test_mlflow_connection():
    """Test MLflow tracking server connection"""
    try:
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)

        # Try to create a test experiment
        experiment_name = "pipeline_test"
        experiment = mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="infrastructure_test"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.99)

        logging.info(f"✓ MLflow connection successful: {mlflow_uri}")
        return True
    except Exception as e:
        logging.error(f"✗ MLflow connection failed: {str(e)}")
        return False


def test_data_availability():
    """Test if required data files exist"""
    try:
        data_path = "data/stud.csv"
        if os.path.exists(data_path):
            logging.info(f"✓ Data file found: {data_path}")
            return True
        else:
            logging.error(f"✗ Data file not found: {data_path}")
            return False
    except Exception as e:
        logging.error(f"✗ Data check failed: {str(e)}")
        return False


def test_artifacts_directory():
    """Test if artifacts directory is accessible"""
    try:
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        # Check if we can write to it
        test_file = os.path.join(artifacts_dir, ".test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)

        logging.info(f"✓ Artifacts directory accessible: {artifacts_dir}")
        return True
    except Exception as e:
        logging.error(f"✗ Artifacts directory check failed: {str(e)}")
        return False


def test_imports():
    """Test if all required modules can be imported"""
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        from src.components.model_trainer import ModelTrainer
        from src.Pipelines.predict_pipeline import PredictPipeline

        logging.info("✓ All required modules can be imported")
        return True
    except Exception as e:
        logging.error(f"✗ Import failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  MLOps Pipeline Infrastructure Test")
    print("="*70 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("Data Availability", test_data_availability),
        ("Artifacts Directory", test_artifacts_directory),
        ("MLflow Connection", test_mlflow_connection),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...", end=" ")
        result = test_func()
        results.append(result)
        print("✓" if result else "✗")

    print("\n" + "="*70)
    if all(results):
        print("✅ All tests passed! Pipeline infrastructure is ready.")
        print("\nYou can now run:")
        print("  python run_pipeline.py")
        print("="*70 + "\n")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
