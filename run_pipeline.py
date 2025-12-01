#!/usr/bin/env python3
"""
Complete MLOps Pipeline Runner

This script runs the full ML pipeline:
1. Data Ingestion
2. Data Transformation
3. Model Training (with MLflow tracking)
4. Artifacts saved and versioned with DVC

Usage:
    python run_pipeline.py
"""

import os
import sys
import mlflow
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exceptions import CustomException


def run_complete_pipeline():
    """Run the complete ML pipeline end-to-end"""

    try:
        logging.info("="*70)
        logging.info("Starting Complete MLOps Pipeline")
        logging.info("="*70)

        # Set MLflow tracking URI (use environment variable or default)
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        logging.info(f"MLflow Tracking URI: {mlflow_uri}")

        # Step 1: Data Ingestion
        logging.info("\n[1/3] Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"✓ Data Ingestion Complete")
        logging.info(f"  - Train data: {train_data_path}")
        logging.info(f"  - Test data: {test_data_path}")

        # Step 2: Data Transformation
        logging.info("\n[2/3] Starting Data Transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"✓ Data Transformation Complete")
        logging.info(f"  - Preprocessor saved: {preprocessor_path}")

        # Step 3: Model Training with MLflow
        logging.info("\n[3/3] Starting Model Training (with MLflow tracking)...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"✓ Model Training Complete")
        logging.info(f"  - Best Model R2 Score: {r2_score:.4f}")

        # Summary
        logging.info("\n" + "="*70)
        logging.info("Pipeline Execution Summary")
        logging.info("="*70)
        logging.info(f"✓ Data Ingestion: Success")
        logging.info(f"✓ Data Transformation: Success")
        logging.info(f"✓ Model Training: Success (R2={r2_score:.4f})")
        logging.info(f"✓ MLflow Tracking: Enabled")
        logging.info(f"✓ Artifacts saved in: artifacts/")
        logging.info("\nNext Steps:")
        logging.info("  1. View experiments: MLflow UI at http://localhost:5000")
        logging.info("  2. Version artifacts: dvc add artifacts/")
        logging.info("  3. Commit changes: git add . && git commit -m 'Update model'")
        logging.info("="*70 + "\n")

        return r2_score

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n" + "🚀 " * 30)
    print("  COMPLETE MLOPS PIPELINE RUNNER")
    print("🚀 " * 30 + "\n")

    try:
        score = run_complete_pipeline()
        print(f"\n✅ Pipeline completed successfully! Final R2 Score: {score:.4f}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}\n")
        sys.exit(1)
