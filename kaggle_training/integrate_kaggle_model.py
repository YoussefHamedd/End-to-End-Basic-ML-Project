#!/usr/bin/env python3
"""
Integrate Kaggle-Trained Model into Pipeline

This script helps you integrate a model trained on Kaggle back into your
local MLOps pipeline. It handles downloading, versioning with DVC, and
registering the model with MLflow.

Usage:
    python kaggle_training/integrate_kaggle_model.py --kaggle-model-path /path/to/downloaded/model
"""

import os
import sys
import json
import shutil
import argparse
import mlflow
import mlflow.pytorch
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Integrate Kaggle-trained model into pipeline"
    )
    parser.add_argument(
        "--kaggle-model-path",
        required=True,
        help="Path to downloaded Kaggle model directory",
    )
    parser.add_argument(
        "--model-name",
        default="roberta_student_performance",
        help="Name for the model in artifacts",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    return parser.parse_args()


def validate_kaggle_model(model_path):
    """Validate that the Kaggle model has all required files"""
    print(f"🔍 Validating Kaggle model at: {model_path}")

    required_files = [
        "pytorch_model.bin",  # or model.safetensors
        "config.json",
        "tokenizer_config.json",
        "metadata.json",
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            # Check for alternative names
            if file == "pytorch_model.bin" and os.path.exists(
                os.path.join(model_path, "model.safetensors")
            ):
                continue
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False

    print("✓ All required files present")
    return True


def load_metadata(model_path):
    """Load model metadata"""
    metadata_path = os.path.join(model_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def copy_model_to_artifacts(kaggle_model_path, model_name):
    """Copy Kaggle model to artifacts directory"""
    print(f"📦 Copying model to artifacts...")

    # Create artifacts directory for transformers
    artifacts_dir = "artifacts/transformers"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Destination path
    dest_path = os.path.join(artifacts_dir, model_name)

    # Remove existing if present
    if os.path.exists(dest_path):
        print(f"  Removing existing model at: {dest_path}")
        shutil.rmtree(dest_path)

    # Copy model directory
    shutil.copytree(kaggle_model_path, dest_path)

    print(f"✓ Model copied to: {dest_path}")
    return dest_path


def log_model_to_mlflow(model_path, metadata):
    """Log the model to MLflow"""
    print("📝 Logging model to MLflow...")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("kaggle_model_integration")

    with mlflow.start_run(run_name=f"integrate_{metadata.get('model_name', 'model')}"):
        # Log metadata as parameters
        for key, value in metadata.items():
            mlflow.log_param(key, value)

        # Log model location
        mlflow.log_param("model_location", model_path)
        mlflow.log_param("source", "kaggle")

        # Log model artifacts
        mlflow.log_artifacts(model_path, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"✓ Logged to MLflow run: {run_id}")

        return run_id


def register_model_in_registry(model_uri, model_name):
    """Register model in MLflow Model Registry"""
    print(f"🏷️  Registering model in MLflow Model Registry...")

    try:
        model_version = mlflow.register_model(model_uri, model_name)
        print(f"✓ Model registered: {model_name} (version {model_version.version})")
        return model_version
    except Exception as e:
        print(f"⚠️  Could not register model: {str(e)}")
        return None


def setup_dvc_tracking(model_path):
    """Set up DVC tracking for the model"""
    print("🔄 Setting up DVC tracking...")

    try:
        import subprocess

        # Add to DVC
        result = subprocess.run(
            ["dvc", "add", model_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print(f"✓ Model tracked with DVC")
            print(f"  DVC file: {model_path}.dvc")
            print(f"\nNext: git add {model_path}.dvc && git commit")
            return True
        else:
            print(f"⚠️  DVC tracking failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"⚠️  DVC not available: {str(e)}")
        return False


def create_model_card(model_path, metadata, run_id=None):
    """Create a model card with metadata"""
    print("📄 Creating model card...")

    model_card = f"""# {metadata.get('model_name', 'Model')}

## Model Information
- **Type**: {metadata.get('model_type', 'N/A')}
- **Base Model**: {metadata.get('model_name', 'N/A')}
- **Task**: {metadata.get('task', 'N/A')}
- **Target**: {metadata.get('target', 'N/A')}
- **Source**: Kaggle Training
- **Location**: `{model_path}`

## Training Configuration
- **Max Length**: {metadata.get('max_length', 'N/A')}

## Usage

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load model
model = RobertaForSequenceClassification.from_pretrained("{model_path}")
tokenizer = RobertaTokenizer.from_pretrained("{model_path}")

# Make prediction
inputs = tokenizer("Your input text", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.item()
```

## MLflow
- **Run ID**: {run_id or 'N/A'}
- **Tracking URI**: {os.getenv('MLFLOW_TRACKING_URI', 'N/A')}

## Integration Date
{os.popen('date').read().strip()}
"""

    card_path = os.path.join(model_path, "MODEL_CARD.md")
    with open(card_path, "w") as f:
        f.write(model_card)

    print(f"✓ Model card created: {card_path}")


def main():
    """Main integration workflow"""
    args = parse_args()

    print("\n" + "="*70)
    print("  🔗 Kaggle Model Integration")
    print("="*70 + "\n")

    # 1. Validate model
    if not validate_kaggle_model(args.kaggle_model_path):
        print("\n❌ Model validation failed. Please check the model directory.")
        return 1

    # 2. Load metadata
    metadata = load_metadata(args.kaggle_model_path)
    print(f"\n📋 Model Info:")
    print(f"  Type: {metadata.get('model_type')}")
    print(f"  Base: {metadata.get('model_name')}")
    print(f"  Task: {metadata.get('task')}")

    # 3. Copy to artifacts
    dest_path = copy_model_to_artifacts(args.kaggle_model_path, args.model_name)

    # 4. Log to MLflow
    run_id = log_model_to_mlflow(dest_path, metadata)

    # 5. Register in Model Registry (if requested)
    if args.register:
        model_uri = f"runs:/{run_id}/model"
        register_model_in_registry(model_uri, args.model_name)

    # 6. Set up DVC tracking
    setup_dvc_tracking(dest_path)

    # 7. Create model card
    create_model_card(dest_path, metadata, run_id)

    # Summary
    print("\n" + "="*70)
    print("  ✅ Integration Complete!")
    print("="*70)
    print(f"Model Location: {dest_path}")
    print(f"MLflow Run ID: {run_id}")
    print("\nNext Steps:")
    print("  1. Review model card: cat artifacts/transformers/{}/MODEL_CARD.md".format(args.model_name))
    print("  2. Commit DVC file: git add artifacts/transformers/{}.dvc".format(args.model_name))
    print("  3. Test predictions: Use the updated prediction pipeline")
    print("  4. Deploy: docker-compose up --build")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
