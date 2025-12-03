# End-to-End Machine Learning Project

Complete MLOps pipeline with Kaggle integration, MLflow tracking, DVC versioning, and Docker deployment.

## Features

### Data Pipeline
• Structured modular data pipeline using separate Python scripts for ingestion, validation, transformation, and storage
• YAML-based config management and artifact tracking with DVC
• Support for both traditional ML and transformer models

### Model Architecture & Training
• Modular training loop built from scratch using scikit-learn
• Kaggle integration for training large transformer models (RoBERTa-large) with free GPUs
• Encapsulated models as services using OOP principles for reusability
• MLflow experiment tracking and model registry

### Deployment/Serving Setup
• End-to-end pipeline containerized via Docker
• Integrated CI/CD workflow using GitHub Actions (automated build, test, push stages)
• Flask API for predictions
• Support for multiple model types (sklearn, transformers)

### Monitoring/Alerting
• Unit test coverage for all modules
• GitHub Actions test logs + Docker layer caching + ML pipeline logging
• Prometheus/Grafana integration for runtime monitoring
• MLflow UI for experiment tracking

## Quick Start

### For Windows Users 🪟

See **[WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)** for complete step-by-step instructions.

Quick setup:
```powershell
# Run automated setup
.\setup_windows.ps1

# Start MLflow
python start_mlflow_for_ngrok.py

# In new terminal, start ngrok
ngrok http 5000
```

### For Mac/Linux Users 🐧

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Start MLflow
python start_mlflow_for_ngrok.py

# In new terminal, start ngrok
ngrok http 5000
```

## Documentation

- **[WINDOWS_QUICKSTART.md](WINDOWS_QUICKSTART.md)** - Complete Windows setup guide
- **[KAGGLE_INTEGRATION.md](KAGGLE_INTEGRATION.md)** - Train models on Kaggle with free GPUs
- **[MLFLOW_KAGGLE_SETUP.md](MLFLOW_KAGGLE_SETUP.md)** - MLflow tracking from Kaggle
- **[QUICKSTART_MLOPS.md](QUICKSTART_MLOPS.md)** - Quick MLOps workflow guide
- **[MLOPS_WORKFLOW.md](MLOPS_WORKFLOW.md)** - Detailed workflow examples

## MLOps Workflow

1. **Train on Kaggle** → Use free GPUs for large models
2. **Track with MLflow** → Monitor experiments via ngrok
3. **Download Model** → Get trained model from Kaggle
4. **Integrate** → Use junction/symlink to save disk space
5. **Version with DVC** → Track model versions
6. **Deploy** → Docker or Flask application

## Project Structure

```
.
├── app.py                          # Flask application
├── artifacts/                      # Model artifacts
│   └── transformers/              # Transformer models
├── kaggle_training/               # Kaggle integration
│   ├── train_roberta_fakenews.py  # Fake news detection training
│   └── integrate_kaggle_model.py  # Model integration script
├── src/
│   ├── components/                # Pipeline components
│   └── Pipelines/                 # Prediction pipelines
├── docker-compose.yml             # Multi-service deployment
├── Dockerfile                     # Container definition
├── start_mlflow_for_ngrok.py     # MLflow server with ngrok support
├── setup_windows.ps1              # Windows setup script
└── requirements.txt               # Python dependencies
```

## Technologies

- **Python 3.8+** - Core language
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data and model versioning
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **Flask** - Web API
- **Transformers** - HuggingFace models
- **scikit-learn** - Traditional ML models
- **Prometheus + Grafana** - Monitoring
- **ngrok** - Tunnel for Kaggle→local MLflow

## Use Cases

This project demonstrates:
- Training large models on Kaggle with free GPUs
- Real-time MLflow tracking from remote environments
- Efficient model storage with symbolic links/junctions
- Complete MLOps lifecycle (train → track → version → deploy)
- Multi-model support (sklearn + transformers)

## Contributing

Feel free to open issues or submit PRs!

## License

MIT License
