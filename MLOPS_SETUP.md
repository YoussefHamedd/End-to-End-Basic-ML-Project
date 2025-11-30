# MLOps Setup Guide

This project implements a complete MLOps pipeline for student performance prediction.

## 🎯 Features

### 1. **Experiment Tracking** (MLflow)
- Track all training experiments
- Log hyperparameters, metrics, and models
- Model registry for versioning
- Web UI at http://localhost:5000

### 2. **Data Versioning** (DVC)
- Version control for data and models
- Reproducible ML pipelines
- Track data lineage

### 3. **CI/CD Pipeline** (GitHub Actions)
- Automated testing on every push
- Code quality checks (Black, Flake8)
- Model training automation
- Docker image building and pushing
- Integration testing

### 4. **Containerization** (Docker)
- Reproducible environments
- Easy deployment
- Multi-container orchestration with Docker Compose

### 5. **Monitoring** (Prometheus + Grafana)
- Real-time metrics collection
- Model performance tracking
- Data drift detection
- Custom dashboards

### 6. **Testing**
- Unit tests for all components
- Model validation tests
- Integration tests
- Code coverage reports

## 🚀 Quick Start

### Local Development

1. **Set up virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the Flask app:**
```bash
python app.py
# Access at http://localhost:5000
```

3. **Run tests:**
```bash
pytest tests/ -v --cov=src
```

4. **Set up pre-commit hooks:**
```bash
pre-commit install
```

### Using Docker Compose (Recommended)

1. **Start all services:**
```bash
docker-compose up -d
```

This starts:
- **ML Application**: http://localhost:8080
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

2. **View logs:**
```bash
docker-compose logs -f ml-app
```

3. **Stop services:**
```bash
docker-compose down
```

## 📊 MLflow Usage

### Starting MLflow UI locally:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Training with MLflow tracking:
```python
from src.components.model_trainer import ModelTrainer

trainer = ModelTrainer()
# Training automatically logs to MLflow
trainer.initiate_model_trainer(train_array, test_array)
```

### View experiments:
- Navigate to http://localhost:5000
- Browse experiments, compare runs
- View metrics, parameters, and artifacts

## 🔄 DVC Pipeline

### Run the complete pipeline:
```bash
dvc repro
```

### View pipeline DAG:
```bash
dvc dag
```

### Track changes:
```bash
dvc status
dvc add artifacts/model.pkl
git add artifacts/model.pkl.dvc
git commit -m "Update model"
```

## 🧪 Testing

### Run all tests:
```bash
pytest tests/ -v
```

### Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### View coverage report:
```bash
open htmlcov/index.html
```

## 📈 Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:
- **prediction_counter**: Total predictions made
- **prediction_duration**: Time taken for predictions
- **error_counter**: Total errors encountered
- **feature_drift_score**: Data drift metrics

### Access Monitoring Tools:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Example Prometheus Queries:
```promql
# Prediction rate
rate(model_predictions_total[5m])

# Average prediction time
rate(model_prediction_duration_seconds_sum[5m]) /
rate(model_prediction_duration_seconds_count[5m])

# Error rate
rate(model_errors_total[5m])
```

## 🔧 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/mlops-pipeline.yml`) runs on every push:

1. **Code Quality**: Black formatting, Flake8 linting
2. **Unit Tests**: Run pytest with coverage
3. **Model Training**: Execute DVC pipeline (on main/develop)
4. **Docker Build**: Build and push Docker images
5. **Integration Tests**: Test with Docker Compose
6. **Deployment**: Deploy to production (configure as needed)

### Required GitHub Secrets:
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password/token

## 🏗️ Project Structure

```
.
├── .github/workflows/      # CI/CD pipelines
├── artifacts/              # Model artifacts and data
├── monitoring/             # Prometheus config
├── src/                    # Source code
│   ├── components/         # ML components
│   ├── Pipelines/          # Training and prediction pipelines
│   ├── monitoring.py       # Model monitoring
│   ├── logger.py           # Logging configuration
│   └── utils.py            # Utility functions
├── tests/                  # Test suite
├── templates/              # Flask templates
├── app.py                  # Flask application
├── dvc.yaml                # DVC pipeline definition
├── docker-compose.yml      # Multi-container setup
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── .pre-commit-config.yaml # Pre-commit hooks
```

## 🔐 Best Practices

1. **Always use virtual environments**
2. **Run tests before committing**: `pytest tests/`
3. **Use pre-commit hooks**: `pre-commit install`
4. **Track experiments with MLflow**
5. **Version data with DVC**
6. **Monitor model performance** in production
7. **Review CI/CD logs** for issues
8. **Keep Docker images updated**

## 🐛 Troubleshooting

### MLflow tracking issues:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Docker Compose issues:
```bash
docker-compose down -v  # Remove volumes
docker-compose up --build  # Rebuild images
```

### DVC issues:
```bash
dvc doctor  # Check DVC setup
```

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🤝 Contributing

1. Create a feature branch
2. Make changes
3. Run tests: `pytest tests/ -v`
4. Format code: `black src/ tests/`
5. Commit and push
6. Create pull request

## 📝 License

This project is for educational purposes.
