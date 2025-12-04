# Guide Projet MLOps - 8 Semaines

Guide détaillé semaine par semaine pour réaliser votre projet MLOps.

---

## 📅 Semaine 1 : Cadrage et Préparation

### Objectifs
- Définir le problème métier
- Collecter les données
- Configurer l'environnement de développement
- Mettre en place le repository Git

### Actions

#### 1. Définition du projet
```markdown
- Problème métier : [Décrire le problème à résoudre]
- Type de ML : [Classification / Régression / Clustering]
- Métriques de succès : [R², Accuracy, F1-Score, etc.]
- Contraintes : [Performance, latence, etc.]
```

#### 2. Setup de l'environnement

```bash
# Créer le projet
mkdir mon-projet-mlops
cd mon-projet-mlops

# Initialiser Git
git init
git remote add origin <votre-repo-url>

# Créer la structure
mkdir -p src/{components,Pipelines} data artifacts tests logs Notebook

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate  # Windows

# Installer les outils de base
pip install pandas numpy scikit-learn matplotlib seaborn
pip install mlflow dvc pytest flask

# Créer requirements.txt
pip freeze > requirements.txt
```

#### 3. Configuration Git

Créer `.gitignore`:
```
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
artifacts/
logs/*.log
.env
mlruns/
.dvc/cache/
```

#### 4. Initialiser DVC

```bash
# Initialiser DVC pour le versioning des données
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

### Livrables semaine 1
- [ ] Repository Git structuré
- [ ] Environnement Python fonctionnel
- [ ] DVC initialisé
- [ ] README.md avec description du projet
- [ ] Données collectées dans `data/`

---

## 📊 Semaine 2 : Exploration et Feature Engineering

### Objectifs
- Analyse exploratoire des données (EDA)
- Feature engineering reproductible
- Versioning des données avec DVC
- Notebooks d'expérimentation

### Actions

#### 1. Analyse Exploratoire (EDA)

Créer `Notebook/01_EDA.ipynb`:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('../data/dataset.csv')

# Vue d'ensemble
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualisations
# - Distribution des variables
# - Corrélations
# - Outliers
# - Target distribution
```

#### 2. Feature Engineering

Créer `src/components/data_transformation.py`:
```python
"""
Transformation reproductible des données
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def transform(self, df):
        """Appliquer les transformations"""
        # Vos transformations ici
        pass

    def save_preprocessor(self, path):
        """Sauvegarder le preprocessor"""
        pass
```

#### 3. Versioning des données

```bash
# Ajouter les données à DVC
dvc add data/dataset.csv
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"

# Push vers le remote storage (optionnel)
# dvc remote add -d myremote /path/to/storage
# dvc push
```

#### 4. Validation des données

Créer `src/components/data_validation.py`:
```python
"""
Validation des données entrantes
"""
def validate_schema(df):
    """Valider le schéma des données"""
    required_columns = ['col1', 'col2', 'target']
    assert all(col in df.columns for col in required_columns)
    return True

def validate_quality(df):
    """Valider la qualité des données"""
    assert df.isnull().sum().sum() < len(df) * 0.1  # Max 10% missing
    return True
```

### Livrables semaine 2
- [ ] Notebook EDA complet avec visualisations
- [ ] Script de transformation reproductible
- [ ] Données versionnées avec DVC
- [ ] Script de validation des données
- [ ] Documentation des features créées

---

## 🤖 Semaine 3 : Développement du Modèle

### Objectifs
- Implémentation des premiers modèles
- MLflow pour le tracking
- Définition des métriques
- Validation croisée

### Actions

#### 1. Setup MLflow

```bash
# Créer le dossier mlruns
mkdir mlruns

# Démarrer MLflow UI (dans un terminal séparé)
mlflow ui --port 5000
```

Accéder à: http://localhost:5000

#### 2. Training avec MLflow

Créer `src/components/model_trainer.py`:
```python
"""
Formation de modèles avec tracking MLflow
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self, X_train, y_train, X_test, y_test):
        """Former le modèle avec tracking MLflow"""

        with mlflow.start_run(run_name="random_forest_v1"):
            # Log des paramètres
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

            # Formation
            model = RandomForestRegressor(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)

            # Prédictions
            y_pred = model.predict(X_test)

            # Métriques
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            # Sauvegarder le modèle
            mlflow.sklearn.log_model(model, "model")

            return model, r2
```

#### 3. Expérimentation

Créer `Notebook/02_Model_Experiments.ipynb`:
```python
# Tester plusieurs modèles
models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'LinearRegression': LinearRegression()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Former et logger
        pass
```

#### 4. Tests unitaires

Créer `tests/test_model.py`:
```python
import pytest
from src.components.model_trainer import ModelTrainer

def test_model_training():
    """Tester que le modèle se forme correctement"""
    # Données de test
    X_train, y_train = ...

    trainer = ModelTrainer(config={})
    model, score = trainer.train(X_train, y_train, X_test, y_test)

    assert model is not None
    assert score > 0.5  # Score minimum acceptable
```

### Livrables semaine 3
- [ ] Au moins 3 modèles différents testés
- [ ] Toutes les expériences trackées dans MLflow
- [ ] Métriques de validation définies
- [ ] Tests unitaires pour le training
- [ ] Notebook de comparaison des modèles

---

## 🔧 Semaine 4 : Pipeline de Formation

### Objectifs
- Pipeline de formation reproductible
- Containerisation Docker
- Automatisation du preprocessing
- Tests complets

### Actions

#### 1. Pipeline reproductible

Créer `run_pipeline.py`:
```python
"""
Pipeline complet de formation
"""
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    """Exécuter le pipeline complet"""

    # 1. Ingestion
    print("=== Data Ingestion ===")
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # 2. Transformation
    print("=== Data Transformation ===")
    transformation = DataTransformation()
    X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(
        train_data, test_data
    )

    # 3. Training
    print("=== Model Training ===")
    trainer = ModelTrainer()
    model, score = trainer.train(X_train, y_train, X_test, y_test)

    print(f"Training completed! R² Score: {score:.4f}")

    return model, score

if __name__ == "__main__":
    run_training_pipeline()
```

#### 2. Configuration DVC Pipeline

Créer/modifier `dvc.yaml`:
```yaml
stages:
  data_ingestion:
    cmd: python src/components/data_ingestion.py
    deps:
      - data/raw/dataset.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  data_transformation:
    cmd: python src/components/data_transformation.py
    deps:
      - data/processed/train.csv
      - data/processed/test.csv
    outs:
      - artifacts/preprocessor.pkl

  model_training:
    cmd: python src/components/model_trainer.py
    deps:
      - data/processed/train.csv
      - artifacts/preprocessor.pkl
    outs:
      - artifacts/model.pkl
    metrics:
      - metrics.json:
          cache: false
```

Lancer le pipeline:
```bash
dvc repro
```

#### 3. Dockerisation

Créer `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Installer le package en mode éditable
RUN pip install -e .

# Exposer le port
EXPOSE 8080

# Commande par défaut
CMD ["python", "app.py"]
```

Build et test:
```bash
docker build -t mlops-project .
docker run -p 8080:8080 mlops-project
```

#### 4. Tests complets

Créer `tests/test_pipeline.py`:
```python
def test_full_pipeline():
    """Tester le pipeline complet"""
    model, score = run_training_pipeline()
    assert model is not None
    assert score > 0.5
```

### Livrables semaine 4 (CHECKPOINT 1)
- [ ] Pipeline reproductible (`run_pipeline.py`)
- [ ] Pipeline DVC configuré
- [ ] Docker fonctionnel
- [ ] Tests unitaires couvrant tous les composants
- [ ] Documentation technique complète

---

## 🚀 Semaine 5 : API et Inférence

### Objectifs
- API de prédiction (Flask/FastAPI)
- Containerisation de l'API
- Tests d'intégration
- Documentation API

### Actions

#### 1. Pipeline de prédiction

Créer `src/Pipelines/predict_pipeline.py`:
```python
"""
Pipeline de prédiction
"""
import pickle
import pandas as pd

class PredictPipeline:
    def __init__(self):
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        with open('artifacts/model.pkl', 'rb') as f:
            return pickle.load(f)

    def load_preprocessor(self):
        with open('artifacts/preprocessor.pkl', 'rb') as f:
            return pickle.load(f)

    def predict(self, data):
        """Faire une prédiction"""
        # Prétraiter
        data_transformed = self.preprocessor.transform(data)

        # Prédire
        prediction = self.model.predict(data_transformed)

        return prediction
```

#### 2. API Flask

Créer `app.py`:
```python
"""
API Flask pour les prédictions
"""
from flask import Flask, request, jsonify
from src.Pipelines.predict_pipeline import PredictPipeline
import pandas as pd

app = Flask(__name__)
pipeline = PredictPipeline()

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données
        data = request.get_json()
        df = pd.DataFrame([data])

        # Prédire
        prediction = pipeline.predict(df)

        return jsonify({
            "prediction": float(prediction[0]),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

#### 3. Tests d'intégration

Créer `tests/test_app.py`:
```python
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test endpoint home"""
    response = client.get('/')
    assert response.status_code == 200

def test_predict(client):
    """Test endpoint predict"""
    data = {
        'feature1': 10,
        'feature2': 20,
        'feature3': 30
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert 'prediction' in response.json
```

#### 4. Documentation API

Créer `API_DOCUMENTATION.md`:
```markdown
# API Documentation

## Endpoints

### GET /
Informations sur l'API

### GET /health
Health check

### POST /predict
Faire une prédiction

**Request:**
\```json
{
  "feature1": 10,
  "feature2": 20,
  "feature3": 30
}
\```

**Response:**
\```json
{
  "prediction": 42.5,
  "status": "success"
}
\```
```

### Livrables semaine 5
- [ ] API Flask fonctionnelle
- [ ] Tests d'intégration passants
- [ ] Documentation API complète
- [ ] Docker pour l'API
- [ ] Exemple d'utilisation de l'API

---

## ⚙️ Semaine 6 : CI/CD et Déploiement

### Objectifs
- GitHub Actions pour CI/CD
- Déploiement automatisé
- Environnements dev/staging/prod
- Tests automatisés

### Actions

#### 1. GitHub Actions

Créer `.github/workflows/mlops-pipeline.yml`:
```yaml
name: MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src

      - name: Lint code
        run: |
          pip install flake8
          flake8 src/ --max-line-length=120

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: |
          docker build -t mlops-project:${{ github.sha }} .

      - name: Test Docker image
        run: |
          docker run -d -p 8080:8080 mlops-project:${{ github.sha }}
          sleep 5
          curl http://localhost:8080/health

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Commandes de déploiement
```

#### 2. Docker Compose pour multi-services

Créer `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    volumes:
      - ./artifacts:/app/artifacts

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlflow/mlruns

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

#### 3. Gestion des environnements

Créer `.env.example`:
```bash
# Development
ENVIRONMENT=development
MLFLOW_TRACKING_URI=http://localhost:5000
LOG_LEVEL=DEBUG

# Production
# ENVIRONMENT=production
# MLFLOW_TRACKING_URI=http://mlflow-prod:5000
# LOG_LEVEL=INFO
```

#### 4. Script de déploiement

Créer `scripts/deploy.sh`:
```bash
#!/bin/bash
# Script de déploiement

ENV=$1

if [ "$ENV" == "prod" ]; then
    echo "Deploying to PRODUCTION..."
    docker-compose -f docker-compose.prod.yml up -d
elif [ "$ENV" == "staging" ]; then
    echo "Deploying to STAGING..."
    docker-compose -f docker-compose.staging.yml up -d
else
    echo "Deploying to DEVELOPMENT..."
    docker-compose up -d
fi

echo "Deployment completed!"
```

### Livrables semaine 6 (CHECKPOINT 2)
- [ ] CI/CD fonctionnel (GitHub Actions)
- [ ] Docker Compose multi-services
- [ ] Scripts de déploiement automatisés
- [ ] Documentation de déploiement
- [ ] Tests passent automatiquement

---

## 📊 Semaine 7 : Monitoring et Logs

### Objectifs
- Monitoring des modèles
- Logging structuré
- Détection de drift
- Alertes

### Actions

#### 1. Monitoring avec Prometheus

Modifier `app.py` pour ajouter des métriques:
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Métriques
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
errors_counter = Counter('prediction_errors_total', 'Total prediction errors')

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        # ... code de prédiction ...

        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)

        return jsonify({"prediction": result})

    except Exception as e:
        errors_counter.inc()
        return jsonify({"error": str(e)}), 400
```

#### 2. Logging structuré

Créer `src/logger.py`:
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('logs/app.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_prediction(self, input_data, prediction, latency):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "prediction",
            "input": input_data,
            "prediction": prediction,
            "latency_ms": latency * 1000
        }
        self.logger.info(json.dumps(log_entry))
```

#### 3. Détection de drift

Créer `src/monitoring/drift_detector.py`:
```python
"""
Détection de drift des données
"""
from scipy.stats import ks_2samp
import pandas as pd

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    def detect_drift(self, new_data, threshold=0.05):
        """Détecter le drift avec test de Kolmogorov-Smirnov"""
        drift_detected = {}

        for column in self.reference_data.columns:
            statistic, p_value = ks_2samp(
                self.reference_data[column],
                new_data[column]
            )

            drift_detected[column] = p_value < threshold

        return drift_detected
```

#### 4. Alertes

Créer `src/monitoring/alerts.py`:
```python
"""
Système d'alertes
"""
def send_alert(metric, value, threshold):
    """Envoyer une alerte si seuil dépassé"""
    if value > threshold:
        # Envoyer email/Slack/etc.
        print(f"ALERT: {metric} = {value} > {threshold}")
```

### Livrables semaine 7
- [ ] Métriques Prometheus collectées
- [ ] Dashboard Grafana configuré
- [ ] Logging structuré en place
- [ ] Détection de drift implémentée
- [ ] Système d'alertes fonctionnel

---

## 🎓 Semaine 8 : Finalisation et Présentation

### Objectifs
- Finaliser le système complet
- Documentation complète
- Préparer la présentation
- Tests finaux

### Actions

#### 1. Documentation complète

- [ ] README.md à jour
- [ ] Documentation API
- [ ] Guide d'installation
- [ ] Guide de déploiement
- [ ] Architecture diagram
- [ ] Décisions techniques

#### 2. Tests finaux

```bash
# Tous les tests
pytest tests/ -v --cov=src

# Tests d'intégration bout-en-bout
python test_pipeline.py

# Load testing de l'API
# Utiliser locust ou ab (Apache Bench)
```

#### 3. Présentation

Structure recommandée (15-20 minutes):
1. **Problème métier** (2 min)
2. **Architecture MLOps** (3 min)
3. **Pipeline de données** (2 min)
4. **Modèles et résultats** (3 min)
5. **Déploiement et monitoring** (3 min)
6. **Démo live** (3 min)
7. **Leçons apprises** (2 min)

#### 4. Checklist finale

- [ ] Tous les tests passent
- [ ] CI/CD fonctionne
- [ ] Application déployée accessible
- [ ] Monitoring opérationnel
- [ ] Documentation complète
- [ ] Code review finalisée
- [ ] Repository propre et organisé

### Livrables semaine 8 (LIVRAISON FINALE)
- [ ] Système complet en production
- [ ] Documentation exhaustive
- [ ] Présentation PowerPoint/PDF
- [ ] Démo live fonctionnelle
- [ ] Rapport final du projet

---

## 📈 Critères d'évaluation

### Technique (60%)
- Qualité du code (10%)
- Tests et couverture (10%)
- Pipeline reproductible (10%)
- MLflow tracking (10%)
- Déploiement (10%)
- Monitoring (10%)

### Documentation (20%)
- README et guides (10%)
- Documentation API (5%)
- Commentaires code (5%)

### Présentation (20%)
- Clarté de la présentation (10%)
- Démo fonctionnelle (10%)

---

## 🎯 Conseils

### Gestion de projet
- Utiliser un board Kanban (GitHub Projects)
- Faire des revues de code régulières
- Commiter souvent avec des messages clairs
- Documenter au fur et à mesure

### Technique
- Commencer simple, itérer
- Tester chaque composant
- Utiliser des branches Git (feature/fix)
- Automatiser au maximum

### Communication
- Stand-ups réguliers en équipe
- Demander de l'aide tôt
- Partager les connaissances
- Documenter les décisions importantes

---

## 📚 Ressources

### Documentation officielle
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [DVC Docs](https://dvc.org/doc)
- [Docker Docs](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

### Tutoriels
- [Made With ML - MLOps](https://madewithml.com/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)

### Livres
- "Machine Learning Engineering" - Andriy Burkov
- "Building Machine Learning Pipelines" - Hannes Hapke

Bon courage pour votre projet! 🚀
