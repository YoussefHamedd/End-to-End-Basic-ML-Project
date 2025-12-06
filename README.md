# Projet MLOps - Détection de Fake News

Projet académique MLOps sur 8 semaines implémentant un pipeline complet de détection de fake news avec orchestration Airflow.

**Enseignant:** Sonia Gharsalli

## 🔥 Architecture MLOps Complète

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Airflow   │───▶│  Data Prep   │───▶│  Training   │───▶│   MLflow     │
│ Orchestrator│    │  Validation  │    │   RoBERTa   │    │  Tracking    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Docker    │◀───│   Flask API  │◀───│     DVC     │◀───│  Registered  │
│  Deployment │    │  Prediction  │    │  Versioning │    │    Model     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

## 🎯 Use Case: Fake News Detection

- 🤖 **Modèle**: RoBERTa-base/large pour NLP
- 📊 **Performance**: F1 Score ~0.93-0.96, Accuracy ~92-95%
- 🔄 **Orchestration**: Airflow DAG automatisé
- 📈 **Tracking**: MLflow experiment tracking
- 🚀 **API**: Flask REST API avec interface web

## 📋 Pipeline Airflow (7 Étapes)

```
Data Ingestion → Validation → Preparation → Training → Evaluation → Registration → Notification
```

**Voir guide complet**: [AIRFLOW_SETUP.md](AIRFLOW_SETUP.md)

## 🚀 Quick Start

### Méthode 1: Airflow Orchestration ⭐ (Recommandé)

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data: Placer fake_news.csv dans data/

# 3. Airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8081 &
airflow scheduler &

# 4. MLflow (terminal séparé)
mlflow ui --port 5000 &

# 5. Ouvrir http://localhost:8081 → Trigger DAG
```

### Méthode 2: Manuel (Sans Airflow)

```bash
# Test rapide
python run_pipeline.py --sample 1000 --epochs 1

# API
python app.py
# http://localhost:8080
```

## 📁 Structure Modulaire

```
├── airflow/dags/fake_news_ml_pipeline.py    # 🚀 DAG Orchestration
├── src/components/
│   ├── data_ingestion.py                    # 📥 Chargement
│   ├── data_validation.py                   # ✅ Validation
│   ├── data_preparation.py                  # 🧹 Nettoyage
│   └── model_trainer.py                     # 🤖 Training
├── src/Pipelines/predict_pipeline.py        # 🔮 Prédictions
├── app.py                                    # 🌐 Flask API
└── run_pipeline.py                          # 🎯 Pipeline manuel
```

## 🛠️ Technologies

- **Orchestration**: Apache Airflow 2.8
- **ML**: Transformers (RoBERTa), PyTorch
- **MLOps**: MLflow, DVC
- **API**: Flask
- **Container**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **CI/CD**: GitHub Actions

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| **[AIRFLOW_SETUP.md](AIRFLOW_SETUP.md)** | 🔄 Setup & utilisation Airflow |
| **[FAKE_NEWS_GUIDE.md](FAKE_NEWS_GUIDE.md)** | 📰 Guide fake news detection |
| **[PROJET_MLOPS.md](PROJET_MLOPS.md)** | 🎓 Guide académique 8 semaines |
| **[QUICKSTART_MLOPS.md](QUICKSTART_MLOPS.md)** | ⚡ Quick start général |

## 🎓 Projet Académique - Progression

### ✅ Semaine 1-2: Setup
- [x] Git, DVC, MLflow, Docker, Airflow
- [x] EDA & feature engineering

### ✅ Semaine 3-4: Pipeline (CHECKPOINT 1)
- [x] Modules data prep, validation, training
- [x] Airflow DAG orchestration
- [x] MLflow tracking
- [x] Tests unitaires

### ✅ Semaine 5: API
- [x] Flask API
- [x] Interface web
- [x] Documentation

### ⏳ Semaine 6: CI/CD (CHECKPOINT 2)
- [ ] GitHub Actions
- [ ] Tests automatisés
- [ ] Docker deployment

### ⏳ Semaine 7-8: Production (FINAL)
- [ ] Prometheus + Grafana
- [ ] Monitoring complet
- [ ] Documentation finale

## 🔧 Commandes Essentielles

```bash
# Airflow
airflow webserver --port 8081
airflow scheduler
airflow dags trigger fake_news_detection_pipeline

# MLflow
mlflow ui --port 5000

# Training
python run_pipeline.py --sample 1000 --epochs 1

# API
python app.py

# Tests
pytest tests/ -v

# Docker
docker-compose up --build
```

## 📊 API Endpoints

```bash
# Prédiction
POST /predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "News Title", "text": "Article content..."}'

# Health
GET /health

# Info modèle
GET /info
```

## 🔄 Workflow MLOps

1. **Develop**: Test local avec `run_pipeline.py`
2. **Orchestrate**: Airflow DAG automatique
3. **Track**: MLflow experiments
4. **Version**: DVC + Git
5. **Deploy**: Docker Compose
6. **Monitor**: Airflow UI + Grafana

## 🎯 Livrables Académiques

- **Checkpoint 1 (S4)**: ✅ Pipeline orchestré + MLflow
- **Checkpoint 2 (S6)**: ⏳ API + CI/CD
- **Final (S8)**: ⏳ Production + Monitoring

---

📖 **Voir guides détaillés pour setup complet!**

🚀 **Get Started**: `python run_pipeline.py --sample 500 --epochs 1`
