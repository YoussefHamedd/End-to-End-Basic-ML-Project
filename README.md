# Projet MLOps - Détection de Fake News

Projet académique MLOps complet avec RoBERTa pour détecter les fake news.

**Enseignant:** Sonia Gharsalli

---

## 🎯 Ce que fait ce projet

- 🤖 **Détecte les fake news** avec un modèle RoBERTa (deep learning)
- 📊 **Track les expériences** avec MLflow
- 📦 **Version les modèles** avec DVC
- 🔄 **Orchestre le workflow** avec Prefect (optionnel)
- 🌐 **API Flask** pour faire des prédictions
- 🐳 **Déployable** avec Docker

**Performance**: Accuracy ~92-95%, F1 Score ~0.93-0.96

---

## ⚡ Quick Start (Windows/Mac/Linux)

### 1. Installation (2 minutes)

```bash
# Clone le projet
git clone <repo-url>
cd End-to-End-Basic-ML-Project

# Environnement virtuel
python -m venv venv

# Windows:
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# Installer
pip install -r requirements.txt
```

### 2. Télécharge un dataset fake news

Place ton CSV dans `data/fake_news.csv`

**Format requis**: colonnes `text`, `label` (et optionnel: `title`)

**Datasets suggérés**:
- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 3. Test Rapide (5-10 min)

```bash
# Terminal 1: MLflow
mlflow ui --port 5000

# Terminal 2: Training rapide
python run_pipeline.py --sample 1000 --epochs 1

# Voir les résultats: http://localhost:5000
```

### 4. Lancer l'API

```bash
python app.py
# Ouvrir: http://localhost:8080
```

**Voilà, c'est tout!** 🎉

---

## 📁 Structure du Projet

```
End-to-End-Basic-ML-Project/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # 📥 Charge les données
│   │   ├── data_validation.py      # ✅ Valide la qualité
│   │   ├── data_preparation.py     # 🧹 Nettoie le texte
│   │   └── model_trainer.py        # 🤖 Entraîne RoBERTa
│   │
│   └── Pipelines/
│       └── predict_pipeline.py      # 🔮 Prédictions
│
├── prefect_flows/
│   └── ml_pipeline_flow.py          # 🔄 Orchestration Prefect
│
├── data/
│   └── fake_news.csv                # Ton dataset
│
├── artifacts/                       # Modèles sauvegardés
├── logs/                            # Logs
│
├── app.py                           # 🌐 API Flask
├── run_pipeline.py                  # 🎯 Pipeline principal
└── requirements.txt                 # Dépendances
```

---

## 🚀 Utilisation Complète

### Training

```bash
# Test rapide (500 samples, 1 epoch) - 5 min
python run_pipeline.py --sample 500 --epochs 1

# Training moyen (2000 samples, 2 epochs) - 20 min
python run_pipeline.py --sample 2000 --epochs 2

# Training complet (tout le dataset, 3 epochs) - 1-3h
python run_pipeline.py --epochs 3

# Avec roberta-large (meilleur mais plus lent)
python run_pipeline.py --model roberta-large --epochs 3
```

**Paramètres disponibles**:
- `--data`: Chemin dataset (default: `data/fake_news.csv`)
- `--sample`: Nombre d'échantillons (default: None = tout)
- `--model`: `roberta-base` ou `roberta-large`
- `--epochs`: Nombre d'époques (default: 3)
- `--batch-size`: Taille batch (default: 8)
- `--mlflow-uri`: URI MLflow (default: `http://localhost:5000`)

### Voir les Expériences (MLflow)

```bash
# Démarrer MLflow UI
mlflow ui --port 5000

# Ouvrir: http://localhost:5000
```

Dans MLflow tu peux:
- ✅ Comparer les runs
- ✅ Voir les métriques (accuracy, F1, etc.)
- ✅ Télécharger les modèles
- ✅ Visualiser les paramètres

### API Flask

```bash
# Démarrer l'API
python app.py

# API disponible sur: http://localhost:8080
```

**Endpoints**:

```bash
# Prédiction simple
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Breaking News", "text": "Article content here..."}'

# Prédictions multiples
curl -X POST http://localhost:8080/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"articles": [{"title": "...", "text": "..."}, ...]}'

# Health check
curl http://localhost:8080/health

# Info modèle
curl http://localhost:8080/info
```

**Ou utilise l'interface web**: http://localhost:8080

### Orchestration avec Prefect (Optionnel)

**Prefect** orchestre ton workflow avec gestion d'erreurs automatique, retries, et monitoring.

```bash
# Option 1: Pipeline direct avec Prefect
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# Option 2: Avec UI de monitoring
# Terminal 1: Prefect UI
prefect server start

# Terminal 2: MLflow UI
mlflow ui --port 5000

# Terminal 3: Run pipeline
python prefect_flows/ml_pipeline_flow.py --epochs 3

# Voir résultats:
# - Prefect UI: http://localhost:4200
# - MLflow UI: http://localhost:5000
```

**Avantages Prefect**:
- ✅ Fonctionne nativement sur Windows (pas comme Airflow!)
- ✅ Retries automatiques en cas d'erreur
- ✅ Interface moderne de monitoring
- ✅ Setup simple: `pip install prefect`

**Guide complet**: Voir [PREFECT_SETUP.md](PREFECT_SETUP.md)

### Versionner avec DVC

```bash
# Versionner le modèle
dvc add artifacts/roberta_fakenews

# Commit
git add artifacts/roberta_fakenews.dvc .gitignore
git commit -m "Model v1 - F1: 0.95"
git push

# Optionnel: Push vers remote storage
dvc push
```

---

## 🛠️ Technologies

| Catégorie | Tech |
|-----------|------|
| **ML** | Transformers (RoBERTa), PyTorch |
| **Tracking** | MLflow |
| **Versioning** | DVC, Git |
| **Orchestration** | Prefect (optionnel) |
| **API** | Flask |
| **Deploy** | Docker |
| **Tests** | pytest |

---

## 🎓 Projet Académique (8 Semaines)

### ✅ Semaine 1-2: Setup & EDA
- [x] Git, DVC, MLflow
- [x] Dataset fake news
- [x] EDA & feature engineering

### ✅ Semaine 3-4: Pipeline ML (CHECKPOINT 1)
- [x] Modules modulaires (ingestion, validation, prep, training)
- [x] MLflow tracking
- [x] Pipeline reproductible
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
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Documentation finale
- [ ] Présentation

---

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture
pytest --cov=src tests/

# Test spécifique
pytest tests/test_data_validation.py -v
```

---

## 🐳 Docker (Optionnel)

```bash
# Build
docker build -t fakenews-api .

# Run
docker run -p 8080:8080 fakenews-api

# Ou avec docker-compose
docker-compose up --build
```

---

## 📖 Documentation Complète

| Guide | Description |
|-------|-------------|
| **[FAKE_NEWS_GUIDE.md](FAKE_NEWS_GUIDE.md)** | 📰 Guide détaillé fake news |
| **[PROJET_MLOPS.md](PROJET_MLOPS.md)** | 🎓 Guide académique 8 semaines |
| **[PREFECT_SETUP.md](PREFECT_SETUP.md)** | 🔄 Orchestration avec Prefect |
| **[QUICKSTART_MLOPS.md](QUICKSTART_MLOPS.md)** | ⚡ Quick start rapide |

---

## 🔧 Commandes Essentielles

```bash
# Training
python run_pipeline.py --sample 1000 --epochs 1

# Training avec Prefect (orchestration)
python prefect_flows/ml_pipeline_flow.py --sample 1000 --epochs 1

# MLflow
mlflow ui --port 5000

# Prefect UI (optionnel)
prefect server start

# API
python app.py

# Tests
pytest tests/ -v

# DVC
dvc add artifacts/roberta_fakenews
git add *.dvc && git commit -m "Update model"
```

---

## ❓ FAQ

### Q: Ça marche sur Windows?
**Oui!** Tout fonctionne sur Windows, Mac et Linux.

### Q: J'ai besoin d'un GPU?
**Non**, mais c'est plus rapide. Sur CPU:
- Test (500 samples): ~5-10 min
- Complet: ~1-3 heures

### Q: Le dataset est trop gros?
Utilise `--sample 500` pour tester rapidement.

### Q: Comment changer de dataset?
Place ton CSV dans `data/` avec colonnes `text` et `label`, puis:
```bash
python run_pipeline.py --data data/mon_dataset.csv
```

### Q: MLflow ne se connecte pas?
Vérifie que MLflow UI tourne sur le bon port:
```bash
mlflow ui --port 5000
```

---

## 🎯 Livrables Académiques

- **Checkpoint 1 (Semaine 4)**: ✅ Pipeline + MLflow
- **Checkpoint 2 (Semaine 6)**: ⏳ API + CI/CD
- **Final (Semaine 8)**: ⏳ Production + Monitoring

---

## 🚀 Get Started Maintenant!

```bash
pip install -r requirements.txt
python run_pipeline.py --sample 500 --epochs 1
mlflow ui --port 5000
python app.py
```

**C'est tout!** Simple et efficace. 🎉

---

📖 **Plus de détails?** Voir [FAKE_NEWS_GUIDE.md](FAKE_NEWS_GUIDE.md)

🆘 **Problème?** Ouvre une issue GitHub
