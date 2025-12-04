# Guide Complet - Fake News Detection avec RoBERTa

Intégration complète de la détection de fake news avec RoBERTa dans le workflow MLOps.

## 🎯 Vue d'ensemble

Ce projet intègre un modèle RoBERTa pour la détection de fake news dans un pipeline MLOps complet avec:
- ✅ Training local (pas Kaggle!)
- ✅ MLflow tracking
- ✅ DVC versioning
- ✅ API Flask
- ✅ Docker deployment

## 📋 Prérequis

### 1. Environnement

```bash
# Python 3.8+
python --version

# GPU recommandé (mais pas obligatoire)
nvidia-smi  # Vérifier si GPU disponible
```

### 2. Dataset

Télécharge un dataset de fake news et place-le dans `data/fake_news.csv`

**Datasets recommandés:**
- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Format requis du CSV:**
- Colonnes: `text` (obligatoire), `label` (obligatoire), `title` (optionnel)
- `label`: 0 = Real News, 1 = Fake News
- `text`: Contenu de l'article

## 🚀 Installation

### Étape 1: Setup l'environnement

```bash
# Naviguer vers le projet
cd End-to-End-Basic-ML-Project

# Créer l'environnement virtuel
python -m venv venv

# Activer (Linux/Mac)
source venv/bin/activate

# Activer (Windows)
.\venv\Scripts\Activate

# Installer les dépendances
pip install -r requirements.txt
```

### Étape 2: Préparer les données

```bash
# Créer le dossier data si nécessaire
mkdir -p data

# Copier ton dataset
cp /path/to/your/dataset.csv data/fake_news.csv

# Vérifier le format
head -n 5 data/fake_news.csv
```

## 📊 Training Local

### Option 1: Quick Test (Recommandé pour commencer)

```bash
# Terminal 1: Démarrer MLflow
mlflow ui --port 5000

# Terminal 2: Training avec échantillon (rapide!)
python run_pipeline_fakenews.py --sample 1000 --epochs 1
```

**Durée:** ~5-10 minutes (selon CPU/GPU)

### Option 2: Training Complet

```bash
# Terminal 1: MLflow
mlflow ui --port 5000

# Terminal 2: Training complet
python run_pipeline_fakenews.py \
  --data data/fake_news.csv \
  --model roberta-base \
  --epochs 3 \
  --batch-size 8
```

**Durée:**
- Avec GPU: ~30-60 minutes
- Sans GPU: 2-4 heures

### Paramètres disponibles

```bash
python run_pipeline_fakenews.py --help
```

Options:
- `--data`: Chemin vers le dataset (default: `data/fake_news.csv`)
- `--sample`: Taille échantillon pour test rapide (default: None = full)
- `--model`: Modèle à utiliser (choices: `roberta-base`, `roberta-large`)
- `--epochs`: Nombre d'époques (default: 3)
- `--batch-size`: Taille du batch (default: 8)
- `--lr`: Learning rate (default: 2e-5)
- `--mlflow-uri`: URI MLflow (default: `http://localhost:5000`)

### Exemples d'utilisation

```bash
# Test rapide avec 500 samples
python run_pipeline_fakenews.py --sample 500 --epochs 1

# Training moyen avec 5000 samples
python run_pipeline_fakenews.py --sample 5000 --epochs 2

# Training complet avec roberta-base
python run_pipeline_fakenews.py --epochs 3

# Training avec roberta-large (meilleur mais plus lent)
python run_pipeline_fakenews.py --model roberta-large --epochs 3
```

## 🔍 Tracking avec MLflow

### Démarrer MLflow UI

```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

### Informations trackées

Pour chaque run, MLflow enregistre:

**Paramètres:**
- model_name
- task
- max_length
- batch_size
- epochs
- learning_rate
- train_samples
- test_samples

**Métriques:**
- accuracy
- precision
- recall
- f1_score
- test_loss

**Artifacts:**
- Modèle PyTorch complet
- metadata.json
- label_map.json

### Comparer les runs

1. Ouvre MLflow UI: http://localhost:5000
2. Sélectionne plusieurs runs
3. Clique "Compare"
4. Analyse les métriques côte à côte

## 🧪 Tester les Prédictions

### Option 1: Script Python

```bash
python src/Pipelines/predict_pipeline_fakenews.py
```

### Option 2: Python interactif

```python
from src.Pipelines.predict_pipeline_fakenews import FakeNewsPredictionPipeline

# Charger le modèle
pipeline = FakeNewsPredictionPipeline()

# Prédiction
result = pipeline.predict_single(
    title="Breaking News: Major Discovery",
    text="Scientists have made a breakthrough in renewable energy..."
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

## 🌐 API Flask

### Démarrer l'API

```bash
python app_fakenews.py
```

L'API sera disponible sur: http://localhost:8080

### Interface Web

Ouvre http://localhost:8080 dans ton navigateur pour utiliser l'interface graphique.

### Endpoints API

#### 1. POST /predict - Prédiction simple

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking News",
    "text": "Scientists discover new cancer treatment that could save millions of lives."
  }'
```

Réponse:
```json
{
  "prediction": "Real News",
  "confidence": "92.5%",
  "probabilities": {
    "Real News": "92.5%",
    "Fake News": "7.5%"
  },
  "status": "success"
}
```

#### 2. POST /predict_batch - Prédictions multiples

```bash
curl -X POST http://localhost:8080/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {
        "title": "Article 1",
        "text": "Text 1..."
      },
      {
        "title": "Article 2",
        "text": "Text 2..."
      }
    ]
  }'
```

#### 3. GET /health - Health check

```bash
curl http://localhost:8080/health
```

#### 4. GET /info - Informations du modèle

```bash
curl http://localhost:8080/info
```

## 📦 Versioning avec DVC

### Versionner le modèle

```bash
# Ajouter le modèle à DVC
dvc add artifacts/roberta_fakenews

# Committer
git add artifacts/roberta_fakenews.dvc .gitignore
git commit -m "Add RoBERTa fake news model - F1: 0.95"
git push

# Push vers DVC remote (si configuré)
dvc push
```

### Versionner les données

```bash
# Tracker le dataset
dvc add data/fake_news.csv

# Committer
git add data/fake_news.csv.dvc
git commit -m "Add fake news dataset"
git push
```

## 🐳 Docker Deployment

### Option 1: Docker simple

```bash
# Build
docker build -t fakenews-api .

# Run
docker run -p 8080:8080 fakenews-api
```

### Option 2: Docker Compose (recommandé)

Modifie `docker-compose.yml` pour ajouter le service fake news, puis:

```bash
docker-compose up --build
```

Services disponibles:
- **API Flask**: http://localhost:8080
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📂 Structure des Fichiers

```
End-to-End-Basic-ML-Project/
├── data/
│   └── fake_news.csv                    # Dataset (à télécharger)
├── src/
│   ├── components/
│   │   ├── data_ingestion_fakenews.py   # ✨ Data loading & cleaning
│   │   └── model_trainer_roberta.py     # ✨ RoBERTa training avec MLflow
│   └── Pipelines/
│       └── predict_pipeline_fakenews.py # ✨ Prédictions
├── artifacts/
│   └── roberta_fakenews/                # Modèle entraîné
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── metadata.json
│       └── label_map.json
├── templates/
│   └── index_fakenews.html              # ✨ Interface web
├── run_pipeline_fakenews.py             # ✨ Pipeline principal
├── app_fakenews.py                      # ✨ API Flask
├── requirements.txt                      # Dépendances (transformers, torch)
└── FAKE_NEWS_GUIDE.md                   # ✨ Ce guide
```

## 🎓 Intégration dans le Projet Académique

### Semaine 3-4: Training & MLflow

```bash
# Quick test
python run_pipeline_fakenews.py --sample 1000 --epochs 1

# Full training
python run_pipeline_fakenews.py
```

### Semaine 5: API

```bash
# Tester l'API
python app_fakenews.py
```

### Semaine 6: CI/CD

Ajoute à `.github/workflows/mlops-pipeline.yml`:

```yaml
- name: Test Fake News Model
  run: |
    python run_pipeline_fakenews.py --sample 100 --epochs 1
```

### Semaine 7-8: Monitoring

Le modèle est déjà intégré avec les métriques Prometheus via l'API Flask.

## ❓ FAQ & Troubleshooting

### Q: "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install transformers torch
```

### Q: "CUDA out of memory"

Solutions:
```bash
# Réduire le batch size
python run_pipeline_fakenews.py --batch-size 4

# Utiliser moins de samples
python run_pipeline_fakenews.py --sample 500

# Utiliser CPU (plus lent)
export CUDA_VISIBLE_DEVICES=""
```

### Q: "Dataset not found"

Télécharge un dataset et place-le dans `data/fake_news.csv`. Voir section Prérequis.

### Q: "Model not loaded" dans l'API

```bash
# Vérifie que le modèle existe
ls artifacts/roberta_fakenews/

# Si non, entraîne d'abord:
python run_pipeline_fakenews.py --sample 500 --epochs 1
```

### Q: Training très lent sur CPU

Solutions:
- Utilise `--sample 500` pour tester
- Utilise `--epochs 1`
- Considère utiliser Google Colab avec GPU gratuit
- Ou utilise une instance cloud avec GPU

### Q: Comment changer le dataset?

1. Place ton CSV dans `data/`
2. Assure-toi qu'il a les colonnes: `text`, `label` (et optionnel: `title`)
3. Run: `python run_pipeline_fakenews.py --data data/ton_dataset.csv`

## 🚀 Prochaines Étapes

### Pour le projet académique:

1. ✅ **Semaine 3-4**: Training avec MLflow
   ```bash
   python run_pipeline_fakenews.py --sample 1000 --epochs 1
   ```

2. ✅ **Semaine 5**: API Flask
   ```bash
   python app_fakenews.py
   ```

3. ⏳ **Semaine 6**: CI/CD
   - Ajouter tests automatisés
   - GitHub Actions

4. ⏳ **Semaine 7-8**: Monitoring
   - Prometheus metrics
   - Grafana dashboards

### Améliorations possibles:

- [ ] Fine-tuner sur plus d'époques
- [ ] Essayer roberta-large
- [ ] Ajouter data augmentation
- [ ] Implémenter cross-validation
- [ ] Drift detection
- [ ] A/B testing entre modèles

## 📊 Résultats Attendus

Avec le training par défaut:
- **Accuracy**: ~92-95%
- **F1 Score**: ~0.93-0.96
- **Precision**: ~0.91-0.94
- **Recall**: ~0.92-0.95

## 📝 Livrables pour le Projet

Pour chaque checkpoint, tu as maintenant:

**Semaine 4:**
- ✅ Pipeline reproductible (`run_pipeline_fakenews.py`)
- ✅ MLflow tracking complet
- ✅ Tests unitaires (dans les composants)

**Semaine 6:**
- ✅ API Flask fonctionnelle
- ✅ Interface web
- ✅ Endpoints REST documentés

**Semaine 8:**
- ✅ Système complet intégré
- ✅ Documentation complète
- ✅ Prêt pour présentation

---

**Questions?** Consulte ce guide ou ouvre une issue sur GitHub!

Bon développement! 🚀
