# Projet MLOps - Machine Learning en Production

Projet académique MLOps sur 8 semaines pour mettre en œuvre le cycle de vie complet d'un projet de machine learning en production.

**Enseignant:** Sonia Gharsalli

## 🔥 Use Case: Fake News Detection avec RoBERTa

Ce projet implémente un système complet de détection de fake news avec:
- 🤖 Modèle RoBERTa (state-of-the-art NLP)
- 📊 Training local avec MLflow tracking
- 🚀 API Flask pour prédictions en temps réel
- 📦 Pipeline MLOps complet (DVC + MLflow + Docker)
- 🌐 Interface web interactive

**📖 Guide complet:** [FAKE_NEWS_GUIDE.md](FAKE_NEWS_GUIDE.md)

## 🎯 Objectifs pédagogiques

- ✅ Comprendre et appliquer les principes MLOps
- ✅ Maîtriser les outils de versioning (code, données, modèles)
- ✅ Implémenter des pipelines de ML reproductibles
- ✅ Déployer et monitorer des modèles en production
- ✅ Collaborer efficacement en équipe

## 📋 Structure du projet (8 semaines)

### Semaine 1-2 ✓ : Setup & Exploration
- [x] Configuration environnement (Git, DVC, MLflow, Docker)
- [x] Analyse exploratoire des données
- [x] Feature engineering reproductible
- [x] Versioning des données avec DVC

### Semaine 3-4 : Développement & Pipeline
- [ ] Développement des modèles ML
- [ ] Tracking avec MLflow
- [ ] Pipeline de formation reproductible
- [ ] Tests unitaires

### Semaine 5 : API & Inférence
- [ ] API de prédiction (Flask/FastAPI)
- [ ] Containerisation Docker
- [ ] Tests d'intégration
- [ ] Documentation API

### Semaine 6 : CI/CD & Déploiement
- [ ] GitHub Actions pour CI/CD
- [ ] Déploiement automatisé
- [ ] Environnements dev/staging/prod

### Semaine 7-8 : Monitoring & Production
- [ ] Monitoring des modèles (Prometheus/Grafana)
- [ ] Logging structuré
- [ ] Détection de drift
- [ ] Alertes automatiques

## 🚀 Quick Start

### Installation

```bash
# Cloner le projet
git clone <votre-repo>
cd End-to-End-Basic-ML-Project

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\Activate  # Windows

# Installer les dépendances (inclut transformers & torch)
pip install -r requirements.txt

# Initialiser DVC
dvc init
dvc pull  # Télécharger les données versionnées
```

### Option 1: Fake News Detection 🔥 (Recommandé)

```bash
# 1. Télécharger un dataset de fake news dans data/fake_news.csv

# 2. Terminal 1: Démarrer MLflow
mlflow ui --port 5000

# 3. Terminal 2: Training rapide (test)
python run_pipeline_fakenews.py --sample 1000 --epochs 1

# 4. Lancer l'API
python app_fakenews.py

# 5. Ouvrir http://localhost:8080
```

**Voir le guide complet:** [FAKE_NEWS_GUIDE.md](FAKE_NEWS_GUIDE.md)

### Option 2: Pipeline Original (Student Performance)

```bash
# Lancer le pipeline de formation
python run_pipeline.py

# Démarrer MLflow UI
mlflow ui --port 5000

# Lancer l'application Flask
python app.py
```

### Avec Docker

```bash
# Build et démarrer tous les services
docker-compose up --build

# Services disponibles :
# - Flask App: http://localhost:8080
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## 📁 Structure du projet

```
.
├── src/
│   ├── components/          # Composants ML (ingestion, transformation, training)
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── Pipelines/           # Pipelines de prédiction
│       └── predict_pipeline.py
├── artifacts/               # Modèles et artefacts versionnés
├── data/                    # Données brutes (versionnées avec DVC)
├── logs/                    # Logs de l'application
├── tests/                   # Tests unitaires et d'intégration
├── .github/workflows/       # CI/CD GitHub Actions
├── monitoring/              # Configuration Prometheus/Grafana
├── Notebook/                # Notebooks d'exploration (EDA)
├── docker-compose.yml       # Orchestration des services
├── Dockerfile              # Image Docker de l'application
├── dvc.yaml                # Pipeline DVC
├── requirements.txt        # Dépendances Python
├── app.py                  # API Flask
└── run_pipeline.py         # Script de formation
```

## 🛠️ Technologies utilisées

### Core ML
- **Python 3.8+** - Langage principal
- **scikit-learn** - Modèles ML
- **pandas, numpy** - Manipulation de données

### MLOps Stack
- **MLflow** - Tracking d'expériences et registry de modèles
- **DVC** - Versioning des données et modèles
- **Docker** - Containerisation
- **Flask** - API REST pour les prédictions

### CI/CD & Monitoring
- **GitHub Actions** - Intégration et déploiement continus
- **Prometheus** - Collecte de métriques
- **Grafana** - Visualisation des métriques
- **pytest** - Tests automatisés

## 📊 Pipeline MLOps

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Données   │───▶│  DVC Track   │───▶│  Git Push   │
└─────────────┘    └──────────────┘    └─────────────┘
                            │
                            ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Formation  │───▶│ MLflow Track │───▶│   Modèle    │
└─────────────┘    └──────────────┘    └─────────────┘
                            │
                            ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Docker    │───▶│     API      │───▶│ Monitoring  │
│    Build    │    │  Déployée    │    │ Prometheus  │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/

# Tests avec couverture
pytest --cov=src tests/

# Tests spécifiques
pytest tests/test_model.py
pytest tests/test_app.py
```

## 📖 Documentation détaillée

- **[PROJET_MLOPS.md](PROJET_MLOPS.md)** - Guide complet semaine par semaine
- **[QUICKSTART_MLOPS.md](QUICKSTART_MLOPS.md)** - Guide de démarrage rapide
- **[MLOPS_WORKFLOW.md](MLOPS_WORKFLOW.md)** - Workflow détaillé
- **[MLOPS_SETUP.md](MLOPS_SETUP.md)** - Configuration de l'environnement

## 🔄 Workflow de développement

### 1. Expérimentation (Notebooks)
```bash
# Exploration dans Notebook/EDA.ipynb
jupyter notebook
```

### 2. Développement (Scripts Python)
```bash
# Coder dans src/components/
# Ajouter tests dans tests/
pytest tests/
```

### 3. Versioning
```bash
# Versionner le code
git add .
git commit -m "Ajout nouvelle feature"
git push

# Versionner les données/modèles
dvc add data/train.csv
dvc add artifacts/model.pkl
git add data/train.csv.dvc artifacts/model.pkl.dvc
git commit -m "Update data and model"
git push
dvc push
```

### 4. Training avec MLflow
```bash
# Lancer formation avec tracking
python run_pipeline.py

# Voir les résultats
mlflow ui
```

### 5. Déploiement
```bash
# Build et déployer
docker-compose up --build

# Tester l'API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [...]}'
```

## 📈 Métriques et Monitoring

### MLflow
- Tracking des hyperparamètres
- Métriques de performance (R², MSE, etc.)
- Versioning des modèles
- Comparaison d'expériences

### Prometheus
- Latence des prédictions
- Nombre de requêtes
- Erreurs API
- Métriques système

### Grafana
- Dashboards interactifs
- Alertes sur métriques critiques
- Visualisation temps réel

## 🚨 Bonnes pratiques

### Code Quality
- ✅ Suivre PEP 8
- ✅ Écrire des docstrings
- ✅ Tester chaque composant
- ✅ Review de code en équipe

### Data Science
- ✅ Versionner les données avec DVC
- ✅ Logger toutes les expériences avec MLflow
- ✅ Valider les modèles avant déploiement
- ✅ Documenter les choix métier

### DevOps
- ✅ Containeriser avec Docker
- ✅ Automatiser avec CI/CD
- ✅ Monitorer en production
- ✅ Gérer les secrets avec .env

## 👥 Équipe

- **Nom Étudiant 1** - Rôle
- **Nom Étudiant 2** - Rôle
- **Nom Étudiant 3** - Rôle

## 📝 Livrables attendus

### Semaine 4 (Checkpoint 1)
- [ ] Repository Git bien structuré
- [ ] Pipeline de formation fonctionnel
- [ ] Tests unitaires
- [ ] Documentation technique

### Semaine 6 (Checkpoint 2)
- [ ] API de prédiction déployée
- [ ] CI/CD opérationnel
- [ ] Containerisation complète

### Semaine 8 (Livraison finale)
- [ ] Système complet en production
- [ ] Monitoring actif
- [ ] Documentation complète
- [ ] Présentation du projet

## 🆘 Support

- **Questions techniques** : Ouvrir une issue GitHub
- **Documentation** : Consulter les fichiers .md
- **Debugging** : Vérifier les logs dans `logs/`

## 📄 Licence

MIT License - Projet académique
