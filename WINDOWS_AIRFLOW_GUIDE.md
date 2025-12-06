# Guide Airflow sur Windows

Airflow ne fonctionne PAS nativement sur Windows. Voici les solutions pour l'utiliser.

## ⚠️ Problème

Sur Windows, tu auras cette erreur:
```
AttributeError: module 'os' has no attribute 'register_at_fork'
```

**Raison**: Airflow utilise des fonctions Unix qui n'existent pas sur Windows.

## ✅ Solution 1: Docker (RECOMMANDÉ)

### Prérequis

1. **Installer Docker Desktop pour Windows**
   - Télécharger: https://www.docker.com/products/docker-desktop/
   - Installer et redémarrer Windows
   - Activer WSL2 si demandé

2. **Vérifier Docker**
   ```powershell
   docker --version
   docker-compose --version
   ```

### Lancer Airflow avec Docker

```powershell
# 1. Naviguer vers le projet
cd C:\Users\user\Desktop\End-to-End-Basic-ML-Project

# 2. Démarrer TOUS les services (Airflow + MLflow + Flask + Prometheus + Grafana)
docker-compose -f docker-compose-airflow.yml up -d

# 3. Attendre 1-2 minutes que tout démarre

# 4. Vérifier que tout tourne
docker-compose -f docker-compose-airflow.yml ps
```

### Accéder aux Services

Une fois démarrés, ouvre dans ton navigateur:

- **Airflow UI**: http://localhost:8081
  - Username: `admin`
  - Password: `admin`

- **MLflow UI**: http://localhost:5000

- **Flask API**: http://localhost:8080

- **Prometheus**: http://localhost:9090

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`

### Utiliser Airflow

1. **Activer le DAG**
   - Va sur http://localhost:8081
   - Login: admin/admin
   - Cherche `fake_news_detection_pipeline`
   - Clique sur le toggle pour l'activer (il doit devenir bleu)

2. **Trigger le DAG manuellement**
   - Clique sur le nom du DAG
   - Clique sur le bouton "Play" (▶️) en haut à droite
   - Clique "Trigger DAG"

3. **Voir l'exécution**
   - Clique sur "Graph" pour voir le flow
   - Clique sur une tâche pour voir les logs
   - Rafraîchis la page pour voir la progression

### Commandes Utiles

```powershell
# Voir les logs
docker-compose -f docker-compose-airflow.yml logs -f

# Voir logs d'un service spécifique
docker-compose -f docker-compose-airflow.yml logs -f airflow-webserver

# Arrêter tous les services
docker-compose -f docker-compose-airflow.yml down

# Redémarrer
docker-compose -f docker-compose-airflow.yml restart

# Rebuild si tu changes du code
docker-compose -f docker-compose-airflow.yml up -d --build

# Nettoyer complètement (attention: supprime tout!)
docker-compose -f docker-compose-airflow.yml down -v
```

### Accéder aux containers

```powershell
# Entrer dans le container Airflow
docker exec -it <container-name> bash

# Lister les DAGs
docker exec -it <container-name> airflow dags list

# Trigger un DAG
docker exec -it <container-name> airflow dags trigger fake_news_detection_pipeline
```

### Problèmes courants

#### Docker Desktop ne démarre pas
1. Assure-toi que la virtualisation est activée dans le BIOS
2. Active WSL2: `wsl --install`
3. Redémarre Windows

#### Les services ne démarrent pas
```powershell
# Voir les erreurs
docker-compose -f docker-compose-airflow.yml logs

# Nettoyer et recommencer
docker-compose -f docker-compose-airflow.yml down -v
docker-compose -f docker-compose-airflow.yml up -d
```

#### Port déjà utilisé
Si le port 8081 est déjà utilisé, modifie dans `docker-compose-airflow.yml`:
```yaml
ports:
  - "8082:8080"  # Change 8081 en 8082
```

---

## ✅ Solution 2: WSL2 (Windows Subsystem for Linux)

### Installation WSL2

```powershell
# Installer WSL2 (PowerShell en admin)
wsl --install

# Redémarrer Windows

# Installer Ubuntu
wsl --install -d Ubuntu

# Vérifier
wsl -l -v
```

### Setup dans WSL2

```bash
# Ouvrir Ubuntu (WSL2)
# Dans le terminal Ubuntu:

# 1. Mettre à jour
sudo apt update && sudo apt upgrade -y

# 2. Installer Python
sudo apt install python3.11 python3.11-venv python3-pip -y

# 3. Cloner ou copier le projet
# (Les fichiers Windows sont dans /mnt/c/Users/...)
cd /mnt/c/Users/user/Desktop/End-to-End-Basic-ML-Project

# 4. Setup Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Initialiser Airflow
airflow db init

# 6. Créer utilisateur
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# 7. Démarrer Airflow
# Terminal 1:
airflow webserver --port 8081

# Terminal 2:
airflow scheduler

# 8. MLflow (Terminal 3)
mlflow ui --port 5000
```

Accéder depuis Windows: http://localhost:8081

---

## ✅ Solution 3: Sans Airflow (Pipeline Manuel)

Si tu ne veux pas t'embêter avec Docker ou WSL2, utilise le pipeline manuel:

### Sur Windows PowerShell

```powershell
# 1. Setup (une fois)
cd C:\Users\user\Desktop\End-to-End-Basic-ML-Project
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements-windows.txt  # Sans Airflow

# 2. Démarrer MLflow
mlflow ui --port 5000

# 3. Dans un autre terminal: Lancer le pipeline
.\venv\Scripts\Activate
python run_pipeline.py --sample 1000 --epochs 1

# 4. Démarrer l'API
python app.py
```

### Créer requirements-windows.txt

```txt
# Core ML Libraries
pandas
numpy
seaborn
scikit-learn
catboost
xgboost

# Web Framework
flask

# MLOps Tools (sans Airflow)
mlflow
dvc
dvc[s3]

# Testing
pytest
pytest-cov

# Code Quality
black
flake8

# Monitoring & Logging
prometheus-client
python-dotenv

# Model Serving
gunicorn

# Transformer Models
transformers
torch

-e .
```

---

## 📊 Comparaison des Solutions

| Solution | Difficulté | Recommandé | Airflow |
|----------|------------|------------|---------|
| **Docker** | ⭐ Facile | ✅ OUI | ✅ Oui |
| **WSL2** | ⭐⭐ Moyen | 👍 Acceptable | ✅ Oui |
| **Manuel** | ⭐ Facile | ⚠️ Pour test rapide | ❌ Non |

## 🎯 Recommandation pour ton Projet Académique

**Utilise Docker** ! C'est:
- ✅ Plus simple à setup
- ✅ Fonctionne exactement pareil que Linux
- ✅ Déjà prêt (docker-compose-airflow.yml fourni)
- ✅ Professionnel (montrer que tu sais utiliser Docker)
- ✅ Tous les services en un seul commande

## 🚀 Quick Start Docker (2 minutes)

```powershell
# 1. Avoir Docker Desktop installé et lancé

# 2. Une seule commande !
docker-compose -f docker-compose-airflow.yml up -d

# 3. Attendre 1-2 minutes

# 4. Ouvrir http://localhost:8081 (admin/admin)

# 5. Profit! 🎉
```

---

## ❓ Questions Fréquentes

### Q: Docker est lent sur Windows?
A: Assure-toi d'utiliser WSL2 backend (par défaut sur Docker Desktop récent)

### Q: Je peux développer sur Windows et déployer sur Linux?
A: OUI! C'est justement l'intérêt de Docker - le même container tourne partout

### Q: Airflow est obligatoire pour le projet?
A: Non, mais c'est un gros plus pour montrer l'orchestration MLOps professionnelle

### Q: Je peux mixer Docker + développement local?
A: OUI! Lance Airflow dans Docker, développe localement, test avec `python run_pipeline.py`

---

## ✅ Checklist Windows

- [ ] Docker Desktop installé
- [ ] WSL2 activé (si demandé)
- [ ] `docker-compose -f docker-compose-airflow.yml up -d`
- [ ] http://localhost:8081 accessible
- [ ] DAG visible dans Airflow UI
- [ ] MLflow sur http://localhost:5000
- [ ] Flask API sur http://localhost:8080

**Tout coché?** Tu es prêt! 🚀

---

Besoin d'aide? Les logs sont ton ami:
```powershell
docker-compose -f docker-compose-airflow.yml logs -f
```
