# Airflow Orchestration - Setup Guide

Guide complet pour utiliser Airflow pour orchestrer le pipeline MLOps de détection de fake news.

## 🎯 Qu'est-ce qu'Airflow?

Apache Airflow est une plateforme d'orchestration de workflows qui permet de:
- ✅ Programmer et automatiser les pipelines ML
- ✅ Visualiser les dépendances entre tâches
- ✅ Monitorer l'exécution en temps réel
- ✅ Gérer les retries et erreurs
- ✅ Orchestrer des workflows complexes

## 📊 Architecture du Pipeline

Notre DAG (Directed Acyclic Graph) Airflow:

```
Data Ingestion → Data Validation → Data Preparation → Model Training → Model Evaluation → Model Registration → Success Notification
```

### Tâches du Pipeline

1. **Data Ingestion**: Charge le dataset fake news
2. **Data Validation**: Valide qualité et schéma
3. **Data Preparation**: Nettoie et prépare les données
4. **Model Training**: Entraîne RoBERTa avec MLflow
5. **Model Evaluation**: Évalue les performances
6. **Model Registration**: Enregistre dans MLflow si OK
7. **Success Notification**: Notification de succès

## 🚀 Installation

### Option 1: Installation Locale (Simple)

```bash
# Installer Airflow
pip install apache-airflow==2.8.1
pip install apache-airflow-providers-docker

# Initialiser la base de données
airflow db init

# Créer un utilisateur admin
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Démarrer le webserver
airflow webserver --port 8081

# Dans un autre terminal, démarrer le scheduler
airflow scheduler
```

### Option 2: Docker Compose (Recommandé)

Utilise `docker-compose-airflow.yml`:

```bash
# Démarrer Airflow + MLflow + PostgreSQL
docker-compose -f docker-compose-airflow.yml up -d

# Services disponibles:
# - Airflow UI: http://localhost:8081 (admin/admin)
# - MLflow: http://localhost:5000
# - PostgreSQL: localhost:5432
```

## 📁 Structure des Fichiers

```
End-to-End-Basic-ML-Project/
├── airflow/
│   ├── dags/
│   │   └── fake_news_ml_pipeline.py    # DAG principal
│   ├── logs/                             # Logs Airflow
│   └── plugins/                          # Plugins custom
├── src/
│   ├── components/
│   │   ├── data_ingestion.py           # Tâche 1
│   │   ├── data_validation.py          # Tâche 2
│   │   ├── data_preparation.py         # Tâche 3
│   │   └── model_trainer.py            # Tâche 4
│   └── Pipelines/
└── data/
    └── fake_news.csv                    # Dataset
```

## 🎮 Utilisation

### 1. Préparer les données

```bash
# Télécharge ton dataset fake news
# Place-le dans: data/fake_news.csv
```

### 2. Démarrer Airflow

**Méthode Locale:**
```bash
# Terminal 1: Webserver
airflow webserver --port 8081

# Terminal 2: Scheduler
airflow scheduler
```

**Méthode Docker:**
```bash
docker-compose -f docker-compose-airflow.yml up -d
```

### 3. Accéder à Airflow UI

Ouvre http://localhost:8081
- Username: `admin`
- Password: `admin`

### 4. Activer le DAG

1. Va dans l'onglet "DAGs"
2. Cherche `fake_news_detection_pipeline`
3. Clique sur le toggle pour l'activer
4. Clique sur "Trigger DAG" pour lancer manuellement

### 5. Monitorer l'exécution

Dans Airflow UI:
- **Graph View**: Voir le flow des tâches
- **Tree View**: Historique des exécutions
- **Logs**: Logs détaillés de chaque tâche
- **Gantt**: Timeline d'exécution

## 📅 Configuration du Schedule

Le DAG est configuré pour s'exécuter automatiquement:

```python
schedule_interval='@weekly'  # Chaque semaine
```

Autres options:
- `@daily`: Chaque jour
- `@hourly`: Chaque heure
- `'0 0 * * 1'`: Cron expression (lundi à minuit)
- `None`: Manuel uniquement

Pour modifier:
```python
# Dans airflow/dags/fake_news_ml_pipeline.py
schedule_interval='@daily',  # Changement ici
```

## 🔧 Configuration Avancée

### Variables Airflow

Définir des variables globales:

```python
# Via UI: Admin > Variables
# Ou en CLI:
airflow variables set DATA_PATH data/fake_news.csv
airflow variables set MODEL_NAME roberta-base
airflow variables set EPOCHS 3
```

Utiliser dans le DAG:
```python
from airflow.models import Variable

data_path = Variable.get("DATA_PATH", default_var="data/fake_news.csv")
```

### Connections

Configurer MLflow connection:

```bash
# Via CLI
airflow connections add 'mlflow_default' \
    --conn-type 'http' \
    --conn-host 'localhost' \
    --conn-port '5000'
```

### XCom (Cross-Communication)

Les tâches partagent des données via XCom:

```python
# Push data
context['task_instance'].xcom_push(key='train_path', value=train_path)

# Pull data
train_path = context['task_instance'].xcom_pull(
    key='train_path',
    task_ids='data_ingestion'
)
```

## 📊 Monitoring

### Logs

Consulter les logs d'une tâche:
1. Clique sur la tâche dans le graph
2. Clique sur "Log"
3. Voir les logs en temps réel

### Métriques

Airflow track automatiquement:
- Durée d'exécution
- Taux de succès/échec
- Retries
- SLA violations

### Alertes

Configurer des alertes par email:

```python
default_args = {
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': True,
}
```

## 🐛 Troubleshooting

### DAG n'apparaît pas

```bash
# Vérifier les erreurs
airflow dags list

# Tester le DAG
python airflow/dags/fake_news_ml_pipeline.py

# Vérifier les imports
airflow dags list-import-errors
```

### Tâche échoue

1. Consulter les logs dans Airflow UI
2. Vérifier que les dépendances sont installées
3. Tester la tâche manuellement:

```bash
# Tester une tâche spécifique
airflow tasks test fake_news_detection_pipeline data_ingestion 2024-01-01
```

### Import errors

Assurer que le PYTHONPATH inclut le projet:

```bash
export AIRFLOW__CORE__DAGS_FOLDER=/path/to/End-to-End-Basic-ML-Project/airflow/dags
export PYTHONPATH=/path/to/End-to-End-Basic-ML-Project:$PYTHONPATH
```

## 📈 Workflow Complet

### Scénario: Ré-entraînement hebdomadaire

1. **Lundi 00:00**: Airflow déclenche automatiquement le DAG
2. **Data Ingestion**: Charge les nouvelles données
3. **Validation**: Vérifie la qualité
4. **Preparation**: Nettoie les données
5. **Training**: Entraîne nouveau modèle (2-4h)
6. **Evaluation**: Compare avec modèle actuel
7. **Registration**: Enregistre si meilleur
8. **Notification**: Email de succès/échec

### Scénario: Trigger manuel

1. Ouvre Airflow UI
2. Sélectionne le DAG
3. Clique "Trigger DAG"
4. Optionnel: Passe des paramètres custom
5. Monitore dans Graph View

## 🎯 Intégration avec MLOps

### Avec MLflow

```python
# Le DAG log automatiquement dans MLflow
mlflow_tracking_uri="http://mlflow:5000"
experiment_name="fake_news_detection_airflow"
```

Voir les runs dans MLflow UI: http://localhost:5000

### Avec DVC

Après training réussi:

```bash
# Version le modèle
dvc add artifacts/roberta_fakenews
git add artifacts/roberta_fakenews.dvc
git commit -m "Model trained via Airflow - F1: 0.95"
git push
dvc push
```

### Avec CI/CD

Intégrer Airflow dans GitHub Actions:

```yaml
- name: Trigger Airflow DAG
  run: |
    airflow dags trigger fake_news_detection_pipeline
```

## 🚀 Best Practices

### 1. Idempotence

Assurer que les tâches peuvent être réexécutées:

```python
# Supprimer les fichiers existants avant écriture
if os.path.exists(output_path):
    os.remove(output_path)
```

### 2. Retries

Configurer des retries intelligents:

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
}
```

### 3. Timeouts

Définir des timeouts:

```python
t4_model_training = PythonOperator(
    task_id='model_training',
    execution_timeout=timedelta(hours=4),
    ...
)
```

### 4. Dependencies

Gérer les dépendances externes:

```python
from airflow.sensors.filesystem import FileSensor

wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='data/fake_news.csv',
    timeout=600,
)

wait_for_data >> t1_data_ingestion
```

## 📚 Ressources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [DAG Writing Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

## ✅ Checklist Projet Académique

Pour ton projet MLOps:

- [x] Pipeline orchestré avec Airflow
- [x] Tâches modulaires (ingestion, validation, prep, train)
- [x] MLflow tracking intégré
- [x] Logs structurés
- [x] Monitoring via Airflow UI
- [x] Schedule automatique
- [x] Retry logic
- [ ] Email notifications (optionnel)
- [ ] Custom sensors (optionnel)
- [ ] Alertes SLA (optionnel)

---

🎉 Ton pipeline MLOps est maintenant complètement orchestré avec Airflow!
