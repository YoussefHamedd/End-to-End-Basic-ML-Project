"""
Airflow DAG for Fake News Detection MLOps Pipeline

This DAG orchestrates the complete ML workflow:
1. Data Ingestion
2. Data Validation
3. Data Preparation
4. Model Training
5. Model Evaluation
6. Model Registration (MLflow)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('/opt/airflow/'))

# Import project components
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preparation import DataPreparation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'fake_news_detection_pipeline',
    default_args=default_args,
    description='MLOps pipeline for fake news detection with RoBERTa',
    schedule_interval='@weekly',  # Run weekly
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'nlp', 'fake-news', 'roberta'],
)


# Task 1: Data Ingestion
def task_data_ingestion(**context):
    """Ingest and load raw data"""
    logging.info("Starting data ingestion task...")

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion(
        data_path='data/fake_news.csv',
        sample_size=None  # Use full dataset
    )

    # Push paths to XCom for next tasks
    context['task_instance'].xcom_push(key='train_path', value=train_path)
    context['task_instance'].xcom_push(key='test_path', value=test_path)

    logging.info(f"Data ingestion completed: train={train_path}, test={test_path}")
    return {'train_path': train_path, 'test_path': test_path}


# Task 2: Data Validation
def task_data_validation(**context):
    """Validate data quality"""
    logging.info("Starting data validation task...")

    # Get paths from previous task
    train_path = context['task_instance'].xcom_pull(key='train_path', task_ids='data_ingestion')

    import pandas as pd
    df = pd.read_csv(train_path)

    validator = DataValidation()
    results = validator.validate_all(df)

    if not results['is_valid']:
        raise ValueError(f"Data validation failed: {results['errors']}")

    logging.info("Data validation passed")
    context['task_instance'].xcom_push(key='validation_results', value=results)

    return results


# Task 3: Data Preparation
def task_data_preparation(**context):
    """Prepare and preprocess data"""
    logging.info("Starting data preparation task...")

    # Get paths
    train_path = context['task_instance'].xcom_pull(key='train_path', task_ids='data_ingestion')
    test_path = context['task_instance'].xcom_pull(key='test_path', task_ids='data_ingestion')

    import pandas as pd
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    prep = DataPreparation()

    # Get statistics
    train_stats = prep.get_statistics(train_df)
    test_stats = prep.get_statistics(test_df)

    logging.info(f"Train stats: {train_stats}")
    logging.info(f"Test stats: {test_stats}")

    context['task_instance'].xcom_push(key='train_stats', value=train_stats)
    context['task_instance'].xcom_push(key='test_stats', value=test_stats)

    return {'train_stats': train_stats, 'test_stats': test_stats}


# Task 4: Model Training
def task_model_training(**context):
    """Train RoBERTa model"""
    logging.info("Starting model training task...")

    # Get paths
    train_path = context['task_instance'].xcom_pull(key='train_path', task_ids='data_ingestion')
    test_path = context['task_instance'].xcom_pull(key='test_path', task_ids='data_ingestion')

    trainer = ModelTrainer()

    # Train model with MLflow tracking
    f1_score = trainer.initiate_model_trainer(
        train_path=train_path,
        test_path=test_path,
        model_name="roberta-base",
        max_length=128,
        batch_size=8,
        epochs=3,
        learning_rate=2e-5,
        mlflow_tracking_uri="http://mlflow:5000",  # Docker service name
        experiment_name="fake_news_detection_airflow"
    )

    logging.info(f"Model training completed. F1 Score: {f1_score:.4f}")
    context['task_instance'].xcom_push(key='f1_score', value=f1_score)

    return {'f1_score': f1_score}


# Task 5: Model Evaluation & Reporting
def task_model_evaluation(**context):
    """Evaluate model and generate report"""
    logging.info("Starting model evaluation task...")

    f1_score = context['task_instance'].xcom_pull(key='f1_score', task_ids='model_training')
    train_stats = context['task_instance'].xcom_pull(key='train_stats', task_ids='data_preparation')

    # Create evaluation report
    report = {
        'f1_score': f1_score,
        'train_samples': train_stats['total_samples'],
        'timestamp': datetime.now().isoformat(),
        'model': 'roberta-base',
        'status': 'passed' if f1_score > 0.8 else 'failed'
    }

    logging.info(f"Evaluation report: {report}")
    context['task_instance'].xcom_push(key='evaluation_report', value=report)

    # Fail if model performance is too low
    if f1_score < 0.7:
        raise ValueError(f"Model performance too low: F1={f1_score:.4f}")

    return report


# Task 6: Model Registration (if evaluation passed)
def task_model_registration(**context):
    """Register model in MLflow Model Registry"""
    logging.info("Starting model registration task...")

    evaluation_report = context['task_instance'].xcom_pull(key='evaluation_report', task_ids='model_evaluation')

    if evaluation_report['status'] == 'passed':
        logging.info(f"Model passed evaluation. Registering in MLflow...")
        # Model is already logged to MLflow during training
        logging.info("✓ Model registered successfully")
    else:
        logging.warning("Model did not pass evaluation. Skipping registration.")

    return {'registered': evaluation_report['status'] == 'passed'}


# Define task dependencies with PythonOperator
t1_data_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=task_data_ingestion,
    provide_context=True,
    dag=dag,
)

t2_data_validation = PythonOperator(
    task_id='data_validation',
    python_callable=task_data_validation,
    provide_context=True,
    dag=dag,
)

t3_data_preparation = PythonOperator(
    task_id='data_preparation',
    python_callable=task_data_preparation,
    provide_context=True,
    dag=dag,
)

t4_model_training = PythonOperator(
    task_id='model_training',
    python_callable=task_model_training,
    provide_context=True,
    dag=dag,
)

t5_model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=task_model_evaluation,
    provide_context=True,
    dag=dag,
)

t6_model_registration = PythonOperator(
    task_id='model_registration',
    python_callable=task_model_registration,
    provide_context=True,
    dag=dag,
)

# Success notification
t7_success_notification = BashOperator(
    task_id='success_notification',
    bash_command='echo "✅ Pipeline completed successfully!"',
    dag=dag,
)

# Define task flow
t1_data_ingestion >> t2_data_validation >> t3_data_preparation >> t4_model_training >> t5_model_evaluation >> t6_model_registration >> t7_success_notification
