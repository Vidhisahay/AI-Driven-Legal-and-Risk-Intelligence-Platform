from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT_PATH = "E:/Projects/ai-legal-risk-intelligence"

with DAG(
    dag_id="legal_risk_pipeline",
    start_date=datetime(2024,1,1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    scrape = BashOperator(
        task_id="scrape_data",
        bash_command=f"python {PROJECT_PATH}/scraper/courtlistener_scraper.py"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=f"python {PROJECT_PATH}/models/risk_classifier.py"
    )

    entities = BashOperator(
        task_id="entity_extraction",
        bash_command=f"python {PROJECT_PATH}/models/entity_extraction.py"
    )

    analytics = BashOperator(
        task_id="generate_analytics",
        bash_command=f"python {PROJECT_PATH}/models/legal_analytics.py"
    )

    scrape >> train >> entities >> analytics