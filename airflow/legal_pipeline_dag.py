from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
}

with DAG(
    dag_id="legal_risk_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    scrape_data = BashOperator(
        task_id="scrape_data",
        bash_command="python /opt/project/scraper/courtlistener_scraper.py"
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /opt/project/models/risk_classifier.py"
    )

    entity_extraction = BashOperator(
        task_id="entity_extraction",
        bash_command="python /opt/project/models/entity_extraction.py"
    )

    generate_analytics = BashOperator(
    task_id="generate_analytics",
    bash_command="python /opt/project/models/risk_analytics.py"
    )


    scrape_data >> train_model >> entity_extraction >> generate_analytics
