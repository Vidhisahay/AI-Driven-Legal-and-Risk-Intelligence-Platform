import subprocess

print("Running legal risk intelligence pipeline")

steps = [
    "scraper/courtlistener_scraper.py",
    "models/risk_classifier.py",
    "models/entity_extraction.py",
    "models/legal_analytics.py"
]

for step in steps:

    print("\nRunning:", step)

    subprocess.run(
        ["python", step],
        check=True
    )

print("\nPipeline completed successfully")