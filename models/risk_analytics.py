import pandas as pd
import spacy
import re

print("Loading raw dataset")

df = pd.read_csv("data/raw/legal_cases.csv")

print("Loading NLP model")

nlp = spacy.load("en_core_web_sm")

organizations = []
risk_labels = []
years = []

risk_words = [
    "fraud",
    "penalty",
    "violation",
    "criminal",
    "sanction",
    "illegal",
    "bribery",
    "money laundering"
]

print("Running analytics pipeline")

for text, date in zip(df["text"].fillna(""), df["date"]):

    text_str = str(text)

    # -------- Entity Extraction --------
    doc = nlp(text_str[:2000])

    org = "Unknown"

    for ent in doc.ents:
        if ent.label_ == "ORG":
            org = ent.text
            break

    organizations.append(org)

    # -------- Risk Detection --------
    label = "normal"

    lower_text = text_str.lower()

    for word in risk_words:
        if word in lower_text:
            label = "high_risk"
            break

    risk_labels.append(label)

    # -------- Year Extraction --------
    try:
        year = pd.to_datetime(date).year
    except:
        year = None

    years.append(year)


df["organization"] = organizations
df["risk_level"] = risk_labels
df["year"] = years

print("Saving enriched dataset")

df.to_csv("data/processed/legal_cases_enriched.csv", index=False)

print("Analytics dataset created successfully")
