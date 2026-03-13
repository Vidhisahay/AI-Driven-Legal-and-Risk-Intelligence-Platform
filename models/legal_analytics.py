import pandas as pd

print("Loading datasets")

cases = pd.read_csv("data/raw/legal_cases.csv")
entities = pd.read_csv("data/processed/legal_entities.csv")

# --------------------------------
# Risk distribution
# --------------------------------

print("Generating risk distribution")

cases["risk"] = cases["text"].apply(
    lambda x: "high_risk" if "fraud" in str(x).lower() or "violation" in str(x).lower()
    else "normal"
)

risk_distribution = cases["risk"].value_counts().reset_index()
risk_distribution.columns = ["risk_level", "count"]

risk_distribution.to_csv(
    "analytics/risk_distribution.csv",
    index=False
)

# --------------------------------
# Cases by year
# --------------------------------

print("Generating cases by year")

cases["date"] = pd.to_datetime(cases["date"], errors="coerce")
cases["year"] = cases["date"].dt.year

cases_by_year = cases["year"].value_counts().reset_index()
cases_by_year.columns = ["year", "case_count"]

cases_by_year.to_csv(
    "analytics/cases_by_year.csv",
    index=False
)

# --------------------------------
# Organization mentions
# --------------------------------

print("Generating organization mentions")

org_series = entities["organizations"].str.split(", ")
org_exploded = org_series.explode()

top_orgs = org_exploded.value_counts().reset_index().head(20)
top_orgs.columns = ["organization", "mentions"]

top_orgs.to_csv(
    "analytics/organization_mentions.csv",
    index=False
)

print("Analytics datasets created")