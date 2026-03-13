import pandas as pd

print("Loading entity dataset")

entities = pd.read_csv("data/processed/legal_entities.csv")

print("Loading case dataset")

cases = pd.read_csv("data/raw/legal_cases.csv")


print("Merging datasets")

data = entities.merge(
    cases[["text"]],
    left_on="case_id",
    right_index=True
)

# ------------------------------------------------
# Example analytics: top organizations mentioned
# ------------------------------------------------

print("Computing top organizations")

org_series = data["organizations"].str.split(", ")

org_exploded = org_series.explode()

top_orgs = (
    org_exploded.value_counts()
    .head(20)
    .reset_index()
)

top_orgs.columns = ["organization", "mentions"]

print(top_orgs)

top_orgs.to_csv(
    "data/processed/top_organizations.csv",
    index=False
)

print("Saved analytics dataset")