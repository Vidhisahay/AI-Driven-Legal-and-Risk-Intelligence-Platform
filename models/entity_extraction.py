import pandas as pd
import spacy
from tqdm import tqdm

print("Loading spaCy model")

nlp = spacy.load("en_core_web_sm")

print("Loading dataset")

df = pd.read_csv("data/raw/legal_cases.csv")

print("Total documents:", len(df))


results = []

print("Extracting entities...\n")

for index, row in tqdm(df.iterrows(), total=len(df)):

    text = row["text"]

    if not isinstance(text, str):
        continue

    doc = nlp(text[:5000])  # limit size for speed

    organizations = []
    persons = []
    locations = []

    for ent in doc.ents:

        if ent.label_ == "ORG":
            organizations.append(ent.text)

        elif ent.label_ == "PERSON":
            persons.append(ent.text)

        elif ent.label_ == "GPE":
            locations.append(ent.text)

    results.append({
        "case_id": index,
        "organizations": ", ".join(set(organizations)),
        "persons": ", ".join(set(persons)),
        "locations": ", ".join(set(locations))
    })


entity_df = pd.DataFrame(results)

print("\nSaving extracted entities")

entity_df.to_csv("data/processed/legal_entities.csv", index=False)

print("Entity extraction completed")