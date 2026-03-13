import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_FILE = "data/raw/legal_cases.csv"
OUTPUT_FILE = "data/processed/legal_cases_features.csv"

print("Loading legal dataset")

df = pd.read_csv(INPUT_FILE)

print("Total documents:", len(df))

nlp = spacy.load("en_core_web_sm")


def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


print("Cleaning legal documents")

df["clean_text"] = df["text"].apply(clean_text)


print("Running spaCy processing")

def process_text(text):

    doc = nlp(text)

    tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)

    return " ".join(tokens)


df["processed_text"] = df["clean_text"].apply(process_text)


print("Creating TF IDF features")

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df["processed_text"])

tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


print("Combining dataset")

final_df = pd.concat([df[["case_name", "date", "court"]], tfidf_df], axis=1)


final_df.to_csv(OUTPUT_FILE, index=False)

print("Processed dataset saved")
print("Feature columns:", len(tfidf_df.columns))