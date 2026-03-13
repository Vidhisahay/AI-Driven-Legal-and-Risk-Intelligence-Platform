import pandas as pd
import re
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

print("Loading dataset")

df = pd.read_csv("data/raw/legal_cases.csv")

print("Total documents:", len(df))


# ---------------------------------------------------
# Generate labels
# ---------------------------------------------------

risk_keywords = [
    "fraud",
    "penalty",
    "violation",
    "crime",
    "corruption",
    "bribery",
    "insider trading",
    "money laundering"
]


def assign_risk(text):

    if not isinstance(text, str):
        return "normal"

    text = text.lower()

    for word in risk_keywords:
        if word in text:
            return "high_risk"

    return "normal"


df["risk_level"] = df["text"].apply(assign_risk)

print("\nRisk distribution:")
print(df["risk_level"].value_counts())


noise_ratio = 0.1
noise_idx = df.sample(frac=noise_ratio, random_state=42).index

for idx in noise_idx:
    df.loc[idx, "risk_level"] = np.random.choice(["high_risk", "normal"])

# ---------------------------------------------------
# Remove leakage words
# ---------------------------------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    leakage_patterns = [
        r"fraud\w*",
        r"penalt\w*",
        r"violat\w*",
        r"crime\w*",
        r"corrupt\w*",
        r"briber\w*",
        r"launder\w*",
        r"insider\w*",
        r"illegal\w*",
        r"prosecut\w*",
        r"convict\w*",
        r"defendant",
        r"charged",
        r"guilty"
    ]

    for pattern in leakage_patterns:
        text = re.sub(pattern, "", text)

    return text


df["clean_text"] = df["text"].apply(clean_text)


# ---------------------------------------------------
# Vectorize text
# ---------------------------------------------------

print("\nVectorizing text")

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["risk_level"]


# ---------------------------------------------------
# Model benchmarking
# ---------------------------------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = {}

print("\nRunning cross validation...\n")

for name, model in models.items():

    print("Evaluating:", name)

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    avg_score = scores.mean()
    std_score = scores.std()

    results[name] = avg_score

    print("Accuracy:", round(avg_score,3), "+/-", round(std_score,3))
    print("-"*30)


print("\nModel Comparison\n")

for model, score in results.items():
    print(model, ":", round(score,3))


best_model = max(results, key=results.get)

print("\nBest performing model:", best_model)