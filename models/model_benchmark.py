import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

print("Loading dataset")

df = pd.read_csv("data/raw/legal_cases.csv")

print("Total documents:", len(df))


# ----------------------------
# Generate labels
# ----------------------------

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


# ----------------------------
# Remove leakage keywords
# ----------------------------

def remove_keywords(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    for word in risk_keywords:
        text = re.sub(word, "", text)

    return text


df["clean_text"] = df["text"].apply(remove_keywords)


# ----------------------------
# Vectorization
# ----------------------------

print("\nVectorizing text")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["risk_level"]


# ----------------------------
# Train test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------
# Model benchmarking
# ----------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = {}

print("\nTraining models...\n")

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    results[name] = acc

    print(name, "accuracy:", round(acc, 3))
    print("-"*30)


# ----------------------------
# Final comparison
# ----------------------------

print("\nModel Comparison Results\n")

for model, score in results.items():
    print(model, ":", round(score, 3))


best_model = max(results, key=results.get)

print("\nBest performing model:", best_model)