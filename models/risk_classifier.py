import pandas as pd
import re
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset")

df = pd.read_csv("data/raw/legal_cases.csv")

print("Total documents:", len(df))


# ---------------------------------------------------
# Step 1: generate weak labels using keywords
# ---------------------------------------------------

print("Generating weak labels")


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
noise_indices = df.sample(frac=noise_ratio, random_state=42).index

for idx in noise_indices:
    df.loc[idx, "risk_level"] = np.random.choice(["high_risk", "normal"])

# ---------------------------------------------------
# Step 2: remove leakage keywords from text
# ---------------------------------------------------

print("\nRemoving label leakage")


def remove_keywords(text):

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
        r"insider trading"
    ]

    for pattern in leakage_patterns:
        text = re.sub(pattern, "", text)

    return text


df["clean_text"] = df["text"].apply(remove_keywords)


# ---------------------------------------------------
# Step 3: vectorize text
# ---------------------------------------------------

print("\nVectorizing legal text")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["risk_level"]

print("Feature matrix shape:", X.shape)


# ---------------------------------------------------
# Step 4: train test split
# ---------------------------------------------------

print("\nSplitting dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# Step 5: train model
# ---------------------------------------------------

print("\nTraining SVM model")

model = LinearSVC(class_weight="balanced")

model.fit(X_train, y_train)


# ---------------------------------------------------
# Step 6: evaluate
# ---------------------------------------------------

print("\nRunning predictions")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel accuracy:", round(accuracy,3))

print("\nClassification report:\n")

print(classification_report(y_test, y_pred))