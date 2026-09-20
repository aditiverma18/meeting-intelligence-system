import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv(
    "data/ami_dialogue_dataset.csv"
)

print("Total examples:", len(df))


# -----------------------------------------
# 2. INPUT AND TARGET
# -----------------------------------------

X = df["text"]
y = df["label"]


# -----------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# 5. TF-IDF
# -----------------------------------------

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF shape:")
print("Training:", X_train_tfidf.shape)
print("Testing :", X_test_tfidf.shape)


# -----------------------------------------
# 6. LOGISTIC REGRESSION
# -----------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)


# -----------------------------------------
# 7. PREDICTION
# -----------------------------------------

y_pred = model.predict(X_test_tfidf)


# -----------------------------------------
# 8. EVALUATION
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)