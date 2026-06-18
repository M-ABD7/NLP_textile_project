
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

TRAINING_DATA_PATH = "training_sentences.csv"


def load_training_data(path: str = TRAINING_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["problem"], test_size=test_size,
        random_state=random_state, stratify=df["problem"]
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return pipeline, acc, report


def predict_problem(pipeline: Pipeline, text: str, min_confidence: float = 0.20):
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    best_idx = proba.argmax()
    confidence = proba[best_idx]

    if confidence >= min_confidence:
        return classes[best_idx], confidence
    return None, confidence


if __name__ == "__main__":
    df = load_training_data()
    pipeline, acc, report = train_and_evaluate(df)

    print(f"Test accuracy: {acc:.2f}\n")
    print(report)

    print("--- Sample predictions on new sentences ---")
    samples = [
        "The fabric handle is too rough, customers complaining about feel",
        "Cotton turned slightly yellow during the bleaching step",
        "Some patches of the dyed batch are lighter than the rest",
    ]
    for s in samples:
        label, conf = predict_problem(pipeline, s)
        print(f"{s}\n  -> predicted: {label} (confidence={conf:.2f})\n")
