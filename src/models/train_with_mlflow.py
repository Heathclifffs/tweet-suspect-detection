import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(input_path: str, model_dir: str):
    df = pd.read_csv(input_path)
    X = df["clean_message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    mlflow.set_experiment("tweet_suspect_detection")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Entrainement : {name}...")

            mlflow.log_param("model_name", name)
            mlflow.log_param("vectorizer", "tfidf")
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)

            model.fit(X_train_vec, y_train)
            joblib.dump(model, f"{model_dir}/{name}.pkl")

            y_pred = model.predict(X_test_vec)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"model_{name}")

            print(f"  {name} -> {metrics}")


if __name__ == "__main__":
    train("data/processed/tweets_clean.csv", "models")
