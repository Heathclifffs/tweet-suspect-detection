import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
}


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
    pd.DataFrame({"clean_message": X_test, "label": y_test}).to_csv(
        f"{model_dir}/test_data.csv", index=False
    )

    results = []

    for name, model in MODELS.items():
        print(f"Entraînement : {name}...")
        model.fit(X_train_vec, y_train)
        joblib.dump(model, f"{model_dir}/{name}.pkl")

        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        results.append(pd.DataFrame({
            "model": name,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }))

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(f"{model_dir}/test_predictions.csv", index=False)
    print(f"Modèles entraînés et sauvegardés dans {model_dir}/")


if __name__ == "__main__":
    train("data/processed/tweets_clean.csv", "models")
