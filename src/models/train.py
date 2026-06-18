import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    joblib.dump(model, f"{model_dir}/model.pkl")
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")

    pd.DataFrame({"y_true": y_test, "y_pred": model.predict(X_test_vec)}).to_csv(
        f"{model_dir}/test_predictions.csv", index=False
    )
    print(f"Modèle entraîné et sauvegardé dans {model_dir}/")


if __name__ == "__main__":
    train("data/processed/tweets_clean.csv", "models")
