import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def track(input_dir: str):
    with open(f"{input_dir}/metrics.json") as f:
        all_metrics = json.load(f)

    mlflow.set_experiment("tweet_suspect_detection")

    for name, metrics in all_metrics.items():
        with mlflow.start_run(run_name=name):
            model = joblib.load(f"{input_dir}/{name}.pkl")

            for k, v in model.get_params().items():
                mlflow.log_param(k, v)
            mlflow.log_param("vectorizer", "tfidf")

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"model_{name}")

        print(f"  {name} logged -> {metrics}")

    vectorizer = joblib.load(f"{input_dir}/vectorizer.pkl")
    mlflow.sklearn.log_model(vectorizer, "vectorizer")
    print("  vectorizer logged")


if __name__ == "__main__":
    track("models")
