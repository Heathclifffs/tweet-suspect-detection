import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(predictions_path: str, metrics_path: str):
    df = pd.read_csv(predictions_path)
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées dans {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    evaluate("models/test_predictions.csv", "models/metrics.json")
