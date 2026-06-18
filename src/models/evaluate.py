import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def evaluate(predictions_path: str, output_dir: str):
    df = pd.read_csv(predictions_path)
    models = df["model"].unique()

    all_metrics = {}

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        subset = df[df["model"] == name]
        y_true = subset["y_true"]
        y_pred = subset["y_pred"]

        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }
        all_metrics[name] = metrics

        cm = confusion_matrix(y_true, y_pred)
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name}\nAcc: {metrics['accuracy']} | F1: {metrics['f1_score']}")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for name in models:
        subset = df[df["model"] == name]
        y_true = subset["y_true"]
        y_proba = subset["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("Taux de faux positifs")
    ax_roc.set_ylabel("Taux de vrais positifs")
    ax_roc.set_title("Courbes ROC")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("Métriques par modèle :")
    for name, m in all_metrics.items():
        print(f"  {name}: {m}")
    print(f"Graphiques sauvegardés dans {output_dir}/")


if __name__ == "__main__":
    evaluate("models/test_predictions.csv", "models")
