import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc as sk_auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch


MODEL_NAME = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1_score": f1_score(labels, predictions, zero_division=0),
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


def train_bert(input_path: str, model_dir: str):
    df = pd.read_csv(input_path)
    X = df["clean_message"].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir=f"{model_dir}/bert_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(f"{model_dir}/bert_model")
    tokenizer.save_pretrained(f"{model_dir}/bert_model")

    with open(f"{model_dir}/bert_metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    test_preds = trainer.predict(test_dataset)
    y_pred = np.argmax(test_preds.predictions, axis=-1)
    y_proba = torch.softmax(torch.tensor(test_preds.predictions), dim=-1)[:, 1].numpy()

    pred_df = pd.DataFrame({
        "y_true": test_preds.label_ids,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model": "bert",
    })
    pred_df.to_csv(f"{model_dir}/bert_predictions.csv", index=False)

    cm = confusion_matrix(test_preds.label_ids, y_pred)
    np.save(f"{model_dir}/bert_confusion_matrix.npy", cm)

    fpr, tpr, _ = roc_curve(test_preds.label_ids, y_proba)
    roc_auc_val = sk_auc(fpr, tpr)
    np.savez(f"{model_dir}/bert_roc.npz", fpr=fpr, tpr=tpr, auc=roc_auc_val)

    print(f"Modele BERT entraine et sauvegarde dans {model_dir}/bert_model")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train_bert("data/processed/tweets_clean.csv", "models")
