import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in STOP_WORDS)
    return text.strip()


def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df["clean_message"] = df["message"].fillna("").apply(clean_text)
    df = df[df["clean_message"] != ""].reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Prétraitement terminé -> {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    preprocess("data/raw/tweets.csv", "data/processed/tweets_clean.csv")
