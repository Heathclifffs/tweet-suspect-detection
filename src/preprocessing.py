import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "can't": "cannot", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "it'll": "it will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have",
    "let's": "let us", "that's": "that is", "who's": "who is",
    "what's": "what is", "where's": "where is", "how's": "how is",
    "there's": "there is", "here's": "here is",
}


def _get_wordnet_pos(word: str) -> str:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_map = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|bit\.ly\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#\d+;", " ", text)
    text = re.sub(r"\b(\w)\1{2,}\b", r"\1\1", text)
    text = re.sub(r"rt\s", "", text)

    words = text.split()
    words = [CONTRACTIONS.get(w, w) for w in words]
    text = " ".join(words)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    words = [LEMMATIZER.lemmatize(w, _get_wordnet_pos(w)) for w in words]

    return " ".join(words).strip()


def preprocess(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    df["clean_message"] = df["message"].fillna("").apply(clean_text)
    df = df[df["clean_message"] != ""].reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Prétraitement terminé -> {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Tweets vides supprimés : {len(pd.read_csv(input_path)) - len(df)}")


if __name__ == "__main__":
    preprocess("data/raw/tweets.csv", "data/processed/tweets_clean.csv")
