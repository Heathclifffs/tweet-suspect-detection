import gdown
import os

DATA_RAW_DIR = "data/raw"
FILE_ID = "1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk"
OUTPUT_PATH = os.path.join(DATA_RAW_DIR, "tweets.csv")


def download_dataset(force: bool = False):
    if os.path.exists(OUTPUT_PATH) and not force:
        print(f"Le fichier {OUTPUT_PATH} existe déjà.")
        reponse = input("Voulez-vous le re-télécharger pour le mettre à jour ? (o/N) : ")
        if reponse.lower() != "o":
            print("Téléchargement ignoré.")
            return

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    gdown.download(id=FILE_ID, output=OUTPUT_PATH, quiet=False)
    print(f"Dataset téléchargé dans {OUTPUT_PATH}")


if __name__ == "__main__":
    download_dataset()
