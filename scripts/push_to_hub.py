import os
import sys
from huggingface_hub import HfApi, login

LOCAL_MODEL_DIR = "models/bert_model"
HF_REPO = os.environ.get("HF_REPO_ID")

if not HF_REPO:
    print("Usage: HF_REPO_ID='votre-username/distilbert-tweet-suspect' uv run python scripts/push_to_hub.py")
    sys.exit(1)

token = os.environ.get("HF_TOKEN")
if not token:
    print("Erreur : HF_TOKEN non defini")
    sys.exit(1)

login(token=token)
api = HfApi()

api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=LOCAL_MODEL_DIR, repo_id=HF_REPO, repo_type="model")

print(f"Modele pousse sur Hugging Face Hub : https://huggingface.co/{HF_REPO}")
