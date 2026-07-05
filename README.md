---
title: Detection de Tweets Suspects
emoji: 🕵️
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.58.0
app_file: src/deploy/streamlit_app.py
pinned: false
---

# Détection de Tweets Suspects

Classification automatique de tweets suspects (haineux, offensants, trompeurs) avec un pipeline ML reproductible basé sur DVC.

[Deploiement sur Hugging Face Spaces](https://huggingface.co/new-space) : creer un Space, lier ce depot GitHub, et le SDK Streamlit sera automatiquement detecte.

## Contexte

Projet d'examen M2 FD&IA — *Construction de Modèles et Déploiement*.

L'objectif est de couvrir l'ensemble du cycle de vie d'un projet de Machine Learning : exploration, prétraitement, modélisation, évaluation, optimisation et déploiement.

## Structure du projet

```
tweet-suspect-detection/
├── data/
│   ├── raw/               # Données brutes (trackées par Git)
│   ├── processed/         # Données prétraitées (pipeline DVC)
│   └── interim/           # Données intermédiaires
├── notebooks/
│   ├── 01_eda.ipynb        # Analyse exploratoire des données
│   └── 02_modeling.ipynb   # Modélisation, comparaison, optimisation
├── src/
│   ├── download.py        # Téléchargement interactif du dataset
│   ├── preprocessing.py   # Nettoyage du texte
│   ├── features.py        # Représentation des données
│   ├── models/
│   │   ├── train.py       # Entraînement du modèle
│   │   └── evaluate.py    # Évaluation et métriques
│   └── deploy/
│       ├── streamlit_app.py # Interface Streamlit
│       └── api.py           # API FastAPI
├── models/                 # Modèles sauvegardés (pipeline DVC)
├── reports/
│   ├── figures/            # Visualisations générées
│   └── rapport.pdf         # Rapport final
├── dvc.yaml                # Pipeline DVC
├── dvc.lock                # Verrou DVC
├── pyproject.toml          # Dépendances (uv)
└── uv.lock                 # Lock uv
```

## Prérequis

- **Python ≥ 3.13**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire de paquets)

  ```bash
  # Installer uv (si pas déjà fait)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Installation et reproduction (etape par etape)

### Pipeline ML classique

```bash
# 1. Cloner le dépôt
git clone <repo-url>
cd tweet-suspect-detection

# 2. Créer l'environnement virtuel et installer les dépendances
uv sync

# 3. Télécharger le dataset (confirmation interactive demandée)
uv run python src/download.py

# 4. Exécuter tout le pipeline ML (preprocess → train → evaluate)
uv run dvc repro

# 5. Voir les métriques des modèles
cat models/metrics.json

# 6. Lancer l'application Streamlit (interface graphique)
uv run streamlit run src/deploy/streamlit_app.py

# 7. (Optionnel) Explorer les notebooks
uv run jupyter lab
```

### Bonus — reproduction intégrale

```bash
# 8. DistilBERT (nécessite torch, ~6 min sur CPU)
uv add torch
uv run python src/models/train_bert.py

# 9. MLflow (tracking des expérimentations)
rm -rf mlruns
uv run python src/models/train_with_mlflow.py
uv run mlflow ui            # Interface web sur http://localhost:5000

# 10. CI/CD — automatique à chaque push (GitHub Actions)
# Voir .github/workflows/ci.yml

# 11. Hugging Face Spaces — déploiement cloud
# Créer un Space sur huggingface.co/new-space, SDK Streamlit
```

## Application Streamlit

L'interface propose 3 onglets :

| Onglet | Description |
|--------|-------------|
| **Prediction** | Saisir un tweet ou tester un exemple. Resultats des 3 modeles (prediction + confiance). |
| **Tableau de bord** | Selecteur de modele, matrice de confusion interactive, courbes ROC, comparaison des performances, feature importance (plotly dynamique). |
| **Historique** | Toutes les analyses de la session avec horodatage. |

```bash
uv run streamlit run src/deploy/streamlit_app.py
```

## Pipeline DVC

Le pipeline est défini dans `dvc.yaml` et se compose de 3 étapes :

| Étape | Script | Entrée | Sortie |
|-------|--------|--------|--------|
| `preprocess` | `src/preprocessing.py` | `data/raw/tweets.csv` | `data/processed/tweets_clean.csv` |
| `train` | `src/models/train.py` | `data/processed/tweets_clean.csv` | `models/*.pkl`, `models/test_predictions.csv` |
| `evaluate` | `src/models/evaluate.py` | `models/test_predictions.csv` | `models/metrics.json`, `models/*.png` |

### Commandes DVC

```bash
# Exécuter le pipeline complet
uv run dvc repro

# Vérifier l'état du pipeline (étapes modifiées)
uv run dvc status

# Restaurer les fichiers depuis le cache DVC
uv run dvc checkout

# Pousser vers le stockage distant
uv run dvc push

# Tirer depuis le stockage distant (après clone)
uv run dvc pull
```

### Reproductibilité

Depuis un clone frais :
```bash
git clone <repo-url>
cd tweet-suspect-detection
uv sync
uv run dvc pull        # récupère les données du cache distant
uv run dvc repro        # reproduit tout le pipeline
```

## Dataset

Le jeu de données provient de [Google Drive](https://drive.google.com/file/d/1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk/view).
Il contient 60 000 tweets avec une étiquette binaire :
- **0** : Suspect (negatif, plaintes, contenu sensible)
- **1** : Non suspect (neutre, conversation normale)

Le dataset est tracké par Git (4.7 Mo). Pour le mettre à jour :
```bash
uv run python src/download.py
```

## Prétraitement

Les tweets sont nettoyés via `src/preprocessing.py` :
- Passage en minuscules
- Suppression des URLs et mentions (`@user`)
- Nettoyage des hashtags (garde le mot, enlève `#`)
- Expansion des contractions (`don't` → `do not`)
- Normalisation des répétitions (`goooood` → `good`)
- Suppression des caractères spéciaux, entités HTML
- Suppression des stop words (NLTK) + mots < 3 lettres
- **Lemmatisation** avec POS tagging (WordNet)

Environ 346 tweets vides supprimés sur 60 000.

## Modèles et résultats

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 97.04% | 98.10% | 98.61% | 98.36% |
| Naive Bayes | 92.76% | 92.57% | 99.95% | 96.12% |
| Random Forest | 97.54% | 98.16% | 99.12% | 98.64% |
| **DistilBERT** (bonus) | **98.24%** | **98.85%** | **99.19%** | **99.02%** |

Représentation : TF-IDF (5000 features).  
Gestion du déséquilibre : `class_weight="balanced"`.

## API FastAPI

```bash
uv run uvicorn src.deploy.api:app --reload
```

## Rapport

Le rapport final est disponible dans `reports/rapport.pdf`.

## Bonus

### B.1 — BERT (Transformers)

Un modele **DistilBERT** fine-tune sur le dataset (F1=99.02%, depasse Random Forest 98.64%) :

```bash
uv add torch
uv run python src/models/train_bert.py
cat models/bert_metrics.json   # Affiche les métriques
```

### B.5 — MLflow

Tracking des experimentations avec hyperparametres, metriques et artefacts :

```bash
rm -rf mlruns                  # Depart propre
uv run python src/models/train_with_mlflow.py
uv run mlflow ui               # http://localhost:5000
```

### CI/CD (GitHub Actions)

Le workflow `.github/workflows/ci.yml` execute automatiquement `dvc repro` a chaque push.

### Hugging Face Spaces

Ce depot est compatible avec [Hugging Face Spaces](https://huggingface.co/new-space) (SDK Streamlit). Creer un Space en liant ce depot GitHub.

## Tech Stack

| Outil | Usage |
|-------|-------|
| Python 3.13 | Langage |
| uv | Gestionnaire de dépendances |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Modélisation ML |
| NLTK | NLP (stop words) |
| Matplotlib / Seaborn / WordCloud | Visualisations |
| DVC | Pipeline ML reproductible |
| Git | Versionnement du code et des données |
| Streamlit / FastAPI | Déploiement |

## Auteur

**Yipene Harold Ezekiel BASSOLE** — M2 FD&IA

Projet réalisé dans le cadre du *Construction de Modèles et Déploiement*.

## Lien

[https://github.com/Heathclifffs/tweet-suspect-detection](https://github.com/Heathclifffs/tweet-suspect-detection)
# tweet-suspect-detection
