# Détection de Tweets Suspects

Classification automatique de tweets suspects (haineux, offensants, trompeurs) avec un pipeline ML reproductible basé sur DVC.

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

## Installation

### Prérequis

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets)

### Setup

```bash
# Cloner le dépôt
git clone <repo-url>
cd tweet-suspect-detection

# Créer l'environnement et installer les dépendances
uv sync

# Télécharger le dataset (avec confirmation interactive)
uv run python src/download.py
```

## Guide de démarrage rapide

```bash
# 1. Installer l'environnement
uv sync

# 2. Lancer l'analyse exploratoire (notebooks)
uv run jupyter lab

# 3. Exécuter tout le pipeline ML
uv run dvc repro

# 4. Voir les métriques
cat models/metrics.json
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
- **0** : Non suspect
- **1** : Suspect

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

### Modèles et résultats

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 97.04% | 98.10% | 98.61% | 98.36% |
| Naive Bayes | 92.76% | 92.57% | 99.95% | 96.12% |
| Random Forest | 97.54% | 98.16% | 99.12% | 98.64% |

Représentation : TF-IDF (5000 features).  
Gestion du déséquilibre : `class_weight="balanced"`.

## Déploiement

Deux options prévues :

### Option A : Streamlit
```bash
uv run streamlit run src/deploy/streamlit_app.py
```

### Option B : API FastAPI
```bash
uv run uvicorn src.deploy.api:app --reload
```

## Rapport

Le rapport final est disponible dans `reports/rapport.pdf`.

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

## Auteurs

Projet réalisé dans le cadre du M2 FD&IA — *Construction de Modèles et Déploiement*.
