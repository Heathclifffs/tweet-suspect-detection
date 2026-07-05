---
title: Detection de Tweets Suspects
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# Détection de Tweets Suspects

Classification automatique de tweets suspects (haineux, offensants, trompeurs) avec un pipeline ML reproductible basé sur DVC.

Application en ligne sur Hugging Face Spaces :

**https://huggingface.co/spaces/yipene/tweet-suspect-detection**

Deploiement automatise :

```bash
uv run bash scripts/deploy_hf_space.sh
```

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
│   ├── preprocessing.py   # Nettoyage du texte (10 étapes)
│   ├── models/
│   │   ├── train.py       # Entraînement LR + NB + RF (pipeline DVC)
│   │   ├── evaluate.py    # Métriques, matrices confusion, ROC
│   │   ├── train_bert.py  # Bonus : fine-tuning DistilBERT
│   │   └── train_with_mlflow.py  # Bonus : tracking MLflow
│   └── deploy/
│       ├── streamlit_app.py # Interface Streamlit (4 onglets)
│       └── api.py           # API FastAPI
├── models/                 # Modèles sauvegardés + métriques
│   ├── metrics.json        # Métriques LR, NB, RF
│   ├── bert_metrics.json   # Métriques BERT (bonus)
│   └── *.pkl / *.npy / *.csv  # Modèles, prédictions, matrices
├── reports/
│   ├── figures/            # Visualisations générées
│   ├── rapport.md          # Source du rapport
│   ├── rapport.pdf         # Rapport final (PDF)
│   └── rapport.docx        # Rapport final (DOCX)
├── .github/workflows/
│   └── ci.yml              # CI/CD : dvc repro automatique
├── dvc.yaml                # Pipeline DVC reproductible
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
#    Le modèle sera automatiquement chargé par Streamlit
uv add torch
uv run python src/models/train_bert.py

# 9. MLflow (tracking des expérimentations)
#    L'onglet MLflow dans Streamlit affiche les runs automatiquement
rm -rf mlruns
uv run python src/models/train_with_mlflow.py
uv run mlflow ui            # Interface web sur http://localhost:5000

# 10. CI/CD — automatique à chaque push (GitHub Actions)
# Voir .github/workflows/ci.yml

# 11. Hugging Face Spaces — déploiement cloud
# Créer un Space sur huggingface.co/new-space, SDK Streamlit
```

## Application Streamlit

L'interface propose 4 onglets :

| Onglet | Description |
|--------|-------------|
| **Prediction** | Saisir un tweet ou tester un exemple. Resultats des **4 modeles** (LR, NB, RF, BERT) avec confiance. |
| **Tableau de bord** | Selecteur de modele (dont BERT), matrice de confusion interactive, courbes ROC, comparaison des performances, feature importance (Plotly dynamique). |
| **MLflow** | Affichage des dernieres runs MLflow, lancement de l'interface web. |
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

### Approches classiques (TF-IDF + sklearn)

| Modèle | Accuracy | Precision | Recall | F1-Score | CV 5-fold (F1) |
|--------|----------|-----------|--------|----------|----------------|
| Logistic Regression | 97.04% | 98.10% | 98.61% | 98.36% | 98.31% ± 0.18% |
| Naive Bayes | 92.76% | 92.57% | 99.95% | 96.12% | 96.10% ± 0.10% |
| Random Forest | 97.54% | 98.16% | 99.12% | 98.64% | 98.52% ± 0.15% |

Représentation : **TF-IDF** (5000 features, unigrams + bigrams).  
Gestion du déséquilibre : `class_weight="balanced"`.  
Validation croisée **5-fold** avec stratification.  
Grid Search : C ∈ {0.01, 0.1, 1, 10, 100} pour LR, n_estimators ∈ {50, 100, 200} / max_depth ∈ {5, 10, 20, None} pour RF.

### Bonus : DistilBERT

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **DistilBERT** (1 epoch) | **98.24%** | **98.85%** | **99.19%** | **99.02%** |

Meilleur score global (F1=99.02%), dépasse le Random Forest (98.64%). Intégré à Streamlit comme 4e modèle.

## API FastAPI

```bash
uv run uvicorn src.deploy.api:app --reload
```

## Rapport

Le rapport final est disponible dans `reports/rapport.pdf`.

## Bonus

### B.1 — BERT (Transformers)

**DistilBERT** fine-tune sur le dataset avec `transformers` + `torch`.

| Métrique | Valeur | vs Random Forest |
|----------|--------|------------------|
| Accuracy | 98.24% | +0.70% |
| Precision | 98.85% | +0.69% |
| Recall | 99.19% | +0.07% |
| **F1-Score** | **99.02%** | **+0.38%** |

**Intégration Streamlit** :
- **Prediction** : 4e modèle disponible (chargé automatiquement si présent)
- **Dashboard** : matrice de confusion interactive, courbe ROC (AUC=0.98), métriques comparées aux 3 modèles sklearn
- **Entraînement** : sauvegarde automatique des prédictions, matrice de confusion et courbe ROC pour le dashboard

```bash
# Installation (une fois)
uv add torch

# Entraînement (~6 min CPU pour 1 epoch)
uv run python src/models/train_bert.py

# Voir les métriques
cat models/bert_metrics.json

# Lancer Streamlit (BERT chargé automatiquement)
uv run streamlit run src/deploy/streamlit_app.py
```

### B.5 — MLflow (Tracking des expérimentations)

Chaque entrainement est tracké automatiquement par **MLflow** :

| Ce qui est loggé | Détail |
|-----------------|--------|
| **Hyperparamètres** | `model_name`, `vectorizer`, + tous les params de `get_params()` (C, n_estimators, max_depth, etc.) |
| **Métriques** | accuracy, precision, recall, f1_score |
| **Artéfacts** | Modèle sérialisé (`model_logistic_regression`, etc.) |

**Intégration Streamlit** : onglet dédié affichant la liste des dernières runs avec leurs métriques, bouton pour lancer l'interface web.

```bash
# Lancer le tracking (3 runs : LR, NB, RF)
rm -rf mlruns                  # Départ propre
uv run python src/models/train_with_mlflow.py

# Interface web
uv run mlflow ui               # http://localhost:5000

# Via Streamlit (onglet MLflow)
uv run streamlit run src/deploy/streamlit_app.py
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
| Scikit-learn | Modélisation ML (LR, NB, RF) |
| Transformers / torch | DistilBERT (bonus B.1) |
| MLflow | Tracking des expérimentations (bonus B.5) |
| NLTK | NLP (stop words, lemmatisation) |
| Matplotlib / Seaborn / WordCloud | Visualisations |
| Plotly | Graphiques interactifs (dashboard Streamlit) |
| DVC | Pipeline ML reproductible |
| Git | Versionnement du code et des données |
| Streamlit | Interface utilisateur (4 onglets) |
| FastAPI | API REST |
| GitHub Actions | CI/CD (bonus B.3) |
| Hugging Face Spaces | Déploiement cloud (bonus B.2) |

## Auteur

**Yipene Harold Ezekiel BASSOLE** — M2 FD&IA

Projet réalisé dans le cadre du *Construction de Modèles et Déploiement*.

## Lien

[https://github.com/Heathclifffs/tweet-suspect-detection](https://github.com/Heathclifffs/tweet-suspect-detection)
# tweet-suspect-detection
