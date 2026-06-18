# Skill - Projet Détection de Tweets Suspects

> Checklist de suivi pour l'examen final. On coche au fur et à mesure.

---

##  Bonnes pratiques transversales (à maintenir TOUT au long du projet)

### GIT
- [ ] **G.1** Commits réguliers avec messages explicites (en français ou anglais)
- [ ] **G.2** `.gitignore` propre (inclure `.dvc/`, `__pycache__/`, `.env`, `data/`, etc.)
- [ ] **G.3** Ne jamais versionner les données brutes ou les modèles volumineux dans Git
- [ ] **G.4** Faire un `git push` après chaque avancée significative

### Structure du projet
- [ ] **S.1** Arborescence claire et organisée (sources, notebooks, données, modèles, rapports)
- [ ] **S.2** Code modulaire : fonctions réutilisables dans des scripts `.py` séparés
- [ ] **S.3** Notebooks réservés à l'exploration / démo — pas la logique métier principale

### Environnement Python (avec uv)
- [ ] **E.1** Utiliser **uv** pour la gestion de l'environnement et des dépendances
- [ ] **E.2** Initialiser le projet avec `uv init` ou `uv venv`
- [ ] **E.3** Installer les dépendances avec `uv add <package>` (pas de `pip install`)
- [ ] **E.4** Générer/Maintenir `pyproject.toml` et `uv.lock` pour la reproductibilité
- [ ] **E.5** Utiliser `uv sync` pour reproduire l'environnement depuis un clone frais
- [ ] **E.6** Ne pas installer de packages en dehors de l'environnement uv

### Qualité du code
- [ ] **Q.1** Nommage cohérent des variables, fonctions, fichiers (snake_case)
- [ ] **Q.2** Fonctions avec docstrings (au moins 1 ligne décrivant le rôle)
- [ ] **Q.3** Ne pas laisser de cellules cassées ou de code mort dans les notebooks
- [ ] **Q.4** Utiliser des chemins relatifs ou des variables d'environnement, pas de chemins absolus

### Reproductibilité DVC
- [ ] **R.1** Les dépendances de chaque étape du pipeline DVC doivent être explicites
- [ ] **R.2** Les sorties de chaque étape doivent être versionnées par DVC
- [ ] **R.3** Tester `dvc repro` après chaque modification du pipeline

### Rapport et rendu
- [ ] **Z.1** Prendre des captures d'écran au fur et à mesure (ne pas tout laisser à la fin)
- [ ] **Z.2** Faire un point de validation avant de passer à la partie suivante

---

##  Partie 1 — Exploration et prétraitement (15 pts)

- [ ] **1.1** Charger les données avec Pandas
- [ ] **1.2** Identifier les variables disponibles
- [ ] **1.3** Vérifier les valeurs manquantes
- [ ] **1.4** Étudier la distribution des classes
- [ ] **1.5** Produire des visualisations pertinentes (histogrammes, wordcloud, etc.)
- [ ] **1.6** Conversion en minuscules
- [ ] **1.7** Suppression des caractères spéciaux
- [ ] **1.8** Suppression des URLs
- [ ] **1.9** Suppression des stop words
- [ ] **1.10** Lemmatisation ou stemming (optionnel mais recommandé)
- [ ] **1.11** Documenter les choix de prétraitement

##  Partie 2 — Gestion des données avec DVC (15 pts)

- [ ] **2.1** Initialiser Git dans le projet
- [ ] **2.2** Initialiser DVC (`dvc init`)
- [ ] **2.3** Ajouter le dataset à DVC (`dvc add`)
- [ ] **2.4** Configurer un stockage distant (local / Google Drive / S3)
- [ ] **2.5** Créer un pipeline DVC — étape **prétraitement**
- [ ] **2.6** Créer un pipeline DVC — étape **entraînement**
- [ ] **2.7** Créer un pipeline DVC — étape **évaluation**
- [ ] **2.8** Vérifier la reproductibilité (`dvc repro`)
- [ ] **2.9** Pousser les fichiers DVC et le code sur GitHub
- [ ] **2.10** Vérifier que `dvc pull` + `dvc repro` fonctionne depuis un clone frais

##  Partie 3 — Représentation des données (15 pts)

- [ ] **3.1** Choisir et implémenter **TF-IDF** ou **Bag of Words**
- [ ] **3.2** *(Bonus)* Implémenter une approche avancée : Word2Vec / GloVe / FastText / Sentence Transformers / BERT
- [ ] **3.3** Justifier le choix retenu
- [ ] **3.4** *(Optionnel)* Appliquer une réduction de dimension : PCA / t-SNE / UMAP

##  Partie 4 — Construction des modèles (20 pts)

- [ ] **4.1** Gérer le déséquilibre des classes (SMOTE / Undersampling / Class Weights)
- [ ] **4.2** Implémenter **Modèle 1** : Logistic Regression
- [ ] **4.3** Implémenter **Modèle 2** : Naive Bayes
- [ ] **4.4** Implémenter **Modèle 3** : Random Forest (ou SVM / XGBoost / LSTM / BERT)
- [ ] **4.5** Comparer les performances des 3 modèles

##  Partie 5 — Entraînement et validation (10 pts)

- [ ] **5.1** Séparer les données en train / test
- [ ] **5.2** Utiliser une validation croisée adaptée
- [ ] **5.3** Calculer Accuracy
- [ ] **5.4** Calculer Precision
- [ ] **5.5** Calculer Recall
- [ ] **5.6** Calculer F1-Score

##  Partie 6 — Évaluation et optimisation (10 pts)

- [ ] **6.1** Produire la matrice de confusion
- [ ] **6.2** Produire la courbe ROC
- [ ] **6.3** Calculer l'AUC
- [ ] **6.4** Effectuer une recherche d'hyperparamètres (Grid Search ou Random Search)
- [ ] **6.5** Présenter les paramètres optimaux obtenus

##  Partie 7 — Déploiement (10 pts)

- [ ] **7.1** Choisir : **Streamlit** OU **API (Flask/FastAPI)**
- [ ] **7.2** Développer l'interface
- [ ] **7.3** Tester que la prédiction fonctionne

##  Partie 8 — Documentation et rapport (5 pts)

- [ ] **8.1** Rédiger l'**Introduction** — présentation du problème
- [ ] **8.2** Rédiger la **Méthodologie** — prétraitement, représentation, modèles
- [ ] **8.3** Rédiger les **Résultats** — comparaison des performances, visualisations
- [ ] **8.4** Rédiger la **Discussion** — limites, difficultés, perspectives
- [ ] **8.5** Rédiger le **README.md** détaillé du dépôt GitHub
- [ ] **8.6** Générer le rapport au format **PDF**
- [ ] **8.7** Préparer les **captures d'écran** (EDA, pipeline DVC, matrice de confusion, courbe ROC, interface)

##  Bonus (+5 pts)

- [ ] **B.1** Utilisation de Transformers avancés (BERT, etc.)
- [ ] **B.2** Déploiement sur le cloud (Hugging Face Spaces / Render / Railway)
- [ ] **B.3** Intégration CI/CD (GitHub Actions)
- [ ] **B.4** Dashboard de monitoring
- [ ] **B.5** Expérimentation avec MLflow en complément de DVC

---

## Récapitulatif des notes

| Partie                            | Points | Obtenu |
| :-------------------------------- | :----: | :----: |
| 1\. Exploration & prétraitement   |  15    |        |
| 2\. DVC                           |  15    |        |
| 3\. Représentation des données    |  15    |        |
| 4\. Construction des modèles      |  20    |        |
| 5\. Validation & évaluation       |  10    |        |
| 6\. Optimisation                  |  10    |        |
| 7\. Déploiement                   |  10    |        |
| 8\. Documentation & rapport       |   5    |        |
| **Total**                         | **100**|        |
| Bonus                             |  +5    |        |
