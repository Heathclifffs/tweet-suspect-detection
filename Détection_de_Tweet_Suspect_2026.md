# Examen Final \- Détection de Tweets Suspects avec GiT et DVC

## Contexte

Les réseaux sociaux constituent une source majeure d’information et de communication. Cependant, ils peuvent également être utilisés pour diffuser des contenus suspects, haineux, offensants ou trompeurs. Dans ce projet, vous devrez développer une solution complète de classification automatique permettant d’identifier les tweets suspects à partir d’un jeu de données fourni.

L’objectif de cette évaluation est de mettre en pratique l’ensemble du cycle de vie d’un projet de Machine Learning, depuis l’exploration des données jusqu’au déploiement, en intégrant les bonnes pratiques de gestion des données et des modèles à l’aide de **DVC (Data Version Control)**.

# Objectifs pédagogiques

À l’issue de ce projet, l’étudiant devra être capable de :

1. Réaliser une analyse exploratoire de données textuelles.  
2. Prétraiter et représenter des données textuelles sous forme exploitable par des algorithmes de Machine Learning.  
3. Construire, comparer et évaluer plusieurs modèles de classification.  
4. Gérer le déséquilibre des classes.  
5. Mettre en place un pipeline reproductible avec DVC.  
6. Déployer un modèle sous forme d’application ou d’API.  
7. Documenter un projet de Data Science de manière professionnelle.

# Dataset

Le jeu de données est disponible à l’adresse suivante :

[https://drive.google.com/file/d/1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk/view?usp=share\_link](https://drive.google.com/file/d/1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk/view?usp=share_link)

Le dataset contient des tweets ainsi qu’une étiquette indiquant si le contenu est considéré comme suspect ou non.

# Travail demandé

## Partie 1 \- Exploration et prétraitement des données (15 points)

### Analyse exploratoire

* Charger les données avec Pandas.  
* Identifier les variables disponibles.  
* Vérifier les valeurs manquantes.  
* Étudier la distribution des classes.  
* Produire des visualisations pertinentes.

### Prétraitement du texte

Mettre en œuvre plusieurs opérations de nettoyage :

* Conversion en minuscules.  
* Suppression des caractères spéciaux.  
* Suppression des URLs.  
* Suppression des stop words.  
* Lemmatisation ou stemming (optionnel).

Documenter les choix effectués.

## Partie 2 \- Gestion des données avec DVC (15 points)

L'utilisation de **GiT** et **DVC est obligatoire**.

Vous devez :

### Versionner le dataset

* Initialiser DVC dans le projet.  
* Ajouter le dataset à DVC.  
* Configurer un stockage distant (local, Google Drive, S3, etc.).

### Construire un pipeline DVC

Créer un pipeline comprenant au minimum :

1. Prétraitement des données.  
2. Entraînement du modèle.  
3. Évaluation du modèle.

### Reproductibilité

Le projet doit permettre à une autre personne de reproduire les résultats à l’aide des commandes :

dvc pull  
dvc repro

Les fichiers DVC et les dépendances devront être présents dans le dépôt GitHub.

## Partie 3 \- Représentation des données (15 points)

Transformer les tweets en représentations numériques.

Choisir au moins une des approches suivantes :

### Approches classiques

* Bag of Words  
* TF-IDF

### Approches avancées

* Word2Vec  
* GloVe  
* FastText  
* Sentence Transformers  
* BERT

Justifier le choix retenu.

### Optionnel

Appliquer une technique de réduction de dimension :

* PCA  
* t-SNE  
* UMAP

## Partie 4 \- Construction des modèles (20 points)

### Gestion du déséquilibre des classes

Tester au moins une stratégie parmi :

* Oversampling (SMOTE)  
* Undersampling  
* Class Weights

### Comparaison des modèles

Évaluer au minimum **trois algorithmes différents** parmi :

* Logistic Regression  
* Naive Bayes  
* SVM  
* Random Forest  
* XGBoost  
* LSTM  
* BERT

Comparer leurs performances.

## Partie 5 \- Entraînement et validation (10 points)

* Séparer les données en ensembles d’entraînement et de test.  
* Utiliser une validation croisée adaptée.  
* Présenter les métriques obtenues.

Métriques minimales :

* Accuracy  
* Precision  
* Recall  
* F1-Score

## Partie 6 \- Évaluation et optimisation (10 points)

### Évaluation

Produire :

* Matrice de confusion  
* Courbe ROC  
* AUC

### Optimisation

Effectuer une recherche d’hyperparamètres :

* Grid Search ou  
* Random Search

Présenter les paramètres optimaux obtenus.

## Partie 7 \- Déploiement (10 points)

Choisir une des deux solutions :

### Option A : Application Streamlit

Créer une interface permettant :

* Saisir un tweet.  
* Obtenir la prédiction.  
* Afficher la probabilité associée.

### Option B : API

Développer une API avec :

* Flask ou  
* FastAPI

L’API doit permettre d’envoyer un texte et de recevoir la prédiction.

## Partie 8 \- Documentation et rapport (5 points)

Rédiger un rapport contenant :

### Introduction

* Présentation du problème.

### Méthodologie

* Prétraitement.  
* Représentation.  
* Modèles utilisés.

### Résultats

* Comparaison des performances.  
* Visualisations.

### Discussion

* Limites du projet.  
* Difficultés rencontrées.  
* Perspectives d'amélioration.

# Livrables

## 1\. Dépôt GitHub

Le dépôt devra contenir :

* Code source complet.  
* README détaillé.  
* Pipeline DVC fonctionnel.  
* Fichiers DVC nécessaires.  
* Notebooks d'analyse.

## 2\. Rapport PDF

Le rapport final devra être remis au format PDF.

## 3\. Captures d’écran

Inclure des captures montrant :

* Analyse exploratoire.  
* Pipeline DVC.  
* Matrice de confusion.  
* Courbe ROC.  
* Interface Streamlit ou tests API.

# Critères d’évaluation

| Critère | Points |
| :---- | ----: |
| Exploration et prétraitement | 15 |
| Utilisation de DVC | 15 |
| Représentation des données | 15 |
| Construction des modèles | 20 |
| Validation et évaluation | 10 |
| Optimisation | 10 |
| Déploiement | 10 |
| Documentation et rapport | 5 |
| **Total** | **100** |

# Bonus (+5 points)

Les étudiants pourront obtenir jusqu’à 5 points supplémentaires pour :

* Utilisation de Transformers avancés.  
* Déploiement sur le cloud.  
* Intégration CI/CD.  
* Dashboard de monitoring.  
* Expérimentation avec MLflow en complément de DVC.

