# 🌿 EcoPredict AI - Prédiction d'Émissions de CO2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Sklearn-orange)

**EcoPredict AI** est une application web de Data Science permettant de prédire les émissions de CO2 (g/km) des véhicules.

Le cœur du projet repose sur un pipeline de données complet : les données brutes sont traitées et fusionnées via un Notebook Jupyter pour générer le dataset final utilisé par les modèles d'Ensemble Learning (Random Forest, XGBoost, MLP).

## 🚀 Fonctionnalités

- **Pipeline de Données Automatisé :** Construction du dataset final à partir de sources brutes multiples.
- **Interface Intuitive :** Formulaire web interactif pour saisir les caractéristiques du véhicule.
- **Multi-Modèles :** Comparaison des prédictions entre trois algorithmes distincts (RF, XGBoost, MLP).
- **Logique Métier :** Gestion automatique des spécificités (ex: véhicules électriques).

## 📂 Structure du Projet

L'architecture met en avant la séparation entre les sources brutes et les données exploitables.

```text
EcoPredict-AI/
├── source_de_donnees/     # 🚨 COEUR DU PROJET : Contient les fichiers bruts originaux
│                          # C'est à partir d'ici que tout se construit.
├── notebook_code.ipynb    # Pipeline ETL & Entraînement :
│                          # 1. Lit les fichiers de 'source_de_donnees/'
│                          # 2. Nettoie et fusionne les données
│                          # 3. Génère 'dataset_final.csv'
│                          # 4. Entraîne et exporte les modèles (.pkl)
├── dataset_final.csv      # Dataset consolidé (Généré par le notebook)
├── app.py                 # Application Flask (Utilise le dataset final et les modèles)
├── pickle/                # Modèles entraînés prêts à l'emploi
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── mlp.pkl
├── static/                # Assets (CSS, Images)
├── templates/             # Vues HTML
└── README.md              # Documentation
```

## 📖 Guide d'Utilisation et Interface

Une fois l'application lancée via la commande `python app.py`, suivez ces étapes pour effectuer une prédiction :

### 1. Accès à l'interface

Ouvrez votre navigateur web et accédez à l'adresse locale indiquée dans votre terminal (par défaut) :

> [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Saisie des caractéristiques

L'interface présente un formulaire structuré pour renseigner les détails du véhicule (Marque, Transmission, Puissance, etc.).

**⚠️ Règles de gestion automatique :**
L'application intègre une logique JavaScript pour garantir la cohérence des données :

* **Véhicules Électriques :** Si vous sélectionnez le carburant  *"Electrique"* .
* **Véhicules Hybrides :** Si vous sélectionnez *"Oui"* dans la liste déroulante  *"Hybride"* .

> **Effet :** Dans ces deux cas, les champs **"Consommation Urbaine"** et **"Consommation Autoroute"** sont automatiquement **fixés à 0** et  **grisés (désactivés)** . Il est inutile de les remplir manuellement, l'algorithme gère cette spécificité.

### 3. Lecture des Résultats

Après avoir cliqué sur  **"Lancer la prédiction"** , la page défile automatiquement vers la section des résultats :

* **Prédiction Principale :** Moyenne pondérée des modèles pour une estimation fiable.
* **Détail par Modèle :** Affichage transparent des résultats individuels de  *Random Forest* , *XGBoost* et *MLP* pour comparaison.
