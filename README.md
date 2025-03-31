# GRETSI-2025-submission

# Algorithme EM stochastique pour modèles d’espace d’états linéaires en bruits impulsifs : application à la radio-interférométrie

Ce dépôt contient le code source associé à l'article soumis au GRETSI 2025. L’objectif est de proposer un algorithme Expectation-Maximization (EM) stochastique adapté aux modèles d’espace d’états linéaires soumis à du bruit impulsif, avec une application concrète à la reconstruction d’images en radio-interférométrie.

## 🧪 Reproductibilité

Les résultats suivants de l'article peuvent être reproduits :

- **Figure 1** : Résultat visuel de la reconstruction dynamique
- **Ligne 3 de la Table 1** : Résultats quantitatifs pour un cas spécifique

## ⚙️ Installation

### Option 1 : Environnement Python avec `requirements.txt`

```bash
./install.sh
```

## ⚙️ Installation

Cloner ce dépôt et exécuter le script suivant pour créer un environnement Python virtuel et installer les dépendances :  
`./install.sh`  
Remarque : si vous préférez utiliser Conda, adaptez le script à partir d’un fichier `environment.yml`.

## ▶️ Exécution

Lancez l'exécution principale avec :  
`./run.sh`  
Cela lance le script `main.py`, qui effectue une itération complète de l'algorithme. Le temps d'exécution est estimé à **12 à 15 minutes par itération** sur la configuration décrite ci-dessous.

## 📂 Structure du dépôt

em-impulsif-gretsi2025/
├── README.md               # Ce fichier
├── LICENSE                 # Licence du projet
├── requirements.txt        # Dépendances Python
├── install.sh              # Script d'installation
├── run.sh                  # Script d'exécution
├── src/                    # Code source de l'algorithme
│   ├── main.py             # Point d'entrée
│   ├── em.py               # Algorithme EM
│   ├── model.py            # Modèle d’observation
│   └── utils.py            # Fonctions utilitaires
├── data/                   # Données d'entrée simulées
│   └── visibilities.npz
├── results/                # Résultats produits (figures, tableaux)
│   ├── fig1.png
│   └── table1_row3.txt
└── notebooks/              # Notebook original (facultatif)
    └── em_notebook_original.ipynb

