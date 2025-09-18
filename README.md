# Algorithme EM stochastique pour modèles d’espace d’états linéaires en bruits impulsifs : application à la radio-interférométrie

<hr>

**_Dépôt labelisé dans le cadre du [Label Reproductible du GRESTI'25](https://gretsi.fr/colloque2025/recherche-reproductible/)_**

| Label décerné | Auteur | Rapporteur | Éléments reproduits | Liens |
|:-------------:|:------:|:----------:|:-------------------:|:------|
| ![](label_argent.png) | Nawel ARAB<br>[@NawelAr](https://github.com/NawelAr) | David JIA<br>[@djia09-research](https://github.com/djia09-research) |  Figure 1<br>Table 1, ligne 3 | 📌&nbsp;[Dépôt&nbsp;original](https://github.com/NawelAr/GRETSI-2025-submission)<br>⚙️&nbsp;[Issue](https://github.com/GRETSI-2025/Label-Reproductible/issues/29)<br>📝&nbsp;[Rapport](https://github.com/GRETSI-2025/Label-Reproductible/tree/main/rapports/Rapport_issue_29) |

<hr>

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

## Structure 
```
EM-gretsi2025/
├── README.md
├── LICENSE
├── requirements.txt
├── install.sh
├── run.sh
├── src/
│   ├── main.py
│   ├── em.py
│   ├── em_opt.py
│   ├── model.py
│   └── utils.py
├── data/
│   └── visibilities.npz
├── results/
    └── (output files will be saved here)
```

