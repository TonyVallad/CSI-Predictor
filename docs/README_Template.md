<div align="center">

# CSI‑Predictor

</div>

**Predictor quantitatif du Critical Success Index (CSI)** pour l’analyse d’images pulmonaires via deep learning.

---

## 🧭 1. Contexte & Objectif

Ce projet vise à construire un modèle d’IA capable de prédire le CSI à partir d’images thoraciques, en appliquant un pipeline complet :

1. Prétraitement (segmentation, crop, histogrammes, encodage multi-canal)
2. Extraction / sélection de zones (6 sous-régions pulmonaires)
3. Entraînement (fine-tuning + entraînement complet)
4. Évaluation (ROC, AUC, CSI, analyses visuelles)

---

## 📂 2. Structure du dépôt

```

CSI‑Predictor/
├── data/                  # images originales, masques, annotations
├── notebooks/            # notebook d’analyse & visualisation
├── src/
│   ├── preprocessing/     # segmentation, cropping, histogrammes
│   ├── models/            # définition du modèle, entraînement
│   └── evaluation/        # ROC, CSI, visualisations
├── experiments/          # logs, checkpoints, sorties graphiques
├── requirements.txt      # dépendances Python
└── README.md             # ce fichier

````

---

## 🛠️ 3. Installation

```bash
git clone https://github.com/TonyVallad/CSI-Predictor.git
cd CSI-Predictor
pip install -r requirements.txt
````

**Optionnel** : créer un environnement conda pour assurer la reproductibilité :

```bash
conda env create -f environment.yml
conda activate csi-env
```

---

## ▶️ 4. Utilisation

1. **Prétraitement** :

   ```bash
   python src/preprocessing/segment.py --input_dir data/raw --output_dir data/processed --crop_margin 50
   python src/preprocessing/histograms.py
   ```
2. **Entraînement** (ex. fine-tuning) :

   ```bash
   python src/models/train.py --mode finetune --epochs 50
   ```

   Ou pour entraînement complet :

   ```bash
   python src/models/train.py --mode full_train --epochs 100
   ```
3. **Évaluation** :

   ```bash
   python src/evaluation/evaluate.py --checkpoint experiments/latest.ckpt
   ```

   Tu obtiendras : ROC, AUC, CSI moyen, visualisations (masques vs GT).

---

## 📊 5. Jeux de données

* **data/raw/** : images DICOM initiales + masques.
* **data/processed/** : images segmentées/croppées + histogrammes + masques extrêmes.
* **data/annotations/** : labels Ground Truth + FileID pour filtrage.

---

## 📐 6. Méthodologie

* **Segmentation** : séparation poumons gauche/droit, extraction des zones principales.
* **Crop** : marge configurable, suppression des zones parasites.
* **Histogrammes & encodage RGB** : analyse de distribution + granularité augmentée des features.
* **Focus zonal** : 6 sous-régions avec masques dédiés.
* **Expérimentations** :

  * fine-tuning de la tête du réseau,
  * entraînement complet via réinitialisation des poids,
  * détection des pixels extrêmes (min/max) pour filtrer bruit/artefacts.
* **Évaluation** : ROC, AUC, CSI, overlay visuel pour validation qualitative.

---

## 📈 7. Résultats actuels

* Segmentation précise avec séparation pulmonaire.
* Crop stable à 50 px.
* Histograms générés et encodage RGB fonctionnel.
* Baseline CSI moyen = `X.XX`, AUC = `0.YY`.
* Entraînement complet en cours d’évaluation — à compléter.

---

## ✔️ 8. Roadmap & perspectives

* \[x] Pipeline de segmentation finalisé
* \[x] Crop optimisé
* \[x] Histogrammes & masques extrêmes
* \[x] Fine-tuning opérationnel
* \[ ] Entraînement complet du modèle
* \[ ] Analyse de sensibilité aux plages d’intensité
* [ ] Validation croisée
* \[ ] Publication des résultats (préprint, article, code packagé)

---

## 📚 9. Contribuer

Tu souhaites contribuer ? Merci !
Merci de :

1. Ouvrir une issue pour toute idée ou bug.
2. Soumettre une PR claire, testée, documentée.

---

## 📄 10. Licence & contacts

* **Licence** : MIT (voir `LICENSE`)
* **Auteur** : Tony Vallad — Github: `TonyVallad`
* **Contact** : [tony.vallad@example.com](mailto:tony.vallad@example.com)

---

## 🔗 11. Références

* Template README scientifique ([en.wikipedia.org][2], [gist.github.com][1], [ubc-library-rc.github.io][3])
* IMRaD structure pour rapports ([en.wikipedia.org][4])
