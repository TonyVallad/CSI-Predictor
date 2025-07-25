# Export NIFTI - Technique et Préservation des Valeurs

## Pipeline d'export NIFTI

Le système d'export NIFTI de ce projet utilise une approche en 4 étapes principales pour préserver l'intégrité des données médicales :

1. **Lecture DICOM original** : Extraction des valeurs brutes (Unités Hounsfield)
2. **Préservation des valeurs** : Conversion en `float32` sans normalisation 
3. **Transformations géométriques** : Application des recadrages et redimensionnements
4. **Export NIFTI** : Sauvegarde avec métadonnées DICOM préservées

### Double pipeline de traitement

Le projet utilise une architecture "double pipeline" :

- **Pipeline de segmentation** : Travaille sur des images normalisées `uint8` (0-255) pour optimiser les algorithmes de segmentation
- **Pipeline d'export NIFTI** : Applique les **mêmes transformations géométriques** aux valeurs originales `float32`

Cette approche garantit que :
- La segmentation fonctionne de manière optimale avec des images normalisées
- L'export NIFTI préserve toutes les valeurs diagnostiques originales

### Fonctions clés 

#### `apply_segmentation_transforms_to_original`
```python
# Applique les mêmes transformations géométriques (recadrage, redimensionnement)
# aux données originales float32 tout en préservant les valeurs HU
```

#### `resize_with_aspect_ratio_preserve_values`
```python
# Redimensionne en préservant le ratio d'aspect ET la gamme de valeurs originales
# Utilise l'interpolation LANCZOS4 pour minimiser la perte d'information
```

## Préservation de la gamme dynamique

### Valeurs dans le NIFTI (selon configuration V2.1)

**Mode Standard** (`APPLY_PERCENTILE_CLIPPING = False`) :
- **Unités Hounsfield (HU) complètes** : De -1024 à +3071 (ou plus selon le scanner)
- **Aucun écrêtage** : Toutes les valeurs originales préservées
- **Analyse quantitative optimale** : Recherche et diagnostic précis

**Mode Optimisé IA** (`APPLY_PERCENTILE_CLIPPING = True`) :
- **Unités Hounsfield écrêtées** : Typiquement de -1024 à +400 HU (99e percentile)
- **Artefacts éliminés** : Valeurs >99e percentile supprimées
- **Entraînement IA optimisé** : Données cohérentes et robustes

**Commun aux deux modes** :
- **Précision float32** : Préservation des valeurs décimales
- **Métadonnées DICOM** : Espacement des pixels, position, orientation

### Valeurs transformées pour la segmentation uniquement
- **Normalisation 0-255** : Pour optimiser TorchXRayVision
- **Type uint8** : Pour réduire l'usage mémoire pendant la segmentation
- **Égalisation d'histogramme** : Pour améliorer le contraste lors de la détection

## Écrêtage des valeurs extrêmes

### Évolution V2.0 → V2.1

**V2.0 (Comportement original)** :
L'écrêtage au 99e percentile s'appliquait **uniquement** pour l'affichage visuel.

**V2.1 (Optimisation IA)** :
L'écrêtage peut maintenant s'appliquer **aussi pendant l'export NIFTI** si activé.

### Où s'applique l'écrêtage en V2.1

1. **Dans `load_image_file` (Image_Exploration)** : Toujours actif pour l'affichage visuel
2. **Dans l'export NIFTI** : Optionnel, contrôlé par `APPLY_PERCENTILE_CLIPPING`

### Configuration et comportement

```python
# Configuration V2.1
APPLY_PERCENTILE_CLIPPING = True   # Écrêtage pendant l'export NIFTI
PERCENTILE_THRESHOLD = 99.0        # Seuil configurable

# Si APPLY_PERCENTILE_CLIPPING = True :
clip_value = np.percentile(image_to_save, 99.0)
nifti_data = np.clip(image_to_save, image_to_save.min(), clip_value)

# Si APPLY_PERCENTILE_CLIPPING = False :
nifti_data = original_dicom_values.astype(np.float32)  # Valeurs complètes préservées
```

### Justification de l'écrêtage optionnel

**Pour modèles d'IA** (`APPLY_PERCENTILE_CLIPPING = True`) :
- Élimination des artefacts perturbateurs
- Standardisation inter-patients  
- Stabilité d'entraînement améliorée

**Pour analyse quantitative pure** (`APPLY_PERCENTILE_CLIPPING = False`) :
- Préservation intégrale des valeurs HU originales
- Aucune perte d'information diagnostique

## Égalisation d'histogramme (CLAHE)

### Configuration CLAHE
```python
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
```

### Paramètres détaillés

- **clipLimit=3.0** : Limite de contraste pour éviter la sur-amplification du bruit
- **tileGridSize=(8, 8)** : Division de l'image en 64 tuiles (8×8) pour un traitement adaptatif local
- **Adaptatif** : Chaque tuile a sa propre courbe d'égalisation optimisée

### Application et portée

**Où s'applique CLAHE :**
- Fonction `enhance_image_preprocessing` 
- Sur l'image normalisée `uint8` (pipeline de segmentation)
- **Uniquement pour améliorer la détection des poumons**

**Où ne s'applique PAS CLAHE :**
- Export NIFTI final (valeurs HU originales préservées)
- Analyse quantitative des valeurs
- Mesures diagnostiques

### Avantages pour la segmentation
1. **Amélioration du contraste local** : Détection plus fine des contours pulmonaires
2. **Réduction des variations d'éclairage** : Homogénéisation inter-patients
3. **Préservation des détails** : Évite la perte d'information dans les zones sombres/claires

## Comparaison Formats d'Images (V2.0)

| Aspect | PNG (Legacy) | NIFTI (Actuel) |
|--------|--------------|----------------|
| **Type de données** | uint8 (0-255) | float32 (HU avec écrêtage 99e percentile) |
| **Gamme dynamique** | Normalisée | Diagnostique préservée (~-1024 à +400 HU) |
| **Valeurs HU** | ❌ Perdues | ✅ Préservées et diagnostiques |
| **Artefacts** | ❌ Normalisés | ✅ Éliminés via écrêtage 99e percentile |
| **Métadonnées médicales** | ❌ Perdues | ✅ Préservées (espacement, orientation) |
| **Taille fichier** | Petite (~50 KB) | Moyenne (~500 KB) |
| **Usage recommandé** | Visualisation web uniquement | **Entraînement IA et analyse quantitative** |
| **Format du projet** | ❌ Non supporté | ✅ **Format principal** |

**Note importante**: Le projet CSI-Predictor utilise désormais **exclusivement le format NIFTI** pour les images principales. Les fichiers PNG sont conservés uniquement pour les masques de segmentation et les overlays de visualisation.

## Exemples pratiques

### Utilisation pour l'analyse quantitative
```python
# Chargement NIFTI avec valeurs HU préservées
nifti_img = nib.load('patient_001.nii.gz')
hu_values = nifti_img.get_fdata()

# Analyse de congestion (valeurs réelles)
normal_lung = hu_values[(hu_values >= -950) & (hu_values <= -700)]  # Poumon aéré
ground_glass = hu_values[(hu_values >= -700) & (hu_values <= -300)]  # Verre dépoli
consolidation = hu_values[(hu_values >= -300) & (hu_values <= 100)]  # Consolidation
```

### Comparaison inter-patients
```python
# Analyse comparative possible grâce à la préservation des valeurs HU
patient_A_mean_hu = np.mean(hu_values_A[lung_mask_A])
patient_B_mean_hu = np.mean(hu_values_B[lung_mask_B])

# Différence cliniquement significative
hu_difference = patient_A_mean_hu - patient_B_mean_hu
```

### Segmentation par seuillage
```python
# Segmentation précise basée sur les valeurs HU préservées
emphysema_mask = hu_values < -950      # Emphysème
normal_lung_mask = (hu_values >= -950) & (hu_values <= -700)
pathology_mask = hu_values > -700      # Pathologie (congestion, etc.)
```

Cette approche garantit que l'export NIFTI conserve toute l'information diagnostique originale tout en bénéficiant d'une segmentation optimisée par le preprocessing adaptatif.

---

## **🚀 Nouvelles Fonctionnalités V2.1 : Optimisation IA**

### Écrêtage au Percentile pour Modèles d'IA

À partir de la version V2.1, le système inclut une option d'**écrêtage au 99e percentile** spécialement conçue pour optimiser les données destinées aux modèles d'intelligence artificielle.

#### Configuration
```python
# ===== AI OPTIMIZATION SETTINGS (V2.1) =====
ENABLE_AI_OPTIMIZATION = True          # Activer l'optimisation IA
APPLY_PERCENTILE_CLIPPING = True       # Écrêter les valeurs extrêmes
PERCENTILE_THRESHOLD = 99.0            # Seuil de percentile (99e percentile)
AI_TARGET_RANGE = (-1024, 400)        # Plage HU attendue pour l'analyse pulmonaire
```

#### Fonctionnement Technique

**Avant (V2.0)** :
```python
# Export NIFTI sans écrêtage
nifti_data = original_values.astype(np.float32)  # [-1024, +3000] HU
```

**Après (V2.1 avec optimisation IA)** :
```python
# Export NIFTI avec écrêtage au 99e percentile
clip_value = np.percentile(original_values, 99.0)
nifti_data = np.clip(original_values, original_values.min(), clip_value)
# Résultat typique : [-1024, +400] HU (artefacts >400 éliminés)
```

### Avantages pour la Prédiction de Score de Congestion

#### 1. **Robustesse aux Artefacts**
```python
# Élimination des valeurs aberrantes :
# - Artefacts métalliques : >2000 HU
# - Erreurs de calibration : valeurs extrêmes
# - Bruit électronique : pics isolés

# Avec écrêtage :
Patient_A: [-1024, ..., +350] HU  # Artefact à +2500 éliminé
Patient_B: [-1024, ..., +380] HU  # Données propres préservées
```

#### 2. **Standardisation Inter-Patients**
- **Plage de valeurs cohérente** entre tous les patients
- **Élimination des variations dues aux artefacts** non-diagnostiques
- **Préservation de toute la plage diagnostique** (-1024 à +400 HU)

#### 3. **Optimisation pour l'Entraînement IA**
- **Stabilité numérique** : Évite les gradients explosifs dus aux valeurs extrêmes
- **Convergence améliorée** : Plage de données plus homogène
- **Généralisation** : Modèle moins sensible aux artefacts spécifiques

### Plages Diagnostiques Préservées

| Zone Anatomique | Plage HU | Impact Écrêtage 99e |
|-----------------|----------|---------------------|
| **Poumon normal** | -950 à -700 | ✅ **100% préservé** |
| **Verre dépoli** | -700 à -300 | ✅ **100% préservé** |
| **Consolidation** | -300 à +100 | ✅ **100% préservé** |
| **Tissus mous** | +20 à +80 | ✅ **100% préservé** |
| **Artefacts** | >+500 | 🔧 **Éliminés (optimal)** |

### Comparaison Avant/Après

#### **Sans Optimisation IA (V2.0)**
```python
# Valeurs complètes préservées
HU_range = [-1024, +3071]  # Inclut tous les artefacts
AI_challenges = [
    "Artefacts métalliques perturbent l'entraînement",
    "Variations extrêmes entre patients",
    "Instabilité numérique potentielle"
]
```

#### **Avec Optimisation IA (V2.1)**
```python
# Valeurs diagnostiques préservées, artefacts éliminés
HU_range = [-1024, +400]   # 99e percentile typique
AI_benefits = [
    "Données cohérentes entre patients",
    "Stabilité d'entraînement améliorée", 
    "Toute la plage diagnostique préservée"
]
```

### Activation/Désactivation

L'écrêtage au percentile est **entièrement optionnel** :

```python
# Pour modèles d'IA (recommandé)
APPLY_PERCENTILE_CLIPPING = True

# Pour analyse quantitative pure (recherche)
APPLY_PERCENTILE_CLIPPING = False
```

### Impact sur le Flou Gaussien

En V2.1, le **flou gaussien est désactivé par défaut** pour les modèles d'IA :

```python
# V2.0 (segmentation optimisée)
ENABLE_GAUSSIAN_BLUR = True   # Améliore la segmentation

# V2.1 (données IA optimisées)  
ENABLE_GAUSSIAN_BLUR = False  # Préserve les détails originaux pour l'IA
```

**Justification** : Les modèles d'IA bénéficient de données originales non-filtrées, le flou pouvant masquer des détails diagnostiques subtils importants pour la prédiction du score de congestion.

### Recommandations d'Usage

#### **Pour Modèles d'IA de Congestion** ✅
- `APPLY_PERCENTILE_CLIPPING = True`
- `PERCENTILE_THRESHOLD = 99.0`
- `ENABLE_GAUSSIAN_BLUR = False`

#### **Pour Analyse Quantitative Pure** 🔬
- `APPLY_PERCENTILE_CLIPPING = False`
- Préservation intégrale des valeurs HU originales

#### **Pour Recherche Comparative** 📊
- `APPLY_PERCENTILE_CLIPPING = True` 
- Cohérence entre tous les échantillons de l'étude

Cette optimisation V2.1 garantit des données NIFTI parfaitement adaptées à l'entraînement et à l'inférence de modèles d'IA pour la prédiction de scores de congestion pulmonaire, tout en préservant la flexibilité pour d'autres applications cliniques. 