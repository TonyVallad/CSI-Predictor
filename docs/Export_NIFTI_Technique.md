# Export NIFTI - Analyse Technique Détaillée

## 🎯 Vue d'ensemble

Ce document détaille le processus d'export NIFTI dans le système ArchiMed Images V2.0, en expliquant comment les valeurs originales sont préservées, comment fonctionne l'écrêtage des valeurs extrêmes, et le rôle de l'égalisation d'histogramme.

## 📋 Table des matières

1. [Pipeline d'export NIFTI](#pipeline-dexport-nifti)
2. [Préservation de la gamme dynamique](#préservation-de-la-gamme-dynamique)
3. [Écrêtage des valeurs extrêmes](#écrêtage-des-valeurs-extrêmes)
4. [Égalisation d'histogramme](#égalisation-dhistogramme)
5. [Comparaison PNG vs NIFTI](#comparaison-png-vs-nifti)
6. [Exemples pratiques](#exemples-pratiques)

---

## 🔄 Pipeline d'export NIFTI

### Architecture du processus

L'export NIFTI suit un pipeline en 4 étapes principales qui diffère fondamentalement de l'export PNG :

```
DICOM original → Préservation valeurs → Transformations géométriques → Export NIFTI
     ↓                    ↓                       ↓                      ↓
[Valeurs HU]      [Float32 préservé]     [Crop + Resize]        [Format médical]
```

### 1. Lecture et préservation initiale

**Fonction : `read_dicom_file()`**

```python
# Lecture du DICOM original avec préservation des valeurs natives
image_array, dicom_data, status = read_dicom_file(dicom_path)
```

**Caractéristiques importantes :**
- **Pas de windowing automatique** : Les valeurs Hounsfield originales sont conservées
- **Type de données préservé** : Conversion en float32 pour éviter la perte de précision
- **Métadonnées DICOM conservées** : Espacements, orientations, positions

### 2. Gestion de la photométrie

```python
if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
    # Inversion pour MONOCHROME1 (0 = blanc)
    image_array = np.max(image_array) - image_array
```

**Pourquoi c'est important :**
- **MONOCHROME1** : 0 = blanc, valeurs élevées = noir
- **MONOCHROME2** : 0 = noir, valeurs élevées = blanc
- **Cohérence d'affichage** : Uniformisation vers MONOCHROME2

---

## 🎛️ Préservation de la gamme dynamique

### Principe fondamental

**Différence clé avec PNG :**
```python
# Pour PNG (8-bit, normalisé)
image_array = normalize_image_array(image_array, 'uint8')  # 0-255

# Pour NIFTI (float32, valeurs originales)
image_array = original_image_array.astype(np.float32)  # Valeurs HU originales
```

### Stratégie de préservation

**1. Double pipeline :**
- **Pipeline de segmentation** : Utilise les images normalisées uint8 pour la détection des poumons
- **Pipeline NIFTI** : Applique les mêmes transformations géométriques aux valeurs originales

**2. Application des transformations géométriques :**

```python
def apply_segmentation_transforms_to_original(original_array, processing_info):
    """
    Applique les transformations de segmentation aux valeurs originales
    """
    if processing_info.get('segmentation_success', False):
        crop_y_min, crop_x_min, crop_y_max, crop_x_max = processing_info['crop_bounds']
        
        # Crop exact identique à celui de la segmentation
        cropped_original = original_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        return cropped_original.astype(np.float32)
```

**3. Redimensionnement préservant les valeurs :**

```python
def resize_with_aspect_ratio_preserve_values(image_array, target_size):
    """
    Redimensionne en préservant la gamme de valeurs originale
    """
    # Stockage des valeurs min/max originales
    original_min = image_array.min()
    original_max = image_array.max()
    
    # Crop pour maintenir l'aspect ratio puis resize
    # Les valeurs originales sont préservées
```

---

## ✂️ Écrêtage des valeurs extrêmes

### Mécanisme d'écrêtage

L'écrêtage se produit **uniquement lors de l'affichage** des NIFTI (pas lors de l'export) :

```python
# Dans load_image_file() - uniquement pour affichage
img_data = np.clip(img_data, 0, np.percentile(img_data, 99))
```

### Pourquoi le 99e percentile ?

**Avantages :**
- **Élimination des artefacts** : Les valeurs extrêmes sont souvent des artefacts
- **Amélioration du contraste** : Concentration de la dynamique sur les valeurs pertinentes
- **Robustesse** : Moins sensible aux outliers que min/max

**Exemple pratique :**
```
Valeurs originales : [-1024, -800, -600, -400, -200, 0, 200, 400, 3071]
99e percentile : 400
Après écrêtage : [-1024, -800, -600, -400, -200, 0, 200, 400, 400]
```

### Impact sur l'export NIFTI

**Important :** L'écrêtage ne s'applique **PAS** à l'export NIFTI final.

```python
# Export NIFTI : valeurs originales complètes conservées
result = save_as_nifti(image_array, output_path, dicom_data)  # Pas d'écrêtage

# Affichage : écrêtage pour la visualisation seulement
img_data = np.clip(img_data, 0, np.percentile(img_data, 99))  # Écrêtage
```

---

## 📊 Égalisation d'histogramme

### Types d'égalisation utilisés

**1. Égalisation standard (option) :**
```python
if enable_hist_eq:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
```

**2. CLAHE (Contrast Limited Adaptive Histogram Equalization) :**

**Paramètres :**
- **clipLimit=3.0** : Limite l'amplification du contraste
- **tileGridSize=(8,8)** : Grille de 8×8 tuiles pour adaptation locale

### Fonctionnement détaillé de CLAHE

**Étapes du processus :**

1. **Division en tuiles** : L'image est divisée en grille 8×8 (64 régions)

2. **Calcul d'histogramme par tuile** :
   ```
   Pour chaque tuile de 64×64 pixels :
   - Calcul de l'histogramme local
   - Application de la limite d'écrêtage (clipLimit)
   ```

3. **Redistribution de l'excès** :
   ```
   Si histogramme[intensité] > clipLimit:
       excès = histogramme[intensité] - clipLimit
       redistribution uniforme de l'excès
   ```

4. **Égalisation locale** :
   ```
   Nouvelle_valeur = transformation_locale(ancienne_valeur)
   ```

5. **Interpolation bilinéaire** : Lissage entre les tuiles adjacentes

### Impact sur la segmentation

**Pourquoi utiliser CLAHE :**
- **Amélioration du contraste local** : Meilleure détection des contours pulmonaires
- **Réduction des variations d'éclairage** : Compensation des différences d'exposition
- **Préservation des détails fins** : Évite la surenhancement global

**Effet sur les valeurs :**
```python
# Avant CLAHE (exemple)
Region_poumon = [180, 185, 175, 190, 182]
Region_fond = [45, 50, 48, 52, 47]

# Après CLAHE
Region_poumon = [195, 210, 185, 220, 205]  # Contraste augmenté
Region_fond = [25, 35, 28, 40, 30]         # Contraste augmenté
```

### Application dans le pipeline

**Séquence d'égalisation :**
```python
def enhance_image_preprocessing(image, enable_hist_eq=True, enable_blur=True):
    # 1. Conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Application CLAHE
    if enable_hist_eq:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image_gray = clahe.apply(image_gray)
    
    # 3. Lissage gaussien (optionnel)
    if enable_blur:
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
    
    return image_gray
```

**Utilisation pour segmentation uniquement :**
- L'égalisation améliore la **détection des poumons**
- Les **valeurs originales** restent inchangées pour l'export NIFTI
- **Double avantage** : segmentation optimisée + données préservées

---

## ⚖️ Comparaison PNG vs NIFTI

### Tableau comparatif

| Aspect | PNG | NIFTI |
|--------|-----|-------|
| **Type de données** | uint8 (0-255) | float32 (valeurs originales) |
| **Gamme dynamique** | Limitée | Complète |
| **Valeurs Hounsfield** | ❌ Perdues | ✅ Préservées |
| **Métadonnées médicales** | ❌ Aucune | ✅ Complètes |
| **Espacement pixels** | ❌ Perdu | ✅ Préservé |
| **Orientation** | ❌ Standard image | ✅ Médicale (RAS) |
| **Taille fichier** | Plus petite | Plus grande |
| **Usage recommandé** | Visualisation | Analyse médicale |

### Exemple de préservation des valeurs

**DICOM original :**
```
Valeurs HU : [-1024, -800, -600, -400, -200, 0, 200, 400, 600]
Air : -1000 HU
Poumon : -800 à -600 HU  
Tissus mous : -100 à +100 HU
Os : +400 à +1000 HU
```

**Export PNG :**
```
Normalisé 0-255 : [0, 51, 76, 102, 127, 153, 179, 204, 230]
Information HU : PERDUE
Diagnostic : IMPOSSIBLE
```

**Export NIFTI :**
```
Float32 conservé : [-1024.0, -800.0, -600.0, -400.0, -200.0, 0.0, 200.0, 400.0, 600.0]
Information HU : PRÉSERVÉE
Diagnostic : POSSIBLE
```

---

## 💡 Exemples pratiques

### Cas d'usage 1 : Analyse quantitative

**Objectif :** Mesurer la densité pulmonaire en HU

```python
# Chargement NIFTI
nifti_img = nib.load('poumon_patient.nii.gz')
data = nifti_img.get_fdata()

# Analyse directe en unités Hounsfield
region_poumon = data[100:200, 150:250]
densite_moyenne = np.mean(region_poumon)
print(f"Densité pulmonaire moyenne : {densite_moyenne:.1f} HU")

# Interprétation médicale possible :
# -900 à -800 HU : Poumon normal
# -800 à -700 HU : Poumon partiellement aéré
# -700 à -600 HU : Consolidation
```

### Cas d'usage 2 : Comparaison avant/après traitement

**Avantage de la préservation des valeurs :**

```python
# Comparaison quantitative possible
avant_traitement = nifti_avant.get_fdata()
apres_traitement = nifti_apres.get_fdata()

difference_hu = apres_traitement - avant_traitement
evolution_moyenne = np.mean(difference_hu)

print(f"Évolution moyenne : {evolution_moyenne:.1f} HU")
# Valeur positive : amélioration (poumon plus aéré)
# Valeur négative : dégradation (poumon moins aéré)
```

### Cas d'usage 3 : Segmentation basée sur les seuils HU

```python
# Segmentation par seuils Hounsfield
poumon_normal = (data >= -950) & (data <= -700)
consolidation = (data >= -100) & (data <= 100)
emphyseme = data < -950

volumes = {
    'poumon_normal': np.sum(poumon_normal) * voxel_volume,
    'consolidation': np.sum(consolidation) * voxel_volume,
    'emphyseme': np.sum(emphyseme) * voxel_volume
}
```

---

## 🔧 Configuration et optimisation

### Paramètres d'export recommandés

```python
# Configuration optimale pour l'analyse médicale
OUTPUT_FORMAT = 'nifti'
ENABLE_HISTOGRAM_EQUALIZATION = True  # Pour améliorer la segmentation
ENABLE_GAUSSIAN_BLUR = True          # Pour réduire le bruit
TARGET_SIZE = (518, 518)             # Taille standardisée
CROP_MARGIN = 25                     # Marge autour des poumons
```

### Vérification de la qualité d'export

```python
# Vérification post-export
nifti_img = nib.load(output_path)
data = nifti_img.get_fdata()

print(f"Forme : {data.shape}")
print(f"Type : {data.dtype}")
print(f"Gamme : {data.min():.1f} à {data.max():.1f} HU")
print(f"Espacement : {nifti_img.header.get_zooms()}")
```

---

## 📖 Conclusion

L'export NIFTI dans ArchiMed Images V2.0 utilise une approche sophistiquée qui :

1. **Préserve l'intégrité médicale** : Conservation des valeurs Hounsfield originales
2. **Optimise la segmentation** : Utilisation d'égalisation d'histogramme pour la détection
3. **Maintient la précision** : Aucune perte d'information diagnostique
4. **Applique des transformations intelligentes** : Crop et resize basés sur la segmentation
5. **Évite les artefacts** : Écrêtage intelligent pour l'affichage uniquement

Cette approche permet une analyse médicale quantitative fiable tout en bénéficiant des améliorations de segmentation automatique. 