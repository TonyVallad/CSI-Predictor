<div align="center">

# Technical Notes — Project Progress

</div>

## 2025-06-24

- Optimized the segmentation module: improved accuracy and explicitly separated left/right lungs through refined morphological post-processing.

## 2025-06-27

- Pipeline overhaul: retained only the primary lung regions via masks, removing spurious segments.
- Configurable crop margin reduced from 180 px to 50 px to focus the model on regions of interest.
- Corrected cropping issues: enforced strict application to masks and final image, with visual overlay for verification.

## 2025-06-30

- Implemented a FileID-based filter to refine sample subsets.
- Debugged and recalculated ROC/AUC curves across all predictions.

## 2025-07-01

- Deployed regional focus on six lung sub-zones, with an option for direct inference on masks.

## 2025-07-02

- Automated histogram generation for each image to characterize intensity distributions and refine preprocessing.
- Experimentation: multi-channel (RGB) encoding of intensities to achieve ×3 grayscale resolution—beware of potential incompatibility with pretrained weights.

## 2025-07-03

- Developed a function to detect pixel-value extremes (min/max) in images for isolating potential artifacts.

## Todo

- ✅ **Switch from using PNGs to NIFTIs** - **COMPLETED in V2.0** (with full diagnostic dynamic range preserved)
- ✅ **Harmonize pixel values** - **COMPLETED** (configurable normalization strategies implemented)
- Clean the dataset (exclude out-of-scope images).
- Compute average CSI and compare to ground truths.
- Conduct manual spot checks to verify visual consistency and softmax outputs.
- Front-end: To launch trainings, see results... (ask Julien if it's worth it)

## Areas to Investigate

- Model sensitivity to artificially varied intensity ranges post-preprocessing.
- New exploration: beyond fine‑tuning the final layers, train the full network (reinitialize all weights) for complete specialization to the target dataset.
