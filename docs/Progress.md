# Development Progress

## 2025-06-24
- Updated segmentation step by improving segmentation accuracy and differenciating between left/right lungs.

## 2025-06-25
- Updated segmentation step by only keeping the main segmentation zone for each lung.
- Reduced crop margin from **** to 50 px.
- 

## Todo
- Remove margin from cropping. (or reduce)
- Keep only main zone during segmentation for each lung.
- Fix ROC curve. (overall)
- Fix pixel value.
- Remove unrelated images.
- Implement zone focus (6 zones + segmentation mask once it's ready)
- Calculate the average CSI and compare it to labeled data.
- Color text for labels on Overlay images.
- Manually test model with a few images to verify everything works as intended.

# Todo (maybe)
- Test model performance when giving it the same image but with different ranges for pixel value.