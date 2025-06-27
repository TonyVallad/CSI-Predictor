# Development Progress

## 2025-06-24
- Updated segmentation step by improving segmentation accuracy and differenciating between left/right lungs.

## 2025-06-27
- Updated segmentation step by only keeping the main segmentation zone for each lung.
- Reduced crop margin from 180 to 50 px. (now can be set at the beginning of the notebook)
- Fixed crop not working correctly. (Overlay cyan rectangle, crop on masks, final image crop)

## Todo
- Fix ROC curve. (overall)
- Fix pixel value.
- Remove unrelated images.
- Implement zone focus (6 zones + segmentation mask once it's ready)
- Calculate the average CSI and compare it to labeled data.
- Manually test model with a few images to verify everything works as intended.

## Todo (maybe)
- Test model performance when giving it the same image but with different ranges for pixel value.