# Development Progress

## 2025-06-24

- Updated segmentation step by improving segmentation accuracy and differenciating between left/right lungs.

## 2025-06-27

- Updated segmentation step by only keeping the main segmentation zone for each lung.
- Reduced crop margin from 180 to 50 px. (now can be set at the beginning of the notebook)
- Fixed crop not working correctly. (Overlay cyan rectangle, crop on masks, final image crop)

## 2025-06-30

- Added FileID filter functionality.
- Fixed ROC curves. (overall)

## 2025-07-01

- Implemented zone focus (6 zones + segmentation mask option)

## 2025-07-02

- Histogram creation for every image for better pixel value analysis and better image preprocessing.
- Experiment: Encoded pixel value over the 3 chanels (RGB) to get 3x pixel value resolution. (more details) (might not be compatible with pretrained models, testing required...)

## 2025-07-03

- Experiment: Created function to generate mask showing pixels with min or max value on the image.

## Todo

- Fix pixel value.
- Remove unrelated images.
- Calculate the average CSI and compare it to labeled data.
- Manually test model with a few images to verify everything works as intended.

## Todo (maybe)

- Test model performance when giving it the same image but with different ranges for pixel value. (might not be necesarry after updating the preprocessing step)
