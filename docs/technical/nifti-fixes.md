<div align="center">

# NIFTI Fixes

This document covers NIFTI format fixes and solutions.

</div>

## ✅ **Current Implementation Status**

**CSI-Predictor V2.0 now uses NIFTI format exclusively for main images.** All coordinate corrections and value preservation techniques described below are automatically applied in the data loading pipeline.

## 🔧 Previous Issues (Now Resolved in V2.0)

### Issue 1: 90-Degree Rotation ✅ FIXED
**Problem**: NIFTI images appeared rotated 90 degrees compared to original DICOM images.

**Root Cause**: Different coordinate system conventions between DICOM and NIFTI formats.
- DICOM typically uses patient-based coordinate systems
- NIFTI uses RAS (Right-Anterior-Superior) radiological coordinate system

**Solution Applied in V2.0**:
```python
# Automatic coordinate corrections in data loader
img_data = np.transpose(img_data)  # Correct rotation
img_data = np.fliplr(img_data)     # Correct horizontal mirroring
```

### Issue 2: Value Range Reduction ✅ FIXED
**Problem**: NIFTI files were saved with values normalized to 0-255 range instead of preserving original DICOM values.

**Solution Applied in V2.0**:
- **Full HU Range Preservation**: Original Hounsfield Unit values maintained as float32
- **99th Percentile Clipping**: Applied during NIFTI creation to remove artifacts
- **Smart Normalization**: Values normalized to 0-1 range for processing while preserving diagnostic information

## 📋 V2.0 Data Pipeline Integration

### Automatic Processing in `src/data.py`:
- **`_load_nifti_image()`**: Handles coordinate corrections and value normalization
- **`CSIDataset.__getitem__()`**: Seamlessly loads NIFTI files with all corrections applied
- **Caching Support**: Raw NIFTI data cached in memory for performance

### Value Processing Chain:
1. **NIFTI Loading**: Load float32 HU values with nibabel
2. **Coordinate Correction**: Apply transpose + horizontal flip
3. **Range Normalization**: Convert to 0-1 range for model compatibility
4. **Transform Pipeline**: Standard PyTorch transforms with configurable normalization 