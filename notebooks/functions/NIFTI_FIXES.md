# NIFTI Conversion Fixes - V2.0.1

## 🔧 Issues Fixed

### Issue 1: 90-Degree Rotation
**Problem**: NIFTI images appeared rotated 90 degrees compared to original DICOM images.

**Root Cause**: Different coordinate system conventions between DICOM and NIFTI formats.
- DICOM typically uses patient-based coordinate systems
- NIFTI uses RAS (Right-Anterior-Superior) radiological coordinate system

**Solution Applied**:
1. **Geometric Correction**: Added `np.rot90(image_array, k=1)` to rotate images 90° counterclockwise
2. **Affine Matrix Update**: Modified the NIFTI affine matrix to use proper radiological orientation:
   ```python
   affine[0, 0] = -1.0  # Flip X for radiological convention
   affine[1, 1] = -1.0  # Flip Y for radiological convention
   affine[2, 2] = 1.0   # Keep Z positive
   ```

### Issue 2: Value Range Reduction to 0-255
**Problem**: NIFTI files were saved with values normalized to 0-255 range instead of preserving original DICOM values.

**Root Cause**: The conversion pipeline normalized all images to uint8 (0-255) before processing, and this normalized data was used for NIFTI output.

**Solution Applied**:
1. **Separate Processing Paths**: Created different handling for PNG vs NIFTI:
   - PNG: Uses normalized uint8 values (0-255) - appropriate for display
   - NIFTI: Uses original DICOM values preserved as float32 - appropriate for analysis

2. **Value Preservation Functions**:
   - `apply_segmentation_transforms_to_original()`: Applies exact same cropping to original values
   - `apply_geometric_transforms_preserve_values()`: Fallback for geometric transformations

3. **Enhanced Conversion Logic**: The `convert_dicom_to_format()` function now:
   - Reads original DICOM data for NIFTI conversion
   - Applies same geometric transformations (cropping, resizing) to original values
   - Preserves full bit depth and value range of original DICOM

## 📋 Functions Modified

### `functions_conversion.py`
- **`save_as_nifti()`**: Added rotation correction and proper affine matrix
- **`convert_dicom_to_format()`**: Added separate processing path for NIFTI with value preservation
- **`apply_geometric_transforms_preserve_values()`**: New helper function for value-preserving transformations

### `functions_pre_processing.py`
- **`apply_segmentation_transforms_to_original()`**: New function to apply exact same cropping to original DICOM values

### Notebook Updates
- Updated main processing pipeline to pass `processing_info` parameter
- Updated test section to use new function signature

## 🧪 Verification

### Test Case 1: Rotation Check
```python
# Before fix: NIFTI image rotated 90° compared to DICOM
# After fix: NIFTI image maintains same orientation as DICOM

# Load both formats and compare orientations
dicom_array, _, _ = read_dicom_file("test.dcm")
nifti_img = nib.load("test.nii.gz")
nifti_array = nifti_img.get_fdata()

# Should have same relative orientations now
```

### Test Case 2: Value Range Check
```python
# Before fix: NIFTI values in 0-255 range
# After fix: NIFTI preserves original DICOM range

dicom_array, _, _ = read_dicom_file("test.dcm")
nifti_img = nib.load("test.nii.gz")
nifti_array = nifti_img.get_fdata()

print(f"Original DICOM range: {dicom_array.min()} - {dicom_array.max()}")
print(f"NIFTI range: {nifti_array.min()} - {nifti_array.max()}")
# Should show similar ranges now (accounting for processing transformations)
```

## 🎯 Expected Results

### Before Fixes:
- ❌ NIFTI images rotated 90° clockwise
- ❌ NIFTI values limited to 0-255 range
- ❌ Loss of original DICOM bit depth and intensity information

### After Fixes:
- ✅ NIFTI images properly oriented (same as DICOM view)
- ✅ NIFTI preserves original DICOM value range and bit depth
- ✅ Maintains all geometric transformations (cropping, resizing)
- ✅ PNG output unaffected (still uses optimized 0-255 range for display)

## 💡 Technical Notes

1. **Backward Compatibility**: PNG output behavior is unchanged - still uses normalized values appropriate for display.

2. **Performance**: NIFTI conversion reads DICOM twice (once for processing, once for value preservation) but ensures data integrity.

3. **Precision**: NIFTI output uses float32 to preserve precision from original DICOM data.

4. **Segmentation**: All lung segmentation and cropping transformations are preserved while maintaining original intensity values.

## 🔄 Version Information

- **Fixed in**: ArchiMed Images V2.0.1
- **Files affected**: `functions_conversion.py`, `functions_pre_processing.py`, notebook pipeline
- **Compatibility**: Fully backward compatible, no configuration changes needed 