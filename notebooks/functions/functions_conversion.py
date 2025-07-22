"""
ArchiMed Images V2.0 - Conversion Functions
Functions for handling DICOM to PNG/NIFTI conversions and format management.
"""

import os
import pydicom
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings


def read_dicom_file(dicom_path: str) -> Tuple[Optional[np.ndarray], Optional[pydicom.Dataset], str]:
    """
    Read and validate a DICOM file.
    
    Args:
        dicom_path (str): Path to the DICOM file
    
    Returns:
        Tuple[image_array, dicom_data, status]: Image array, DICOM dataset, and status message
    """
    try:
        if not os.path.exists(dicom_path):
            return None, None, f"File not found: {dicom_path}"
        
        # Suppress DICOM warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dicom_data = pydicom.dcmread(dicom_path)
        
        # Check if pixel data exists
        if not hasattr(dicom_data, 'pixel_array'):
            return None, None, f"No pixel data found in {dicom_path}"
        
        image_array = dicom_data.pixel_array.copy()
        
        # Handle photometric interpretation
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                # Invert for MONOCHROME1 (0 = white)
                image_array = np.max(image_array) - image_array
        
        return image_array, dicom_data, "Success"
        
    except Exception as e:
        return None, None, f"Error reading DICOM: {str(e)}"


def normalize_image_array(image_array: np.ndarray, target_dtype: str = 'uint8') -> np.ndarray:
    """
    Normalize image array to specified data type range.
    
    Args:
        image_array (np.ndarray): Input image array
        target_dtype (str): Target data type ('uint8', 'uint16', 'float32')
    
    Returns:
        np.ndarray: Normalized image array
    """
    if target_dtype == 'uint8':
        if image_array.max() > 255:
            # Normalize to 0-255 range
            normalized = ((image_array - image_array.min()) / 
                         (image_array.max() - image_array.min()) * 255)
            return normalized.astype(np.uint8)
        else:
            return image_array.astype(np.uint8)
    
    elif target_dtype == 'uint16':
        if image_array.max() > 65535:
            # Normalize to 0-65535 range
            normalized = ((image_array - image_array.min()) / 
                         (image_array.max() - image_array.min()) * 65535)
            return normalized.astype(np.uint16)
        else:
            return image_array.astype(np.uint16)
    
    elif target_dtype == 'float32':
        return image_array.astype(np.float32)
    
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")


def resize_with_aspect_ratio(image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio and center cropping if needed.
    
    Args:
        image_array (np.ndarray): Input image array
        target_size (Tuple[int, int]): Target size (width, height)
    
    Returns:
        np.ndarray: Resized image array
    """
    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array, mode='L')
    else:
        pil_image = Image.fromarray(image_array)
    
    current_width, current_height = pil_image.size
    target_width, target_height = target_size
    current_ratio = current_width / current_height
    target_ratio = target_width / target_height
    
    # Calculate crop dimensions to maintain aspect ratio
    if current_ratio > target_ratio:
        # Image is wider, crop width
        new_width = int(current_height * target_ratio)
        new_height = current_height
        left = (current_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = current_height
    else:
        # Image is taller, crop height
        new_width = current_width
        new_height = int(current_width / target_ratio)
        left = 0
        top = (current_height - new_height) // 2
        right = current_width
        bottom = top + new_height
    
    # Crop and resize
    pil_image = pil_image.crop((left, top, right, bottom))
    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(pil_image)


def save_as_png(image_array: np.ndarray, output_path: str) -> bool:
    """
    Save image array as PNG file.
    
    Args:
        image_array (np.ndarray): Image array to save
        output_path (str): Output file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to PIL Image and save
        if len(image_array.shape) == 2:
            pil_image = Image.fromarray(image_array, mode='L')
        else:
            pil_image = Image.fromarray(image_array)
        
        pil_image.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error saving PNG {output_path}: {e}")
        return False


def save_as_nifti(image_array: np.ndarray, output_path: str, dicom_data: Optional[pydicom.Dataset] = None) -> bool:
    """
    Save image array as NIFTI file with proper orientation and DICOM metadata preservation.
    
    Args:
        image_array (np.ndarray): Image array to save
        output_path (str): Output file path
        dicom_data (Optional[pydicom.Dataset]): DICOM dataset for metadata
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Fix orientation: Rotate 90 degrees counterclockwise to match NIFTI convention
        # This corrects the 90-degree rotation issue
        image_array_corrected = np.rot90(image_array, k=1)  # k=1 means 90 degrees counterclockwise
        
        # Ensure 3D array for NIFTI (add singleton dimension if 2D)
        if image_array_corrected.ndim == 2:
            image_array_corrected = image_array_corrected[:, :, np.newaxis]
        
        # Create affine matrix with proper orientation for medical images
        # Standard radiological orientation: RAS (Right-Anterior-Superior)
        affine = np.eye(4)
        affine[0, 0] = -1.0  # Flip X for radiological convention
        affine[1, 1] = -1.0  # Flip Y for radiological convention
        affine[2, 2] = 1.0   # Keep Z positive
        
        # Try to use DICOM metadata for better spacing and position
        if dicom_data is not None:
            try:
                if hasattr(dicom_data, 'PixelSpacing'):
                    pixel_spacing = dicom_data.PixelSpacing
                    # Apply spacing but maintain orientation flips
                    affine[0, 0] = -float(pixel_spacing[1])  # Column spacing (x) with flip
                    affine[1, 1] = -float(pixel_spacing[0])  # Row spacing (y) with flip
                
                if hasattr(dicom_data, 'SliceThickness'):
                    affine[2, 2] = float(dicom_data.SliceThickness)
                elif hasattr(dicom_data, 'SpacingBetweenSlices'):
                    affine[2, 2] = float(dicom_data.SpacingBetweenSlices)
                
                if hasattr(dicom_data, 'ImagePositionPatient'):
                    position = dicom_data.ImagePositionPatient
                    affine[0, 3] = float(position[0])  # X position
                    affine[1, 3] = float(position[1])  # Y position
                    affine[2, 3] = float(position[2])  # Z position
            except Exception:
                pass  # Use default affine if metadata parsing fails
        
        # Create NIFTI image with corrected orientation
        nifti_img = nib.Nifti1Image(image_array_corrected, affine=affine)
        
        # Save NIFTI file
        nib.save(nifti_img, output_path)
        return True
        
    except Exception as e:
        print(f"Error saving NIFTI {output_path}: {e}")
        return False


def apply_geometric_transforms_preserve_values(original_array: np.ndarray, processed_array: np.ndarray, 
                                             target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Apply the same geometric transformations as processed_array to original_array while preserving values.
    
    Args:
        original_array (np.ndarray): Original image with full value range
        processed_array (np.ndarray): Processed image (normalized, cropped, etc.)
        target_size (Optional[Tuple[int, int]]): Target size for final output
    
    Returns:
        np.ndarray: Original array with same geometric transformations applied
    """
    try:
        # Convert original to float32 to preserve precision
        result = original_array.astype(np.float32)
        
        # If processed array has different dimensions, we need to figure out the transformation
        orig_h, orig_w = original_array.shape[:2]
        proc_h, proc_w = processed_array.shape[:2]
        
        # If dimensions changed, assume it was cropped and/or resized
        if (orig_h, orig_w) != (proc_h, proc_w):
            # For now, simply resize to match processed dimensions
            # This is a simplified approach - in a full implementation, we'd need
            # to track the exact crop bounds and apply them here
            if target_size is not None:
                result = resize_with_aspect_ratio(result, target_size)
            else:
                result = resize_with_aspect_ratio(result, (proc_w, proc_h))
        elif target_size is not None:
            # Same dimensions but target size specified
            result = resize_with_aspect_ratio(result, target_size)
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not apply geometric transforms, using original array: {e}")
        return original_array.astype(np.float32)


def convert_dicom_to_format(dicom_path: str, output_path: str, output_format: str, 
                          target_size: Optional[Tuple[int, int]] = None,
                          processed_image_array: Optional[np.ndarray] = None,
                          processing_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convert DICOM file to specified format (PNG or NIFTI).
    
    Args:
        dicom_path (str): Path to input DICOM file
        output_path (str): Path for output file
        output_format (str): Output format ('png' or 'nifti')
        target_size (Optional[Tuple[int, int]]): Target size for resizing
        processed_image_array (Optional[np.ndarray]): Pre-processed image array
        processing_info (Optional[Dict[str, Any]]): Processing metadata with transformation info
    
    Returns:
        bool: True if successful, False otherwise
    """
    # For NIFTI format, we need to preserve original DICOM values
    if output_format.lower() == 'nifti':
        # Always read original DICOM data for NIFTI to preserve value range
        original_image_array, dicom_data, status = read_dicom_file(dicom_path)
        if original_image_array is None:
            print(f"Failed to read DICOM: {status}")
            return False
        
        # For NIFTI, use original values but apply the same geometric transformations
        # if processed_image_array was provided (for segmentation-based cropping)
        if processed_image_array is not None and processing_info is not None:
            # Use the exact same transformations from segmentation processing
            from functions_pre_processing import apply_segmentation_transforms_to_original
            image_array = apply_segmentation_transforms_to_original(original_image_array, processing_info)
            
            # Apply final resizing if needed
            if target_size is not None:
                image_array = resize_with_aspect_ratio(image_array, target_size)
                
        elif processed_image_array is not None:
            # Fall back to geometric transformation matching
            image_array = apply_geometric_transforms_preserve_values(
                original_image_array, processed_image_array, target_size
            )
        else:
            # No processing was done, use original image
            image_array = original_image_array.astype(np.float32)
            
            # Resize if requested
            if target_size is not None:
                image_array = resize_with_aspect_ratio(image_array, target_size)
        
        return save_as_nifti(image_array, output_path, dicom_data)
    
    # For PNG format, use the standard processing pipeline
    if processed_image_array is not None:
        # Use pre-processed image array
        image_array = processed_image_array
        dicom_data = None
    else:
        # Read and process DICOM file
        image_array, dicom_data, status = read_dicom_file(dicom_path)
        if image_array is None:
            print(f"Failed to read DICOM: {status}")
            return False
        
        # Normalize to uint8 for PNG processing
        image_array = normalize_image_array(image_array, 'uint8')
    
    # Resize if requested
    if target_size is not None:
        image_array = resize_with_aspect_ratio(image_array, target_size)
    
    # Save as PNG
    if output_format.lower() == 'png':
        return save_as_png(image_array, output_path)
    else:
        print(f"Unsupported output format: {output_format}")
        return False


def get_output_path(dicom_path: str, output_dir: str, output_format: str, file_id: Optional[str] = None) -> str:
    """
    Generate output path based on input DICOM path and format.
    
    Args:
        dicom_path (str): Path to input DICOM file
        output_dir (str): Output directory
        output_format (str): Output format ('png' or 'nifti')
        file_id (Optional[str]): Custom file ID (uses DICOM filename if None)
    
    Returns:
        str: Output file path
    """
    if file_id is None:
        file_id = os.path.splitext(os.path.basename(dicom_path))[0]
    
    if output_format.lower() == 'png':
        extension = '.png'
    elif output_format.lower() == 'nifti':
        extension = '.nii.gz'
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    return os.path.join(output_dir, f"{file_id}{extension}")


def validate_dicom_file(dicom_path: str) -> bool:
    """
    Quick validation to check if file is a valid DICOM.
    
    Args:
        dicom_path (str): Path to DICOM file
    
    Returns:
        bool: True if valid DICOM, False otherwise
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return True
    except Exception:
        return False
