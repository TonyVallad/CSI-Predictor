"""
ArchiMed Images V2.0 - Pre-processing Functions
Functions for image enhancement, lung segmentation, and processing operations.
"""

import numpy as np
import cv2
import torch
import sys
import subprocess
from typing import Tuple, Optional, Dict, Any
import warnings


# Global variables for segmentation model
segmentation_model = None
model_type = None


def initialize_segmentation_model() -> Tuple[Any, str]:
    """
    Initialize the lung segmentation model (TorchXRayVision or fallback).
    
    Returns:
        Tuple[model, model_type]: Segmentation model and type identifier
    """
    global segmentation_model, model_type
    
    try:
        import torchxrayvision as xrv
        segmentation_model = xrv.baseline_models.chestx_det.PSPNet()
        model_type = 'torchxray'
        print("✅ TorchXRayVision model loaded successfully")
        
    except ImportError:
        try:
            print("📦 Installing TorchXRayVision...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchxrayvision"])
            import torchxrayvision as xrv
            segmentation_model = xrv.baseline_models.chestx_det.PSPNet()
            model_type = 'torchxray'
            print("✅ TorchXRayVision installed and loaded successfully")
            
        except Exception as e:
            print(f"⚠️ TorchXRayVision unavailable: {e}")
            print("🔄 Using fallback segmentation method")
            segmentation_model = None
            model_type = 'fallback'
    
    return segmentation_model, model_type


def enhance_image_preprocessing(image: np.ndarray, enable_hist_eq: bool = True, 
                              enable_blur: bool = True) -> np.ndarray:
    """
    Apply enhanced preprocessing for better segmentation detection.
    
    Args:
        image (np.ndarray): Input image
        enable_hist_eq (bool): Enable histogram equalization
        enable_blur (bool): Enable Gaussian blur
    
    Returns:
        np.ndarray: Enhanced image
    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image.copy()
    
    # Histogram equalization for better contrast
    if enable_hist_eq:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image_gray = clahe.apply(image_gray)
    
    # Gaussian blur to reduce noise
    if enable_blur:
        image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
    
    return image_gray


def keep_largest_component(binary_mask: np.ndarray, enable_filtering: bool = True) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    
    Args:
        binary_mask (np.ndarray): Binary mask (0s and 1s)
        enable_filtering (bool): Enable component filtering
    
    Returns:
        np.ndarray: Binary mask with only the largest connected component
    """
    if not enable_filtering:
        return binary_mask
    
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:  # No components found (only background)
        return binary_mask
    
    # Find the largest component (excluding background which is label 0)
    largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    
    # Create mask with only the largest component
    largest_component_mask = (labels == largest_component_label).astype(np.uint8)
    
    return largest_component_mask


def segment_with_torchxray_separate(image: np.ndarray, sensitivity: float = 0.0001,
                                  use_multiple_thresholds: bool = True,
                                  aggressive_morphology: bool = True,
                                  enable_debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment lungs using TorchXRayVision with separate left/right detection.
    
    Args:
        image (np.ndarray): Input image
        sensitivity (float): Detection sensitivity threshold
        use_multiple_thresholds (bool): Try multiple threshold values
        aggressive_morphology (bool): Use aggressive morphological operations
        enable_debug (bool): Enable debug output
    
    Returns:
        Tuple[left_mask, right_mask, combined_mask]: Separated and combined lung masks
    """
    try:
        import torchxrayvision as xrv
        
        # Enhanced preprocessing
        image_preprocessed = enhance_image_preprocessing(image)
        
        # Normalize and prepare for model
        image_norm = xrv.datasets.normalize(image_preprocessed, 255)
        image_norm = image_norm[None, ...]
        
        transform = xrv.datasets.XRayResizer(512)
        image_resized = transform(image_norm)
        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)
        
        # Run segmentation
        with torch.no_grad():
            output = segmentation_model(image_tensor)
        
        # Separate left and right lung masks
        left_lung_mask = np.zeros((512, 512))
        right_lung_mask = np.zeros((512, 512))
        
        for i, target in enumerate(segmentation_model.targets):
            if target == 'Left Lung':
                left_lung_mask = output[0, i].cpu().numpy()
            elif target == 'Right Lung':
                right_lung_mask = output[0, i].cpu().numpy()
        
        if enable_debug:
            print(f"Left lung - Min: {left_lung_mask.min():.6f}, Max: {left_lung_mask.max():.6f}")
            print(f"Right lung - Min: {right_lung_mask.min():.6f}, Max: {right_lung_mask.max():.6f}")
        
        # Resize masks to original image size
        left_lung_mask = cv2.resize(left_lung_mask, (image.shape[1], image.shape[0]))
        right_lung_mask = cv2.resize(right_lung_mask, (image.shape[1], image.shape[0]))
        
        def process_lung_mask(lung_mask, lung_name):
            """Process individual lung mask with thresholding and morphology."""
            if use_multiple_thresholds:
                thresholds = [sensitivity, sensitivity * 0.5, sensitivity * 0.1, 0.0001]
                best_mask = None
                best_ratio = 0
                
                for threshold in thresholds:
                    binary_mask = (lung_mask > threshold).astype(np.uint8)
                    lung_ratio = np.sum(binary_mask) / (binary_mask.shape[0] * binary_mask.shape[1])
                    
                    if enable_debug:
                        print(f"{lung_name} threshold {threshold:.6f}: Lung ratio = {lung_ratio:.4f}")
                    
                    # Target reasonable lung area (1%-25% of image for individual lungs)
                    if 0.01 <= lung_ratio <= 0.25:
                        best_mask = binary_mask
                        best_ratio = lung_ratio
                        break
                    elif lung_ratio > 0.0005 and best_mask is None:
                        best_mask = binary_mask
                        best_ratio = lung_ratio
                
                if best_mask is not None:
                    binary_mask = best_mask
                    if enable_debug:
                        print(f"{lung_name} selected mask with ratio: {best_ratio:.4f}")
                else:
                    binary_mask = (lung_mask > 0.0001).astype(np.uint8)
                    if enable_debug:
                        print(f"{lung_name} using ultra-low threshold fallback")
            else:
                binary_mask = (lung_mask > sensitivity).astype(np.uint8)
            
            # Morphological operations
            if aggressive_morphology:
                kernel_size = 15 if np.sum(binary_mask) > 500 else 20
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                
                # Fill holes
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(binary_mask, contours, 1)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            return binary_mask
        
        # Process each lung mask separately
        left_binary_mask = process_lung_mask(left_lung_mask, "Left lung")
        right_binary_mask = process_lung_mask(right_lung_mask, "Right lung")
        
        # Apply largest component filtering to each lung separately
        left_binary_mask = keep_largest_component(left_binary_mask)
        right_binary_mask = keep_largest_component(right_binary_mask)
        
        # Create combined mask
        combined_mask = np.logical_or(left_binary_mask, right_binary_mask).astype(np.uint8)
        
        return left_binary_mask, right_binary_mask, combined_mask
        
    except Exception as e:
        if enable_debug:
            print(f"TorchXRay segmentation failed: {e}")
        return segment_with_fallback_separate(image, aggressive_morphology)


def segment_with_fallback_separate(image: np.ndarray, aggressive_morphology: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback segmentation using classical computer vision methods.
    
    Args:
        image (np.ndarray): Input image
        aggressive_morphology (bool): Use aggressive morphological operations
    
    Returns:
        Tuple[left_mask, right_mask, combined_mask]: Same mask for both lungs (can't differentiate)
    """
    # Enhanced fallback with preprocessing
    image_preprocessed = enhance_image_preprocessing(image)
    
    # Otsu thresholding
    _, otsu_mask = cv2.threshold(image_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel_size = 25 if aggressive_morphology else 20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    
    combined_mask = (mask_clean > 0).astype(np.uint8)
    
    # Apply largest component filtering
    combined_mask = keep_largest_component(combined_mask)
    
    # For fallback, return the same mask for both lungs (can't differentiate)
    return combined_mask, combined_mask, combined_mask


def segment_lungs_separate(image: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function to segment lungs with separate left/right detection.
    
    Args:
        image (np.ndarray): Input image
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        Tuple[left_mask, right_mask, combined_mask]: Separated and combined lung masks
    """
    global segmentation_model, model_type
    
    if model_type == 'torchxray' and segmentation_model is not None:
        return segment_with_torchxray_separate(
            image,
            sensitivity=config.get('MODEL_SENSITIVITY', 0.0001),
            use_multiple_thresholds=config.get('USE_MULTIPLE_THRESHOLDS', True),
            aggressive_morphology=config.get('AGGRESSIVE_MORPHOLOGY', True),
            enable_debug=config.get('ENABLE_DEBUG_OUTPUT', False)
        )
    else:
        return segment_with_fallback_separate(
            image,
            aggressive_morphology=config.get('AGGRESSIVE_MORPHOLOGY', True)
        )


def calculate_crop_bounds(combined_mask: np.ndarray, image_shape: Tuple[int, int], 
                         target_size: Tuple[int, int], crop_margin: int) -> Tuple[int, int, int, int]:
    """
    Calculate proper crop bounds with correct aspect ratio and margin handling.
    
    Args:
        combined_mask (np.ndarray): Binary mask of detected lungs
        image_shape (Tuple[int, int]): (height, width) of the original image
        target_size (Tuple[int, int]): (width, height) target size for final output
        crop_margin (int): Margin around lungs in pixels
    
    Returns:
        Tuple[y_min, x_min, y_max, x_max]: Crop bounds
    """
    img_height, img_width = image_shape[:2]
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height
    
    # Find lung coordinates
    coords = np.column_stack(np.where(combined_mask > 0))
    if len(coords) == 0:
        # No lungs found, return center crop with target aspect ratio
        if img_width / img_height > target_aspect_ratio:
            crop_height = img_height
            crop_width = int(crop_height * target_aspect_ratio)
        else:
            crop_width = img_width
            crop_height = int(crop_width / target_aspect_ratio)
        
        center_x, center_y = img_width // 2, img_height // 2
        x_min = max(0, center_x - crop_width // 2)
        y_min = max(0, center_y - crop_height // 2)
        x_max = min(img_width, x_min + crop_width)
        y_max = min(img_height, y_min + crop_height)
        
        return y_min, x_min, y_max, x_max
    
    # Get lung bounding box with margin
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add margin around lungs
    y_min_padded = max(0, y_min - crop_margin)
    x_min_padded = max(0, x_min - crop_margin)
    y_max_padded = min(img_height, y_max + crop_margin)
    x_max_padded = min(img_width, x_max + crop_margin)
    
    # Get dimensions and center of padded lung area
    lung_width = x_max_padded - x_min_padded
    lung_height = y_max_padded - y_min_padded
    lung_center_x = (x_min_padded + x_max_padded) // 2
    lung_center_y = (y_min_padded + y_max_padded) // 2
    
    # Calculate crop size maintaining target aspect ratio
    min_width = lung_width
    min_height = lung_height
    
    # Adjust to maintain aspect ratio
    if min_width / min_height > target_aspect_ratio:
        crop_width = min_width
        crop_height = int(crop_width / target_aspect_ratio)
    else:
        crop_height = min_height
        crop_width = int(crop_height * target_aspect_ratio)
    
    # Center the crop around lung center
    x_min_crop = lung_center_x - crop_width // 2
    y_min_crop = lung_center_y - crop_height // 2
    x_max_crop = x_min_crop + crop_width
    y_max_crop = y_min_crop + crop_height
    
    # Ensure we don't go outside image bounds
    if x_min_crop < 0:
        x_max_crop -= x_min_crop
        x_min_crop = 0
    if y_min_crop < 0:
        y_max_crop -= y_min_crop
        y_min_crop = 0
    if x_max_crop > img_width:
        x_min_crop -= (x_max_crop - img_width)
        x_max_crop = img_width
    if y_max_crop > img_height:
        y_min_crop -= (y_max_crop - img_height)
        y_max_crop = img_height
    
    # Final bounds check
    x_min_crop = max(0, x_min_crop)
    y_min_crop = max(0, y_min_crop)
    x_max_crop = min(img_width, x_max_crop)
    y_max_crop = min(img_height, y_max_crop)
    
    return y_min_crop, x_min_crop, y_max_crop, x_max_crop


def process_image_with_segmentation(image_array: np.ndarray, file_id: str, 
                                  config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process image with lung segmentation and return cropped image with metadata.
    
    Args:
        image_array (np.ndarray): Input image array
        file_id (str): File identifier
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        Tuple[processed_image, processing_info]: Processed image and metadata
    """
    try:
        # Segment lungs
        left_binary_mask, right_binary_mask, combined_mask = segment_lungs_separate(image_array, config)
        
        # Calculate crop bounds
        crop_y_min, crop_x_min, crop_y_max, crop_x_max = calculate_crop_bounds(
            combined_mask, 
            image_array.shape, 
            config.get('TARGET_SIZE', (518, 518)), 
            config.get('CROP_MARGIN', 25)
        )
        
        # Crop the final image
        if len(image_array.shape) == 3:
            cropped = image_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
        else:
            cropped = image_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # Prepare processing metadata
        processing_info = {
            'file_id': file_id,
            'left_mask': left_binary_mask,
            'right_mask': right_binary_mask,
            'combined_mask': combined_mask,
            'crop_bounds': (crop_y_min, crop_x_min, crop_y_max, crop_x_max),
            'original_shape': image_array.shape,
            'cropped_shape': cropped.shape,
            'segmentation_success': True
        }
        
        return cropped, processing_info
        
    except Exception as e:
        print(f"⚠️ Segmentation failed for {file_id}: {e}")
        processing_info = {
            'file_id': file_id,
            'left_mask': None,
            'right_mask': None,
            'combined_mask': None,
            'crop_bounds': None,
            'original_shape': image_array.shape,
            'cropped_shape': image_array.shape,
            'segmentation_success': False
        }
        return image_array, processing_info
