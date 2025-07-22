"""
ArchiMed Images V2.0 - Visualization Functions
Functions for creating overlays, progress tracking, and result visualization.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm


def create_separate_lung_overlay(image: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, 
                                combined_mask: np.ndarray, crop_bounds: Tuple[int, int, int, int],
                                config: Dict[str, Any]) -> np.ndarray:
    """
    Create overlay with separate left/right lung visualization.
    
    Args:
        image (np.ndarray): Original image
        left_mask (np.ndarray): Left lung mask
        right_mask (np.ndarray): Right lung mask
        combined_mask (np.ndarray): Combined lung mask
        crop_bounds (Tuple): Crop bounds (y_min, x_min, y_max, x_max)
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        np.ndarray: Overlay image with visualizations
    """
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    crop_y_min, crop_x_min, crop_y_max, crop_x_max = crop_bounds
    img_height, img_width = overlay.shape[:2]
    
    # Darken areas outside the crop region
    outside_crop_mask = np.ones((img_height, img_width), dtype=bool)
    outside_crop_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = False
    overlay[outside_crop_mask] = (overlay[outside_crop_mask] * 0.5).astype(np.uint8)
    
    # Get opacity settings
    fill_opacity = config.get('LUNG_FILL_OPACITY', 0.25)
    border_opacity = config.get('LUNG_BORDER_OPACITY', 0.50)
    
    # Left lung visualization (Blue)
    left_areas = left_mask > 0
    if np.any(left_areas):
        # Blue fill for left lung
        left_fill_colored = np.zeros_like(overlay)
        left_fill_colored[left_areas] = [255, 0, 0]  # Blue in BGR
        fill_opacity_actual = max(fill_opacity, 0.3)
        overlay[left_areas] = cv2.addWeighted(
            overlay[left_areas], 1.0 - fill_opacity_actual,
            left_fill_colored[left_areas], fill_opacity_actual, 0
        )
        
        # Blue border for left lung
        left_mask_uint8 = (left_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(left_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_mask = np.zeros_like(left_mask, dtype=np.uint8)
        cv2.drawContours(border_mask, contours, -1, 1, thickness=4)
        
        border_areas = border_mask > 0
        if np.any(border_areas):
            left_border_colored = np.zeros_like(overlay)
            left_border_colored[border_areas] = [255, 0, 0]  # Blue in BGR
            border_opacity_actual = max(border_opacity, 0.7)
            overlay[border_areas] = cv2.addWeighted(
                overlay[border_areas], 1.0 - border_opacity_actual,
                left_border_colored[border_areas], border_opacity_actual, 0
            )
    
    # Right lung visualization (Red)
    right_areas = right_mask > 0
    if np.any(right_areas):
        # Red fill for right lung
        right_fill_colored = np.zeros_like(overlay)
        right_fill_colored[right_areas] = [0, 0, 255]  # Red in BGR
        fill_opacity_actual = max(fill_opacity, 0.3)
        overlay[right_areas] = cv2.addWeighted(
            overlay[right_areas], 1.0 - fill_opacity_actual,
            right_fill_colored[right_areas], fill_opacity_actual, 0
        )
        
        # Red border for right lung
        right_mask_uint8 = (right_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(right_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_mask = np.zeros_like(right_mask, dtype=np.uint8)
        cv2.drawContours(border_mask, contours, -1, 1, thickness=4)
        
        border_areas = border_mask > 0
        if np.any(border_areas):
            right_border_colored = np.zeros_like(overlay)
            right_border_colored[border_areas] = [0, 0, 255]  # Red in BGR
            border_opacity_actual = max(border_opacity, 0.7)
            overlay[border_areas] = cv2.addWeighted(
                overlay[border_areas], 1.0 - border_opacity_actual,
                right_border_colored[border_areas], border_opacity_actual, 0
            )
    
    # Green corner brackets for combined lung area
    lung_coords = np.where(combined_mask > 0)
    if len(lung_coords[0]) > 0:
        actual_y_min = np.min(lung_coords[0])
        actual_y_max = np.max(lung_coords[0])
        actual_x_min = np.min(lung_coords[1])
        actual_x_max = np.max(lung_coords[1])
        
        if (actual_x_min >= 0 and actual_y_min >= 0 and 
            actual_x_max < overlay.shape[1] and actual_y_max < overlay.shape[0] and
            actual_x_max > actual_x_min and actual_y_max > actual_y_min):
            draw_corner_brackets(overlay, actual_x_min, actual_y_min, actual_x_max, actual_y_max, 
                               (0, 255, 0), 5, 60)
    
    # Cyan rectangle for crop area
    cv2.rectangle(overlay, (crop_x_min, crop_y_min), (crop_x_max, crop_y_max), (255, 255, 0), 2)
    
    # Add legend
    overlay = add_legend_to_overlay(overlay, config)
    
    return overlay


def draw_corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                        color: Tuple[int, int, int], thickness: int = 5, length: int = 60):
    """
    Draw corner brackets around a rectangular area.
    
    Args:
        img (np.ndarray): Image to draw on
        x1, y1 (int): Top-left corner coordinates
        x2, y2 (int): Bottom-right corner coordinates
        color (Tuple[int, int, int]): BGR color
        thickness (int): Line thickness
        length (int): Bracket length
    """
    img_height, img_width = img.shape[:2]
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    max_length_x = min(length, (x2 - x1) // 4)
    max_length_y = min(length, (y2 - y1) // 4)
    length = max(20, min(max_length_x, max_length_y))
    
    # Top-left corner
    cv2.line(img, (x1, y1), (min(x1 + length, img_width-1), y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, min(y1 + length, img_height-1)), color, thickness)
    
    # Top-right corner
    cv2.line(img, (max(x2 - length, 0), y1), (x2, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, min(y1 + length, img_height-1)), color, thickness)
    
    # Bottom-left corner
    cv2.line(img, (x1, max(y2 - length, 0)), (x1, y2), color, thickness)
    cv2.line(img, (x1, y2), (min(x1 + length, img_width-1), y2), color, thickness)
    
    # Bottom-right corner
    cv2.line(img, (x2, max(y2 - length, 0)), (x2, y2), color, thickness)
    cv2.line(img, (max(x2 - length, 0), y2), (x2, y2), color, thickness)


def add_legend_to_overlay(overlay: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Add informative legend to overlay image.
    
    Args:
        overlay (np.ndarray): Overlay image
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        np.ndarray: Overlay with legend
    """
    img_height, img_width = overlay.shape[:2]
    
    # Legend dimensions and position
    legend_height = 200
    legend_width = min(800, img_width - 20)
    legend_y_start = img_height - legend_height - 10
    
    # Create semi-transparent background for legend
    legend_background = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    legend_area = overlay[legend_y_start:legend_y_start + legend_height, 10:10 + legend_width]
    overlay[legend_y_start:legend_y_start + legend_height, 10:10 + legend_width] = cv2.addWeighted(
        legend_area, 0.2, legend_background, 0.8, 0
    )
    
    # Add legend text
    text_y = legend_y_start + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    
    # Color-coded legend text
    cv2.putText(overlay, "BLUE = Left lung", (20, text_y), font, font_scale, (255, 0, 0), thickness)
    cv2.putText(overlay, "RED = Right lung", (250, text_y), font, font_scale, (0, 0, 255), thickness)
    
    cv2.putText(overlay, "GREEN = Combined lung brackets", 
               (20, text_y + 35), font, font_scale, (0, 255, 0), thickness)
    
    target_size = config.get('TARGET_SIZE', (518, 518))
    crop_margin = config.get('CROP_MARGIN', 25)
    cv2.putText(overlay, f"CYAN = Crop {target_size[0]}x{target_size[1]} (margin: {crop_margin}px)", 
               (20, text_y + 70), font, font_scale, (255, 255, 0), thickness)
    
    # Processing features information
    features = []
    if config.get('ENABLE_HISTOGRAM_EQUALIZATION', False): features.append("HistEq")
    if config.get('ENABLE_GAUSSIAN_BLUR', False): features.append("Blur")
    if config.get('USE_MULTIPLE_THRESHOLDS', False): features.append("MultiThresh")
    if config.get('AGGRESSIVE_MORPHOLOGY', False): features.append("AggroMorph")
    if config.get('KEEP_LARGEST_COMPONENT_ONLY', False): features.append("LargestOnly")
    features_text = "+".join(features) if features else "Basic"
    
    model_sensitivity = config.get('MODEL_SENSITIVITY', 0.0001)
    cv2.putText(overlay, f"Sensitivity: {model_sensitivity} | Enhanced: {features_text}", 
               (20, text_y + 105), font, 0.6, (255, 255, 255), 1)
    
    # Get model type from config or default
    model_type = config.get('model_type', 'Unknown')
    cv2.putText(overlay, f"Model: {model_type} | Crop margin: {crop_margin}px", 
               (20, text_y + 130), font, 0.6, (255, 255, 255), 1)
    
    cv2.putText(overlay, "Separate L/R lung detection", 
               (20, text_y + 155), font, 0.6, (255, 255, 255), 1)
    
    return overlay


def save_visualization_files(processing_info: Dict[str, Any], masks_path: str, 
                           original_image: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Save all visualization files (masks and overlay).
    
    Args:
        processing_info (Dict[str, Any]): Processing metadata
        masks_path (str): Output directory for masks
        original_image (np.ndarray): Original image for overlay
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(masks_path, exist_ok=True)
        
        file_id = processing_info['file_id']
        left_mask = processing_info['left_mask']
        right_mask = processing_info['right_mask']
        combined_mask = processing_info['combined_mask']
        crop_bounds = processing_info['crop_bounds']
        
        if not processing_info['segmentation_success'] or crop_bounds is None:
            print(f"⚠️ Skipping visualization for {file_id} - segmentation failed")
            return False
        
        # Crop masks to the same area as the final image
        crop_y_min, crop_x_min, crop_y_max, crop_x_max = crop_bounds
        
        left_mask_cropped = left_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        right_mask_cropped = right_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        combined_mask_cropped = combined_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # Save individual lung masks
        left_mask_path = os.path.join(masks_path, f"{file_id}_left_lung_mask.png")
        right_mask_path = os.path.join(masks_path, f"{file_id}_right_lung_mask.png")
        combined_mask_path = os.path.join(masks_path, f"{file_id}_combined_mask.png")
        
        left_mask_image = (left_mask_cropped * 255).astype(np.uint8)
        right_mask_image = (right_mask_cropped * 255).astype(np.uint8)
        combined_mask_image = (combined_mask_cropped * 255).astype(np.uint8)
        
        cv2.imwrite(left_mask_path, left_mask_image)
        cv2.imwrite(right_mask_path, right_mask_image)
        cv2.imwrite(combined_mask_path, combined_mask_image)
        
        # Save overlay with visualization
        overlay_path = os.path.join(masks_path, f"{file_id}_overlay.png")
        overlay = create_separate_lung_overlay(original_image, left_mask, right_mask, 
                                             combined_mask, crop_bounds, config)
        cv2.imwrite(overlay_path, overlay)
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving visualization files for {file_id}: {e}")
        return False


def create_progress_tracker(total_files: int, description: str = "Processing files") -> tqdm:
    """
    Create a progress tracker for file processing.
    
    Args:
        total_files (int): Total number of files to process
        description (str): Description for the progress bar
    
    Returns:
        tqdm: Progress tracker object
    """
    return tqdm(total=total_files, desc=description, unit="file")


def print_processing_summary(results: Dict[str, Any], config: Dict[str, Any]):
    """
    Print a comprehensive summary of processing results.
    
    Args:
        results (Dict[str, Any]): Processing results
        config (Dict[str, Any]): Configuration parameters
    """
    print("=" * 70)
    print("🎉 Processing completed!")
    print("=" * 70)
    
    print("\n📊 Results Summary:")
    print(f"   • Total files found: {results.get('total_files_found', 0)}")
    print(f"   • Successfully processed: {results.get('successfully_processed', 0)}")
    print(f"   • Segmentation successes: {results.get('segmentation_successes', 0)}")
    print(f"   • Failed conversions: {results.get('failed_conversions', 0)}")
    print(f"   • Skipped (already exist): {results.get('skipped_existing', 0)}")
    
    # Output format information
    output_format = config.get('OUTPUT_FORMAT', 'png').upper()
    target_size = config.get('TARGET_SIZE', (518, 518))
    print(f"\n📁 Output Information:")
    print(f"   • Format: {output_format}")
    print(f"   • Size: {target_size[0]}x{target_size[1]}")
    print(f"   • Images directory: {config.get('IMAGES_PATH', 'N/A')}")
    
    if config.get('SAVE_MASKS', False):
        print(f"   • Masks directory: {config.get('MASKS_PATH', 'N/A')}")
        print(f"   • Created mask files: {results.get('mask_files_created', 0)}")
    
    # Processing features summary
    print(f"\n🔧 Processing Features:")
    model_type = results.get('model_type', 'Unknown')
    print(f"   • Segmentation model: {model_type}")
    print(f"   • Sensitivity: {config.get('MODEL_SENSITIVITY', 0.0001)}")
    
    features = []
    if config.get('ENABLE_HISTOGRAM_EQUALIZATION', False): features.append("Histogram Equalization")
    if config.get('ENABLE_GAUSSIAN_BLUR', False): features.append("Gaussian Blur")
    if config.get('USE_MULTIPLE_THRESHOLDS', False): features.append("Multiple Thresholds")
    if config.get('AGGRESSIVE_MORPHOLOGY', False): features.append("Aggressive Morphology")
    if config.get('KEEP_LARGEST_COMPONENT_ONLY', False): features.append("Largest Component Only")
    
    if features:
        print(f"   • Enhancements: {', '.join(features)}")
    else:
        print(f"   • Enhancements: Basic processing")
    
    # Error reporting
    errors = results.get('errors', [])
    if errors:
        print(f"\n❌ Errors encountered: {len(errors)}")
        for i, error in enumerate(errors[:3]):  # Show first 3 errors
            print(f"     {i+1}. {error}")
        if len(errors) > 3:
            print(f"     ... and {len(errors) - 3} more errors")
    
    print(f"\n🏁 Processing pipeline completed successfully!")


def print_archimed_connection_status(authenticated: bool, files_available: int = 0):
    """
    Print ArchiMed connection status information.
    
    Args:
        authenticated (bool): Whether ArchiMed connection is successful
        files_available (int): Number of files available from ArchiMed
    """
    print("\n🔗 ArchiMed Connection Status:")
    if authenticated:
        print("   ✅ Connected to ArchiMed successfully")
        if files_available > 0:
            print(f"   📋 Found {files_available} files in CSV")
        else:
            print("   ⚠️  No files found in CSV or CSV file missing")
    else:
        print("   ❌ ArchiMed connection failed")
        print("   💡 Tip: Check credentials and network connection")
        print("   🔄 Will process local files only")


def print_file_discovery_summary(local_files: int, downloaded_files: int, failed_downloads: int):
    """
    Print file discovery and download summary.
    
    Args:
        local_files (int): Number of local files found
        downloaded_files (int): Number of files successfully downloaded
        failed_downloads (int): Number of failed downloads
    """
    print(f"\n📁 File Discovery Summary:")
    print(f"   • Local files found: {local_files}")
    if downloaded_files > 0:
        print(f"   • Successfully downloaded: {downloaded_files}")
    if failed_downloads > 0:
        print(f"   • Failed downloads: {failed_downloads}")
    
    total_files = local_files + downloaded_files
    print(f"   🎯 Total files ready for processing: {total_files}")


def display_sample_results(results: Dict[str, Any], config: Dict[str, Any], num_samples: int = 3):
    """
    Display information about sample processed files.
    
    Args:
        results (Dict[str, Any]): Processing results
        config (Dict[str, Any]): Configuration parameters
        num_samples (int): Number of sample files to display
    """
    processed_files = results.get('processed_files', [])
    if not processed_files:
        return
    
    print(f"\n📋 Sample Results (showing {min(num_samples, len(processed_files))} files):")
    
    for i, file_info in enumerate(processed_files[:num_samples]):
        file_id = file_info.get('file_id', 'Unknown')
        success = file_info.get('success', False)
        segmentation_success = file_info.get('segmentation_success', False)
        
        status_icon = "✅" if success else "❌"
        seg_icon = "🫁" if segmentation_success else "⚠️"
        
        print(f"   {i+1}. {file_id} {status_icon}")
        if success:
            output_format = config.get('OUTPUT_FORMAT', 'png').upper()
            print(f"      • Format: {output_format}")
            print(f"      • Segmentation: {seg_icon}")
            if config.get('SAVE_MASKS', False) and segmentation_success:
                print(f"      • Masks: Created (L/R/Combined/Overlay)")


def save_processing_log(results: Dict[str, Any], config: Dict[str, Any], log_path: str):
    """
    Save detailed processing log to file.
    
    Args:
        results (Dict[str, Any]): Processing results
        config (Dict[str, Any]): Configuration parameters
        log_path (str): Path to save log file
    """
    try:
        import json
        from datetime import datetime
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': results,
            'version': 'ArchiMed Images V2.0'
        }
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"\n📝 Detailed log saved to: {log_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to save processing log: {e}")
