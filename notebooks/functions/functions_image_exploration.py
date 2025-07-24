"""
ArchiMed Images V2.0 - Image Exploration Functions
Functions for displaying and exporting pipeline images with support for PNG and NIFTI formats.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython.display import display, HTML
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Dict, Any
import nibabel as nib


# Colors for output
ANSI = {
    'R': '\033[91m', 'G': '\033[92m', 'B': '\033[94m', 'Y': '\033[93m',
    'W': '\033[0m', 'M': '\033[95m', 'C': '\033[96m'
}


def load_image_file(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Load an image file (PNG or NIFTI) and optionally resize it.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    target_size : Optional[Tuple[int, int]]
        Target size for resizing (width, height)
    
    Returns:
    --------
    Optional[np.ndarray]
        Loaded image array or None if failed
    """
    try:
        if image_path.lower().endswith(('.nii', '.nii.gz')):
            # Load NIFTI file (2D images)
            nifti_img = nib.load(image_path)
            img_data = nifti_img.get_fdata()
            
            # Handle potential 3D NIFTI with single slice (squeeze any singleton dimensions)
            img_data = np.squeeze(img_data)
            
            # Ensure we have 2D data
            if len(img_data.shape) > 2:
                print(f"{ANSI['Y']}Warning: Unexpected NIFTI shape {img_data.shape} for {image_path}{ANSI['W']}")
                return None
            
            # Normalize to 0-255 range
            img_data = np.clip(img_data, 0, np.percentile(img_data, 99))
            if img_data.max() > img_data.min():
                img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
            else:
                img_data = np.zeros_like(img_data, dtype=np.uint8)
            
            # Fix NIFTI orientation (transpose to correct counterclockwise rotation, then flip horizontally)
            img_data = np.transpose(img_data)
            img_data = np.fliplr(img_data)  # Flip left-right to correct horizontal mirroring
            
        else:
            # Load PNG/JPG file
            img = Image.open(image_path)
            img_data = np.array(img)
        
        # Resize if target size specified
        if target_size is not None:
            if len(img_data.shape) == 3:
                img_data = cv2.resize(img_data, target_size)
            else:
                img_data = cv2.resize(img_data, target_size)
        
        return img_data
        
    except Exception as e:
        print(f"{ANSI['R']}Error loading {image_path}: {e}{ANSI['W']}")
        return None


def find_file_ids_from_path(images_path: str, file_extensions: List[str] = ['.png', '.nii', '.nii.gz']) -> List[str]:
    """
    Find all unique FileIDs from an image path.
    
    Parameters:
    -----------
    images_path : str
        Path to search for images
    file_extensions : List[str]
        List of file extensions to search for
    
    Returns:
    --------
    List[str]
        Sorted list of unique FileIDs
    """
    all_file_ids = set()
    
    if not os.path.exists(images_path):
        print(f"{ANSI['Y']}Warning: Path does not exist: {images_path}{ANSI['W']}")
        return []
        
    for ext in file_extensions:
        pattern = os.path.join(images_path, f"*{ext}")
        files = glob.glob(pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # Remove extension(s)
            file_id = filename
            for extension in file_extensions:
                if file_id.endswith(extension):
                    file_id = file_id[:-len(extension)]
                    break
            all_file_ids.add(file_id)
    
    return sorted(list(all_file_ids))


def find_image_for_file_id(file_id: str, images_path: str, file_extensions: List[str] = ['.png', '.nii', '.nii.gz']) -> Optional[str]:
    """
    Find the image file for a given FileID in the specified path.
    
    Parameters:
    -----------
    file_id : str
        The FileID to search for
    images_path : str
        Path to search in
    file_extensions : List[str]
        List of extensions to try
    
    Returns:
    --------
    Optional[str]
        Path to the found image file or None
    """
    if not os.path.exists(images_path):
        return None
        
    for ext in file_extensions:
        potential_path = os.path.join(images_path, f"{file_id}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    
    return None


def display_pipeline_images(
    images_path: str,
    masks_path: str,
    nb_images_to_display: Optional[int] = None,
    filter_file_ids: Optional[List[str]] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Display images in 3-column pipeline format: Overlay → Mask → Final Result
    
    Parameters:
    -----------
    images_path : str
        Path containing final processed images (PNG or NIFTI)
    masks_path : str
        Path containing overlay and mask images
    nb_images_to_display : Optional[int]
        Number of image rows to display (each row contains 3 images)
    filter_file_ids : Optional[List[str]]
        List of FileIDs to display. If None, display all available FileIDs
    target_size : Optional[Tuple[int, int]]
        Target size for image display (width, height)
    """
    
    print(f"{ANSI['C']}Loading 3-column pipeline view...{ANSI['W']}")
    
    # Step 1: Get all available FileIDs from final images
    available_file_ids = find_file_ids_from_path(images_path)
    
    if not available_file_ids:
        print(f"{ANSI['R']}No final images found in: {images_path}{ANSI['W']}")
        return
    
    print(f"{ANSI['B']}Found {len(available_file_ids)} FileIDs in final images{ANSI['W']}")
    
    # Step 2: Apply filter if provided
    if filter_file_ids:
        filtered_file_ids = [fid for fid in available_file_ids if fid in filter_file_ids]
        print(f"{ANSI['Y']}Filter applied: {len(filtered_file_ids)}/{len(available_file_ids)} FileIDs match filter{ANSI['W']}")
        file_ids_to_display = filtered_file_ids
    else:
        file_ids_to_display = available_file_ids
        print(f"{ANSI['G']}No filter applied - using all available FileIDs{ANSI['W']}")
    
    if not file_ids_to_display:
        print(f"{ANSI['Y']}No FileIDs to display after filtering{ANSI['W']}")
        return
    
    # Step 3: Limit number of images if specified
    if nb_images_to_display and nb_images_to_display < len(file_ids_to_display):
        file_ids_to_display = file_ids_to_display[:nb_images_to_display]
        print(f"{ANSI['C']}Limited to {nb_images_to_display} image rows{ANSI['W']}")
    
    num_rows = len(file_ids_to_display)
    print(f"{ANSI['G']}Display grid: {num_rows} rows × 3 columns{ANSI['W']}")
    
    # Step 4: Create matplotlib figure with 3 columns
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    
    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    column_titles = ['Overlay', 'Mask', 'Final']
    
    # Step 5: Display images for each FileID
    for row_idx, file_id in enumerate(file_ids_to_display):
        print(f"{ANSI['C']}Processing FileID: {file_id}{ANSI['W']}")
        
        # Find corresponding images for this FileID
        overlay_path = os.path.join(masks_path, f"{file_id}_overlay.png")
        mask_path = os.path.join(masks_path, f"{file_id}_combined_mask.png")
        final_path = find_image_for_file_id(file_id, images_path)
        
        # Check if files exist
        if not os.path.exists(overlay_path):
            overlay_path = None
        if not os.path.exists(mask_path):
            mask_path = None
        
        # Display images in 3 columns
        image_paths = [overlay_path, mask_path, final_path]
        image_labels = [f"{file_id}_overlay", f"{file_id}_mask", file_id]
        
        for col_idx, (img_path, img_label) in enumerate(zip(image_paths, image_labels)):
            try:
                if img_path and os.path.exists(img_path):
                    # Load and display image
                    img_data = load_image_file(img_path, target_size)
                    if img_data is not None:
                        axes[row_idx, col_idx].imshow(img_data, cmap='gray' if len(img_data.shape) == 2 else None)
                        axes[row_idx, col_idx].set_title(f"{column_titles[col_idx]}\n{img_label}", fontsize=10, pad=10)
                    else:
                        raise Exception("Failed to load image")
                else:
                    # Missing image
                    axes[row_idx, col_idx].text(0.5, 0.5, f'Missing\n{column_titles[col_idx]}\n{img_label}', 
                                               ha='center', va='center', transform=axes[row_idx, col_idx].transAxes,
                                               fontsize=12, color='red')
                    axes[row_idx, col_idx].set_title(f"{column_titles[col_idx]}\n{img_label} (MISSING)", fontsize=10, pad=10)
                
                axes[row_idx, col_idx].axis('off')
                
            except Exception as e:
                # Handle broken images
                axes[row_idx, col_idx].text(0.5, 0.5, f'Error loading\n{img_label}', 
                                           ha='center', va='center', transform=axes[row_idx, col_idx].transAxes,
                                           fontsize=12, color='red')
                axes[row_idx, col_idx].set_title(f"{column_titles[col_idx]}\n{img_label} (ERROR)", fontsize=10, pad=10)
                axes[row_idx, col_idx].axis('off')
                print(f"{ANSI['R']}Error loading {img_path}: {e}{ANSI['W']}")
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\n{ANSI['G']}Successfully displayed {num_rows} image rows (pipeline view){ANSI['W']}")
    if filter_file_ids:
        print(f"{ANSI['C']}Filter: {len(filter_file_ids)} FileIDs specified{ANSI['W']}")
    if nb_images_to_display:
        print(f"{ANSI['Y']}Limited to: {nb_images_to_display} rows{ANSI['W']}")
    
    print(f"{ANSI['M']}Pipeline: Overlay → Mask → Final Result{ANSI['W']}")


def export_pipeline_images(
    images_path: str,
    masks_path: str,
    export_dir: str,
    fileid_filter: Optional[List[str]] = None,
    nb_rows: int = 10,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
    target_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Export images in 3-column pipeline format to files.
    
    Parameters:
    -----------
    images_path : str
        Path containing final processed images
    masks_path : str
        Path containing overlay and mask images  
    export_dir : str
        Directory to save exported files
    fileid_filter : Optional[List[str]]
        List of FileIDs to ignore during processing
    nb_rows : int
        Number of rows per exported graph
    figsize : Tuple[int, int]
        Figure size tuple
    dpi : int
        Image resolution
    target_size : Optional[Tuple[int, int]]
        Target size for image display
    """
    
    os.makedirs(export_dir, exist_ok=True)
    
    # Get all FileIDs
    file_ids = find_file_ids_from_path(images_path)
    if not file_ids:
        print(f"{ANSI['R']}No images found in: {images_path}{ANSI['W']}")
        return
    
    # Apply filter
    if fileid_filter:
        file_ids = [fid for fid in file_ids if fid not in fileid_filter]
        print(f"{ANSI['Y']}Filtered out {len(fileid_filter)} FileIDs{ANSI['W']}")
    
    file_ids.sort()
    print(f"{ANSI['G']}Exporting {len(file_ids)} FileIDs{ANSI['W']}")
    
    # Split into chunks
    chunks = [file_ids[i:i+nb_rows] for i in range(0, len(file_ids), nb_rows)]
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Exporting chunks")):
        fig, axes = plt.subplots(len(chunk), 3, figsize=(figsize[0], figsize[1]*len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, file_id in enumerate(chunk):
            # Find image paths
            overlay_path = os.path.join(masks_path, f"{file_id}_overlay.png")
            mask_path = os.path.join(masks_path, f"{file_id}_combined_mask.png")
            final_path = find_image_for_file_id(file_id, images_path)
            
            # Check if files exist
            if not os.path.exists(overlay_path):
                overlay_path = None
            if not os.path.exists(mask_path):
                mask_path = None
            
            # Display images
            for col_idx, (img_path, title) in enumerate(zip([overlay_path, mask_path, final_path], ['Overlay', 'Mask', 'Final'])):
                try:
                    if img_path and os.path.exists(img_path):
                        img_data = load_image_file(img_path, target_size)
                        if img_data is not None:
                            axes[row_idx, col_idx].imshow(img_data, cmap='gray' if len(img_data.shape) == 2 else None)
                        else:
                            raise Exception("Failed to load image")
                    else:
                        axes[row_idx, col_idx].text(0.5, 0.5, f'Missing\n{title}', ha='center', va='center', 
                                                  transform=axes[row_idx, col_idx].transAxes, fontsize=12, color='red')
                except:
                    axes[row_idx, col_idx].text(0.5, 0.5, f'Error\n{title}', ha='center', va='center', 
                                              transform=axes[row_idx, col_idx].transAxes, fontsize=12, color='red')
                
                axes[row_idx, col_idx].set_title(f"{title}\n{file_id}", fontsize=10)
                axes[row_idx, col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(export_dir, f"pipeline_batch_{chunk_idx+1:03d}.png"), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    print(f"{ANSI['G']}Exported {len(chunks)} files to {export_dir}{ANSI['W']}")


def get_available_file_ids(images_path: str) -> List[str]:
    """
    Get list of available FileIDs from image path.
    
    Parameters:
    -----------
    images_path : str
        Path to search for images
    
    Returns:
    --------
    List[str]
        Sorted list of available FileIDs
    """
    return find_file_ids_from_path(images_path)


def validate_pipeline_paths(images_path: str, masks_path: str) -> Dict[str, Any]:
    """
    Validate that the required paths exist and contain files.
    
    Parameters:
    -----------
    images_path : str
        Path containing final images
    masks_path : str
        Path containing mask files
    
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    results = {
        'images_path_valid': False,
        'masks_path_valid': False,
        'file_ids_found': 0,
        'overlay_files_found': 0,
        'mask_files_found': 0,
        'image_format': 'unknown'
    }
    
    # Check image path
    results['images_path_valid'] = os.path.exists(images_path)
    
    # Check masks path
    results['masks_path_valid'] = os.path.exists(masks_path)
    
    # Count files and detect format
    if results['images_path_valid']:
        file_ids = find_file_ids_from_path(images_path)
        results['file_ids_found'] = len(file_ids)
        
        # Detect image format
        png_files = glob.glob(os.path.join(images_path, "*.png"))
        nifti_files = glob.glob(os.path.join(images_path, "*.nii")) + glob.glob(os.path.join(images_path, "*.nii.gz"))
        
        if png_files and nifti_files:
            results['image_format'] = 'mixed (PNG + NIFTI)'
        elif png_files:
            results['image_format'] = 'PNG'
        elif nifti_files:
            results['image_format'] = 'NIFTI'
        else:
            results['image_format'] = 'none detected'
    
    if results['masks_path_valid']:
        overlay_files = glob.glob(os.path.join(masks_path, "*_overlay.png"))
        mask_files = glob.glob(os.path.join(masks_path, "*_combined_mask.png"))
        results['overlay_files_found'] = len(overlay_files)
        results['mask_files_found'] = len(mask_files)
    
    return results


def print_validation_summary(validation_results: Dict[str, Any], images_path: str, masks_path: str) -> None:
    """
    Print a summary of path validation results.
    
    Parameters:
    -----------
    validation_results : Dict[str, Any]
        Results from validate_pipeline_paths
    images_path : str
        Path that was validated for images
    masks_path : str
        Path that was validated for masks
    """
    print(f"{ANSI['C']}=== Pipeline Path Validation ==={ANSI['W']}")
    
    # Image path
    images_status = "✅" if validation_results['images_path_valid'] else "❌"
    print(f"{ANSI['G']}Images path: {images_status} {images_path}{ANSI['W']}")
    
    # Masks path
    masks_status = "✅" if validation_results['masks_path_valid'] else "❌"
    print(f"{ANSI['G']}Masks path: {masks_status} {masks_path}{ANSI['W']}")
    
    # File counts and format
    print(f"{ANSI['B']}Files found:{ANSI['W']}")
    print(f"  • FileIDs: {validation_results['file_ids_found']}")
    print(f"  • Image format: {validation_results['image_format']}")
    print(f"  • Overlay files: {validation_results['overlay_files_found']}")
    print(f"  • Mask files: {validation_results['mask_files_found']}")
    
    # Overall status
    overall_valid = (
        validation_results['images_path_valid'] and
        validation_results['masks_path_valid'] and
        validation_results['file_ids_found'] > 0
    )
    
    status_color = ANSI['G'] if overall_valid else ANSI['R']
    status_text = "READY" if overall_valid else "ISSUES DETECTED"
    print(f"\n{status_color}Overall Status: {status_text}{ANSI['W']}")


def detect_image_format(images_path: str) -> str:
    """
    Detect the image format in the given path.
    
    Parameters:
    -----------
    images_path : str
        Path to check for image formats
    
    Returns:
    --------
    str
        Detected format: 'PNG', 'NIFTI', 'mixed', or 'none'
    """
    if not os.path.exists(images_path):
        return 'path_not_found'
    
    png_files = glob.glob(os.path.join(images_path, "*.png"))
    nifti_files = glob.glob(os.path.join(images_path, "*.nii")) + glob.glob(os.path.join(images_path, "*.nii.gz"))
    
    if png_files and nifti_files:
        return 'mixed'
    elif png_files:
        return 'PNG'
    elif nifti_files:
        return 'NIFTI'
    else:
        return 'none'
