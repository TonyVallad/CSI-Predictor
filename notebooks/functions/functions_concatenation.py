"""
ArchiMed Images V2.0 - Concatenation Functions
Functions for concatenating exported pipeline images horizontally with separators.
"""

import os
import re
import glob
from PIL import Image
from typing import List, Tuple, Optional
from tqdm.auto import tqdm


# Colors for output
ANSI = {
    'R': '\033[91m', 'G': '\033[92m', 'B': '\033[94m', 'Y': '\033[93m',
    'W': '\033[0m', 'M': '\033[95m', 'C': '\033[96m'
}


def concat_pipeline_images(
    export_dir: str,
    num_images_per_batch: int,
    output_dir: Optional[str] = None,
    separator_width: int = 50,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    separator_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Concatenate pipeline images horizontally in batches with separators.
    
    Parameters:
    -----------
    export_dir : str
        Directory containing pipeline_batch images to concatenate
    num_images_per_batch : int
        Number of images to stitch together in each batch
    output_dir : Optional[str]
        Directory to save concatenated images. Defaults to same as export_dir
    separator_width : int
        Width of black separator lines between images
    background_color : Tuple[int, int, int]
        RGB background color for the concatenated image
    separator_color : Tuple[int, int, int]
        RGB color for separator lines
    """
    if output_dir is None:
        output_dir = export_dir
    
    print(f"{ANSI['C']}Starting image concatenation...{ANSI['W']}")
    print(f"Export directory: {export_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all pipeline_batch images and sort them numerically
    image_files = get_pipeline_images(export_dir)
    
    if not image_files:
        print(f"{ANSI['R']}No pipeline_batch images found in {export_dir}{ANSI['W']}")
        return
    
    print(f"{ANSI['G']}Found {len(image_files)} images to process{ANSI['W']}")
    print(f"{ANSI['B']}Creating batches of {num_images_per_batch} images each{ANSI['W']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images in batches
    batch_num = 1
    successful_batches = 0
    
    for i in range(0, len(image_files), num_images_per_batch):
        batch_files = image_files[i:i + num_images_per_batch]
        
        # Process incomplete batches at the end
        if len(batch_files) < num_images_per_batch:
            print(f"{ANSI['Y']}Processing incomplete batch with {len(batch_files)} images{ANSI['W']}")
        
        try:
            concatenated_image = concatenate_images_horizontally(
                batch_files, 
                separator_width=separator_width,
                background_color=background_color,
                separator_color=separator_color
            )
            
            # Save the concatenated image
            output_filename = f"Concat_{batch_num:03d}.png"
            output_path = os.path.join(output_dir, output_filename)
            concatenated_image.save(output_path)
            
            print(f"{ANSI['G']}Created {output_filename} from images {i+1} to {i+len(batch_files)}{ANSI['W']}")
            batch_num += 1
            successful_batches += 1
            
        except Exception as e:
            print(f"{ANSI['R']}Error processing batch {batch_num}: {str(e)}{ANSI['W']}")
            batch_num += 1
            continue
    
    print(f"\n{ANSI['G']}Concatenation complete!{ANSI['W']}")
    print(f"Successfully created {successful_batches} concatenated images")
    print(f"Output saved to: {output_dir}")


def get_pipeline_images(export_dir: str) -> List[str]:
    """
    Get all pipeline_batch images from the export directory, sorted numerically.
    
    Parameters:
    -----------
    export_dir : str
        Directory to search for pipeline_batch images
    
    Returns:
    --------
    List[str]
        List of full paths to pipeline_batch images, sorted numerically
    """
    if not os.path.exists(export_dir):
        print(f"{ANSI['R']}Export directory does not exist: {export_dir}{ANSI['W']}")
        return []
    
    # Pattern to match pipeline_batch_XXX.png files
    pattern = re.compile(r'^pipeline_batch_(\d+)\.png$')
    
    image_files = []
    for filename in os.listdir(export_dir):
        match = pattern.match(filename)
        if match:
            image_files.append((int(match.group(1)), os.path.join(export_dir, filename)))
    
    # Sort by the numeric part of the filename
    image_files.sort(key=lambda x: x[0])
    
    # Return just the file paths
    return [filepath for _, filepath in image_files]


def concatenate_images_horizontally(
    image_paths: List[str],
    separator_width: int = 50,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    separator_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Concatenate multiple images horizontally with separator lines between them.
    
    Parameters:
    -----------
    image_paths : List[str]
        List of paths to images to concatenate
    separator_width : int
        Width of separator lines in pixels
    background_color : Tuple[int, int, int]
        RGB background color
    separator_color : Tuple[int, int, int]
        RGB color for separator lines
        
    Returns:
    --------
    Image.Image
        Concatenated image with vertical separators
    """
    images = []
    
    # Load all images
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path)
        images.append(img)
    
    # Get dimensions - all images should have the same height
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    # Use the maximum height and sum of widths plus separators
    max_height = max(heights)
    num_separators = len(images) - 1  # One separator between each pair of images
    total_width = sum(widths) + (num_separators * separator_width)
    
    # Create a new image with the combined dimensions
    concatenated = Image.new('RGB', (total_width, max_height), background_color)
    
    # Paste each image with separators
    x_offset = 0
    for i, img in enumerate(images):
        # If image height is less than max_height, center it vertically
        y_offset = (max_height - img.height) // 2
        concatenated.paste(img, (x_offset, y_offset))
        x_offset += img.width
        
        # Add vertical separator after each image except the last one
        if i < len(images) - 1:
            # Draw a separator line of specified width
            for x in range(separator_width):
                for y in range(max_height):
                    concatenated.putpixel((x_offset + x, y), separator_color)
            x_offset += separator_width
    
    return concatenated


def concat_with_progress(
    export_dir: str,
    num_images_per_batch: int,
    output_dir: Optional[str] = None,
    separator_width: int = 50
) -> None:
    """
    Concatenate images with progress bar for better user experience.
    
    Parameters:
    -----------
    export_dir : str
        Directory containing pipeline_batch images
    num_images_per_batch : int
        Number of images per concatenated batch
    output_dir : Optional[str]
        Output directory (defaults to export_dir)
    separator_width : int
        Width of separator lines
    """
    if output_dir is None:
        output_dir = export_dir
    
    # Get all pipeline images
    image_files = get_pipeline_images(export_dir)
    
    if not image_files:
        print(f"{ANSI['R']}No pipeline_batch images found{ANSI['W']}")
        return
    
    print(f"{ANSI['C']}Concatenating {len(image_files)} images...{ANSI['W']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of batches
    num_batches = (len(image_files) + num_images_per_batch - 1) // num_images_per_batch
    
    # Process with progress bar
    with tqdm(total=num_batches, desc="Concatenating batches") as pbar:
        batch_num = 1
        successful_batches = 0
        
        for i in range(0, len(image_files), num_images_per_batch):
            batch_files = image_files[i:i + num_images_per_batch]
            
            try:
                concatenated_image = concatenate_images_horizontally(
                    batch_files, 
                    separator_width=separator_width
                )
                
                output_filename = f"Concat_{batch_num:03d}.png"
                output_path = os.path.join(output_dir, output_filename)
                concatenated_image.save(output_path)
                
                successful_batches += 1
                
            except Exception as e:
                tqdm.write(f"{ANSI['R']}Error in batch {batch_num}: {str(e)}{ANSI['W']}")
            
            batch_num += 1
            pbar.update(1)
    
    print(f"\n{ANSI['G']}Concatenation complete! Created {successful_batches}/{num_batches} batches{ANSI['W']}")


def get_concat_summary(export_dir: str) -> dict:
    """
    Get summary information about available images for concatenation.
    
    Parameters:
    -----------
    export_dir : str
        Directory to analyze
    
    Returns:
    --------
    dict
        Summary information about available images
    """
    pipeline_images = get_pipeline_images(export_dir)
    
    if not pipeline_images:
        return {
            'total_images': 0,
            'directory_exists': os.path.exists(export_dir),
            'first_image': None,
            'last_image': None,
            'estimated_batches': {}
        }
    
    # Load first image to get dimensions
    try:
        first_img = Image.open(pipeline_images[0])
        img_width, img_height = first_img.size
        first_img.close()
    except:
        img_width, img_height = None, None
    
    # Calculate estimated batches for different batch sizes
    estimated_batches = {}
    for batch_size in [3, 4, 5, 6, 8, 10]:
        num_batches = (len(pipeline_images) + batch_size - 1) // batch_size
        estimated_batches[batch_size] = num_batches
    
    return {
        'total_images': len(pipeline_images),
        'directory_exists': True,
        'first_image': os.path.basename(pipeline_images[0]) if pipeline_images else None,
        'last_image': os.path.basename(pipeline_images[-1]) if pipeline_images else None,
        'image_dimensions': (img_width, img_height),
        'estimated_batches': estimated_batches
    }


def print_concat_summary(export_dir: str) -> None:
    """
    Print a summary of available images for concatenation.
    
    Parameters:
    -----------
    export_dir : str
        Directory to analyze
    """
    summary = get_concat_summary(export_dir)
    
    print(f"{ANSI['C']}=== Concatenation Summary ==={ANSI['W']}")
    print(f"Export directory: {export_dir}")
    print(f"Directory exists: {'✅' if summary['directory_exists'] else '❌'}")
    print(f"Available pipeline images: {summary['total_images']}")
    
    if summary['total_images'] > 0:
        print(f"First image: {summary['first_image']}")
        print(f"Last image: {summary['last_image']}")
        
        if summary['image_dimensions'][0]:
            print(f"Image dimensions: {summary['image_dimensions'][0]}×{summary['image_dimensions'][1]}")
        
        print(f"\n{ANSI['B']}Estimated batches for different sizes:{ANSI['W']}")
        for batch_size, num_batches in summary['estimated_batches'].items():
            print(f"  {batch_size} images/batch → {num_batches} output files")
    
    print(f"\n{ANSI['Y']}💡 Recommended: 6 images per batch for screen viewing{ANSI['W']}")


def clean_concat_files(output_dir: str) -> None:
    """
    Clean existing concatenated files from output directory.
    
    Parameters:
    -----------
    output_dir : str
        Directory to clean
    """
    if not os.path.exists(output_dir):
        print(f"{ANSI['Y']}Output directory doesn't exist: {output_dir}{ANSI['W']}")
        return
    
    # Find existing concat files
    concat_pattern = os.path.join(output_dir, "Concat_*.png")
    concat_files = glob.glob(concat_pattern)
    
    if not concat_files:
        print(f"{ANSI['G']}No existing concatenated files found{ANSI['W']}")
        return
    
    print(f"{ANSI['Y']}Found {len(concat_files)} existing concatenated files{ANSI['W']}")
    
    for file_path in concat_files:
        try:
            os.remove(file_path)
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"{ANSI['R']}Error removing {file_path}: {e}{ANSI['W']}")
    
    print(f"{ANSI['G']}Cleanup complete{ANSI['W']}") 