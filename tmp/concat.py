import os
import re
from PIL import Image
from typing import List, Tuple

EXPORT_DIR = '/home/pyuser/data/Paradise_Exports'

def concat_images(num_images_per_batch: int, output_dir: str = None) -> None:
    """
    Concatenate images horizontally in batches from the Paradise_Exports folder.
    
    Args:
        num_images_per_batch (int): Number of images to stitch together in each batch
        output_dir (str, optional): Directory to save concatenated images. 
                                   Defaults to the same as EXPORT_DIR.
    """
    if output_dir is None:
        output_dir = EXPORT_DIR
    
    # Get all pipeline_batch images and sort them numerically
    image_files = get_pipeline_images()
    
    if not image_files:
        print(f"No pipeline_batch images found in {EXPORT_DIR}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Creating batches of {num_images_per_batch} images each")
    
    # Process images in batches
    batch_num = 1
    for i in range(0, len(image_files), num_images_per_batch):
        batch_files = image_files[i:i + num_images_per_batch]
        
        # Skip incomplete batches at the end if desired
        # Comment out the next two lines if you want to process incomplete batches
        if len(batch_files) < num_images_per_batch:
            print(f"Skipping incomplete batch with {len(batch_files)} images")
            break
        
        try:
            concatenated_image = concatenate_images_horizontally(batch_files)
            
            # Save the concatenated image
            output_filename = f"Concat_{batch_num:03d}.png"
            output_path = os.path.join(output_dir, output_filename)
            concatenated_image.save(output_path)
            
            print(f"Created {output_filename} from images {i+1} to {i+len(batch_files)}")
            batch_num += 1
            
        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            continue

def get_pipeline_images() -> List[str]:
    """
    Get all pipeline_batch images from the export directory, sorted numerically.
    
    Returns:
        List[str]: List of full paths to pipeline_batch images, sorted numerically
    """
    if not os.path.exists(EXPORT_DIR):
        print(f"Export directory does not exist: {EXPORT_DIR}")
        return []
    
    # Pattern to match pipeline_batch_XXX.png files
    pattern = re.compile(r'^pipeline_batch_(\d+)\.png$')
    
    image_files = []
    for filename in os.listdir(EXPORT_DIR):
        match = pattern.match(filename)
        if match:
            image_files.append((int(match.group(1)), os.path.join(EXPORT_DIR, filename)))
    
    # Sort by the numeric part of the filename
    image_files.sort(key=lambda x: x[0])
    
    # Return just the file paths
    return [filepath for _, filepath in image_files]

def concatenate_images_horizontally(image_paths: List[str]) -> Image.Image:
    """
    Concatenate multiple images horizontally with 1px black vertical lines between them.
    
    Args:
        image_paths (List[str]): List of paths to images to concatenate
        
    Returns:
        Image.Image: Concatenated image with vertical separators
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
    separator_width = 1
    num_separators = len(images) - 1  # One separator between each pair of images
    total_width = sum(widths) + (num_separators * separator_width)
    
    # Create a new image with the combined dimensions
    concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste each image with separators
    x_offset = 0
    for i, img in enumerate(images):
        # If image height is less than max_height, center it vertically
        y_offset = (max_height - img.height) // 2
        concatenated.paste(img, (x_offset, y_offset))
        x_offset += img.width
        
        # Add vertical separator after each image except the last one
        if i < len(images) - 1:
            # Draw a black vertical line
            for y in range(max_height):
                concatenated.putpixel((x_offset, y), (0, 0, 0))  # Black pixel
            x_offset += separator_width
    
    return concatenated

def main():
    """
    Example usage of the concat_images function.
    """
    # Example: Concatenate 5 images per batch
    num_images = int(input("Enter number of images per batch: (6 to fill screen) "))
    concat_images(num_images)

if __name__ == "__main__":
    main()
