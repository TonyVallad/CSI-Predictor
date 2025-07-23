"""
NIFTI Histogram Analysis Functions
Functions for creating and analyzing histograms from NIFTI image files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from typing import List, Dict, Optional, Tuple, Any
import warnings

# Set seaborn style for better looking plots
sns.set_style("whitegrid")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def analyze_bits_used_distribution(nifti_path: str, histogram_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Analyze the distribution of bits used across all NIFTI files
    and create a summary histogram.
    
    Parameters:
    -----------
    nifti_path : str
        Path to directory containing NIFTI files
    histogram_path : str
        Path to directory where histograms will be saved
        
    Returns:
    --------
    Tuple[List[Dict], List[int]]
        File information and bits distribution list
    """
    # Create histogram directory if it doesn't exist
    Path(histogram_path).mkdir(parents=True, exist_ok=True)
    
    # Get all NIFTI files
    nifti_files = []
    for root, dirs, files in os.walk(nifti_path):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                nifti_files.append(os.path.join(root, file))
    
    print(f"Analyzing bits used distribution for {len(nifti_files)} NIFTI files...")
    
    bits_used_list = []
    file_info = []
    
    for nifti_file in tqdm(nifti_files, desc="Analyzing files"):
        try:
            # Read NIFTI file
            nii_img = nib.load(nifti_file)
            
            # Get pixel data
            pixel_array = nii_img.get_fdata()
            
            # Get max pixel value
            max_pixel = np.max(pixel_array)
            
            # Calculate bits used
            if max_pixel > 0:
                bits_used = int(np.ceil(np.log2(max_pixel + 1)))
            else:
                bits_used = 1
            
            bits_used_list.append(bits_used)
            file_info.append({
                'file': os.path.basename(nifti_file),
                'bits_used': bits_used,
                'max_pixel': max_pixel,
                'dtype': str(pixel_array.dtype),
                'shape': pixel_array.shape
            })
            
        except Exception as e:
            print(f"Error processing {nifti_file}: {str(e)}")
            continue
    
    # Create colored bar chart showing bits used distribution
    plt.figure(figsize=(10, 6))
    
    unique_bits_used = sorted(list(set(bits_used_list)))
    x_pos = np.arange(len(unique_bits_used))
    counts_used = [bits_used_list.count(bit) for bit in unique_bits_used]
    
    bars = plt.bar(x_pos, counts_used, alpha=0.7, edgecolor='black')
    
    # Color bars based on bits used (same color scheme as histograms)
    for i, (bar, bits) in enumerate(zip(bars, unique_bits_used)):
        if bits <= 8:
            bar.set_color('lightblue')
        elif bits <= 10:
            bar.set_color('blue')
        elif bits <= 12:
            bar.set_color('green')
        elif bits <= 13:
            bar.set_color('orange')
        elif bits <= 14:
            bar.set_color('red')
        elif bits <= 15:
            bar.set_color('purple')
        else:
            bar.set_color('darkred')
    
    plt.title('File Count by Bits Used (Colored by Histogram Color Scheme)', fontsize=14, fontweight='bold')
    plt.xlabel('Bits Used', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.xticks(x_pos, unique_bits_used)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(counts_used):
        plt.text(i, v + max(counts_used) * 0.01, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Save the distribution plot
    output_path = os.path.join(histogram_path, "bits_used_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== BITS USED DISTRIBUTION ANALYSIS ===")
    print(f"Total files processed: {len(bits_used_list)}")
    print(f"Bits used range: {min(bits_used_list)} - {max(bits_used_list)}")
    print(f"Most common bits used: {max(set(bits_used_list), key=bits_used_list.count)} ({bits_used_list.count(max(set(bits_used_list), key=bits_used_list.count))} files)")
    print(f"\nBreakdown by bits used:")
    for bits in unique_bits_used:
        count = bits_used_list.count(bits)
        percentage = (count / len(bits_used_list)) * 100
        print(f"  {bits} bits: {count} files ({percentage:.1f}%)")
    
    return file_info, bits_used_list


def create_nifti_histograms(nifti_path: str, histogram_path: str, bins: Optional[int] = None) -> None:
    """
    Create histograms for all NIFTI files in the nifti_path directory
    and save them to the histogram_path directory.
    
    Parameters:
    -----------
    nifti_path : str
        Path to directory containing NIFTI files
    histogram_path : str
        Path to directory where histograms will be saved
    bins : int, optional
        Number of bins to use for all histograms. If None (default), 
        bins will be automatically calculated based on data range.
        Examples: bins=256 for exactly 256 bars, bins=512 for 512 bars.
    """
    # Create histogram directory if it doesn't exist
    Path(histogram_path).mkdir(parents=True, exist_ok=True)
    
    # Get all NIFTI files
    nifti_files = []
    for root, dirs, files in os.walk(nifti_path):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                nifti_files.append(os.path.join(root, file))
    
    print(f"Found {len(nifti_files)} NIFTI files")
    
    for nifti_file in tqdm(nifti_files, desc="Creating histograms"):
        try:
            # Read NIFTI file
            nii_img = nib.load(nifti_file)
            
            # Get pixel data
            pixel_array = nii_img.get_fdata()
            
            # Get pixel value statistics
            min_pixel = np.min(pixel_array)
            max_pixel = np.max(pixel_array)
            
            # Get data type info
            data_dtype = str(pixel_array.dtype)
            
            # Calculate actual bits used (based on maximum pixel value)
            if max_pixel > 0:
                bits_used = int(np.ceil(np.log2(max_pixel + 1)))
            else:
                bits_used = 1
            
            # Determine color based on bits used
            if bits_used <= 8:
                color = 'lightblue'
            elif bits_used <= 10:
                color = 'blue'
            elif bits_used <= 12:
                color = 'green'
            elif bits_used <= 13:
                color = 'orange'
            elif bits_used <= 14:
                color = 'red'
            elif bits_used <= 15:
                color = 'purple'
            else:
                color = 'darkred'
            
            # Determine bins: use parameter if provided, otherwise auto-calculate
            if bins is None:
                # Automatic bin calculation based on data range
                data_range = max_pixel - min_pixel
                if data_range <= 256:
                    calculated_bins = min(256, int(data_range) + 1)
                elif data_range <= 4096:
                    calculated_bins = min(4096, int(data_range / 16) + 1)
                elif data_range <= 16384:
                    calculated_bins = min(16384, int(data_range / 64) + 1)
                else:
                    calculated_bins = min(65536, int(data_range / 256) + 1)
                
                # Ensure we have at least 50 bins for good visualization
                histogram_bins = max(calculated_bins, 50)
            else:
                # Use the specified number of bins
                histogram_bins = bins
            
            # Create histogram with consistent coloring using seaborn
            plt.figure(figsize=(10, 6))
            sns.histplot(pixel_array.flatten(), bins=histogram_bins, color=color, 
                        stat='count', element='bars', fill=True, alpha=0.7,
                        edgecolor='none', legend=False)
            
            plt.title(f'Pixel Value Distribution - {os.path.basename(nifti_file)}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Pixel Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            # Seaborn already provides a nice grid, but we can customize it
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add comprehensive information text box with clean formatting
            info_text = f'Data Type: {data_dtype}\nBits Used: {bits_used}\nShape: {pixel_array.shape}\nMin Pixel Value: {min_pixel:.2f}\nMax Pixel Value: {max_pixel:.2f}'
            plt.text(0.02, 0.98, info_text, 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.8),
                    fontsize=10)
            
            # Save histogram
            base_name = os.path.splitext(os.path.basename(nifti_file))[0]
            # Handle .nii.gz files
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]
            output_path = os.path.join(histogram_path, f"{base_name}_histogram.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing {nifti_file}: {str(e)}")
            continue


def create_histogram_grid(histogram_path: str, rows: int, cols: int) -> List[str]:
    """
    Create multiple grid layouts of histogram images from histogram_path.
    Creates as many grids as necessary to include all histogram images.
    
    Parameters:
    -----------
    histogram_path : str
        Path to directory containing histogram images
    rows : int
        Number of rows in each grid
    cols : int
        Number of columns in each grid
        
    Returns:
    --------
    List[str]
        List of created grid file paths
    """
    # Get all histogram files (only files ending with '_histogram.png')
    histogram_files = [f for f in os.listdir(histogram_path) 
                      if f.endswith('_histogram.png')]
    histogram_files.sort()  # Ensure consistent ordering
    
    if not histogram_files:
        print("No histogram files found in histogram_path")
        return []
    
    total_images = len(histogram_files)
    images_per_grid = rows * cols
    
    # Calculate how many grids we need
    num_full_grids = total_images // images_per_grid
    remaining_images = total_images % images_per_grid
    total_grids = num_full_grids + (1 if remaining_images > 0 else 0)
    
    print(f"Found {total_images} histogram files")
    print(f"Creating {total_grids} grids ({rows}x{cols} = {images_per_grid} images per grid)")
    
    grid_files = []  # Keep track of created grid files
    
    # Create each grid
    for grid_num in range(total_grids):
        start_idx = grid_num * images_per_grid
        end_idx = min(start_idx + images_per_grid, total_images)
        images_in_this_grid = end_idx - start_idx
        
        print(f"\nCreating grid {grid_num + 1}/{total_grids} with {images_in_this_grid} images...")
        
        # Create figure with appropriate size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        # Handle single row or column case
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Fill the grid with progress bar
        for cell_index in tqdm(range(rows * cols), desc=f"Grid {grid_num + 1}"):
            i = cell_index // cols  # Row index
            j = cell_index % cols   # Column index
            image_index = start_idx + cell_index  # Index in the overall file list
            
            if image_index < end_idx:
                # Load and display image
                img_path = os.path.join(histogram_path, histogram_files[image_index])
                img = plt.imread(img_path)
                axes[i, j].imshow(img)
                # Use shorter title to fit better
                title = os.path.splitext(histogram_files[image_index])[0].replace('_histogram', '')
                axes[i, j].set_title(title, fontsize=8, pad=5)
                axes[i, j].axis('off')
            else:
                # Hide empty subplots
                axes[i, j].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the grid with grid number
        if total_grids == 1:
            output_path = os.path.join(histogram_path, f"histogram_grid_{rows}x{cols}.png")
        else:
            output_path = os.path.join(histogram_path, f"histogram_grid_{rows}x{cols}_part{grid_num + 1}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        grid_files.append(output_path)
        print(f"Grid {grid_num + 1} saved as: {os.path.basename(output_path)}")
    
    print(f"\n=== GRID CREATION COMPLETE ===")
    print(f"Total images processed: {total_images}")
    print(f"Total grids created: {total_grids}")
    print(f"Grid files created:")
    for grid_file in grid_files:
        print(f"  - {os.path.basename(grid_file)}")
    
    return grid_files


def create_global_nifti_histogram(nifti_path: str, histogram_path: str, bins: Optional[int] = None) -> None:
    """
    Create a single global histogram combining pixel values from all NIFTI files.
    
    Parameters:
    -----------
    nifti_path : str
        Path to directory containing NIFTI files
    histogram_path : str
        Path to directory where histogram will be saved
    bins : int, optional
        Number of bins to use for the histogram. If None, auto-calculated.
    """
    # Create histogram directory if it doesn't exist
    Path(histogram_path).mkdir(parents=True, exist_ok=True)
    
    # Get all NIFTI files
    nifti_files = []
    for root, dirs, files in os.walk(nifti_path):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                nifti_files.append(os.path.join(root, file))
    
    print(f"Creating global histogram from {len(nifti_files)} NIFTI files...")
    
    all_pixel_values = []
    
    for nifti_file in tqdm(nifti_files, desc="Loading NIFTI files"):
        try:
            # Read NIFTI file
            nii_img = nib.load(nifti_file)
            pixel_array = nii_img.get_fdata()
            
            # Append flattened pixel values
            all_pixel_values.extend(pixel_array.flatten())
            
        except Exception as e:
            print(f"Error processing {nifti_file}: {str(e)}")
            continue
    
    # Convert to numpy array for analysis
    all_pixel_values = np.array(all_pixel_values)
    
    # Calculate statistics
    min_val = np.min(all_pixel_values)
    max_val = np.max(all_pixel_values)
    mean_val = np.mean(all_pixel_values)
    std_val = np.std(all_pixel_values)
    
    # Determine bins
    if bins is None:
        data_range = max_val - min_val
        if data_range <= 256:
            histogram_bins = 256
        elif data_range <= 4096:
            histogram_bins = 1024
        else:
            histogram_bins = 2048
    else:
        histogram_bins = bins
    
    # Create global histogram
    plt.figure(figsize=(12, 8))
    
    sns.histplot(all_pixel_values, bins=histogram_bins, color='darkblue', 
                stat='count', element='bars', fill=True, alpha=0.7,
                edgecolor='none', legend=False)
    
    plt.title('Global Pixel Value Distribution - All NIFTI Files Combined', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Pixel Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add comprehensive statistics text box
    info_text = f'Total Files: {len(nifti_files)}\nTotal Pixels: {len(all_pixel_values):,}\nMin Value: {min_val:.2f}\nMax Value: {max_val:.2f}\nMean: {mean_val:.2f}\nStd Dev: {std_val:.2f}'
    plt.text(0.02, 0.98, info_text, 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.8),
            fontsize=12)
    
    # Save global histogram
    output_path = os.path.join(histogram_path, "global_nifti_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\n=== GLOBAL HISTOGRAM COMPLETE ===")
    print(f"Global histogram saved as: {os.path.basename(output_path)}")
    print(f"Statistics:")
    print(f"  Total files: {len(nifti_files)}")
    print(f"  Total pixels: {len(all_pixel_values):,}")
    print(f"  Value range: {min_val:.2f} - {max_val:.2f}")
    print(f"  Mean ± Std: {mean_val:.2f} ± {std_val:.2f}") 