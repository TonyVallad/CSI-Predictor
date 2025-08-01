"""
CSI Dataset for CSI-Predictor.

This module contains the CSIDataset class extracted from the original src/data.py file.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

# Import nibabel for NIFTI file support
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None

from ..config import Config, cfg, ANSI
from .transforms import get_default_transforms, get_raddino_processor

# CSI zone column names (6 zones)
CSI_COLUMNS = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

class CSIDataset(Dataset):
    """
    Torch-compatible Dataset for CSI prediction.
    
    Supports both lazy loading from disk and pre-caching in memory based on configuration.
    Returns image tensors and label tensors where labels are shape (6,) with values in {0,1,2,3,4}.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_path: str,
        transform: Optional = None,
        phase: str = "train",
        load_to_memory: bool = False,
        use_official_processor: bool = False,
        model_arch: str = "resnet50",
        config: Optional[Config] = None
    ):
        """
        Initialize CSI dataset.
        
        Args:
            dataframe: DataFrame with FileID and CSI zone columns
            data_path: Path to image directory
            transform: Image transformations (optional)
            phase: Dataset phase for default transforms
            load_to_memory: Whether to pre-cache images in memory
            use_official_processor: Whether to use official RadDINO processor
            model_arch: Model architecture for default transforms
            config: Configuration object for normalization and file format settings
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.phase = phase
        self.load_to_memory = load_to_memory
        self.use_official_processor = use_official_processor
        self.model_arch = model_arch
        self.config = config if config is not None else cfg
        
        # Set up processor for RadDINO if needed
        self.raddino_processor = None
        if (self.model_arch.lower().replace('_', '').replace('-', '') == 'raddino' and 
            self.use_official_processor):
            self.raddino_processor = get_raddino_processor(use_official=True)
            if self.raddino_processor is None:
                print(f"{ANSI['Y']}Warning: Failed to load RadDINO processor, falling back to standard transforms{ANSI['W']}")
                self.use_official_processor = False
        
        # Set default transform if none provided
        if transform is None:
            self.transform = get_default_transforms(phase, model_arch, use_official_processor, self.config)
        else:
            self.transform = transform
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Pre-cache images if requested
        self.cached_images = {}
        if self.load_to_memory:
            self._cache_images()
        
        print(f"{ANSI['B']}Initialized{ANSI['W']} {phase} dataset with {len(self.dataframe)} samples")
        if self.use_official_processor and self.raddino_processor:
            print(f"{ANSI['B']}Using official RadDINO AutoImageProcessor{ANSI['W']}")
        if self.load_to_memory:
            print(f"{ANSI['B']}Pre-cached{ANSI['W']} {len(self.cached_images)} images in memory")
    
    def _load_nifti_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a NIFTI image with coordinate corrections.
        
        Args:
            image_path: Path to the NIFTI file
            
        Returns:
            Preprocessed image array as float32, or None if loading failed
        """
        if not NIBABEL_AVAILABLE:
            raise RuntimeError("nibabel is required for NIFTI support. Install with: pip install nibabel")
        
        try:
            # Load NIFTI file
            nifti_img = nib.load(str(image_path))
            img_data = nifti_img.get_fdata().astype(np.float32)
            
            # Handle potential 3D NIFTI with single slice (squeeze any singleton dimensions)
            img_data = np.squeeze(img_data)
            
            # Ensure we have 2D data
            if len(img_data.shape) > 2:
                print(f"Warning: Unexpected NIFTI shape {img_data.shape} for {image_path}")
                return None
            
            # Apply coordinate corrections (same as functions_image_exploration.py)
            # Fix NIFTI orientation (transpose to correct counterclockwise rotation, then flip horizontally)
            img_data = np.transpose(img_data)
            img_data = np.fliplr(img_data)  # Flip left-right to correct horizontal mirroring
            
            # Normalize to 0-1 range (NIFTI files already have 99th percentile clipping applied)
            if img_data.max() > img_data.min():
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            else:
                img_data = np.zeros_like(img_data, dtype=np.float32)
            
            return img_data
            
        except Exception as e:
            print(f"{ANSI['R']}Error loading NIFTI file {image_path}:{ANSI['W']} {e}")
            return None
    
    def _cache_images(self) -> None:
        """Pre-cache all images in memory."""
        print(f"{ANSI['B']}Pre-caching{ANSI['W']} {len(self.dataframe)} images in memory...")
        
        failed_files = []
        
        for idx in tqdm(range(len(self.dataframe)), desc="Caching images"):
            file_id = self.dataframe.iloc[idx]['FileID']
            # Use helper function to get proper filename
            image_filename = self._get_image_filename(file_id)
            image_path = self.data_path / image_filename
            
            try:
                # Check if file exists first
                if not image_path.exists():
                    print(f"{ANSI['Y']}Warning: NIFTI file not found:{ANSI['W']} {image_path}")
                    failed_files.append(f"{file_id} (file not found)")
                    self.cached_images[idx] = None
                    continue
                
                # Load NIFTI file as float32 array
                image_array = self._load_nifti_image(image_path)
                if image_array is not None:
                    # Store the raw float32 array (coordinate-corrected and normalized to 0-1)
                    self.cached_images[idx] = image_array
                else:
                    # Store None for failed images
                    print(f"{ANSI['Y']}Warning: Failed to load NIFTI data from{ANSI['W']} {image_path}")
                    failed_files.append(f"{file_id} (load failed)")
                    self.cached_images[idx] = None
            except Exception as e:
                print(f"{ANSI['Y']}Warning: Failed to cache image {image_path}:{ANSI['W']} {e}")
                failed_files.append(f"{file_id} (exception: {e})")
                # Store None for failed images
                self.cached_images[idx] = None
        
        # Report summary
        successful_count = len(self.dataframe) - len(failed_files)
        print(f"{ANSI['G']}Caching complete:{ANSI['W']} {successful_count}/{len(self.dataframe)} images loaded successfully")
        
        if failed_files:
            print(f"{ANSI['Y']}Failed to load {len(failed_files)} images:{ANSI['W']}")
            for failed_file in failed_files[:10]:  # Show first 10 failures
                print(f"  - {failed_file}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
            
            # Suggest solutions
            print(f"\n{ANSI['B']}Possible solutions:{ANSI['W']}")
            print("1. Check that all FileIDs in your CSV have corresponding .nii.gz files")
            print("2. Verify the data path is correct:", self.data_path)
            print("3. Ensure NIFTI files were created successfully during conversion")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataframe)
    
    def _get_image_filename(self, file_id) -> str:
        """
        Convert FileID to proper image filename, handling float to int conversion.
        
        Args:
            file_id: FileID from CSV (may be float or string)
            
        Returns:
            Proper image filename without .0 suffix
        """
        try:
            # Convert FileID to integer to remove .0 suffix
            file_id_int = int(float(file_id))
            return f"{file_id_int}{self.config.image_extension}"
        except (ValueError, TypeError):
            # If conversion fails, use original file_id
            return f"{file_id}{self.config.image_extension}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get dataset item with NIFTI loading support.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label_tensor, file_id)
            
        Note: Returns classification targets (class indices) for cross-entropy loss.
        Label tensor shape: (6,) with values in {0,1,2,3,4} representing classes.
        """
        row = self.dataframe.iloc[idx]
        file_id = str(row['FileID'])  # Ensure file_id is string for consistency
        
        # Load image (from cache or disk)
        if self.load_to_memory and idx in self.cached_images:
            image_array = self.cached_images[idx]
            if image_array is None:
                # Handle cached failed image with more specific error
                image_filename = self._get_image_filename(file_id)
                image_path = self.data_path / image_filename
                raise RuntimeError(f"Failed to load cached NIFTI image for FileID '{file_id}' (index {idx}). "
                                 f"Expected file: {image_path}. "
                                 f"Check that this NIFTI file exists and is valid.")
        else:
            # Load from disk
            image_filename = self._get_image_filename(file_id)
            image_path = self.data_path / image_filename
            try:
                image_array = self._load_nifti_image(image_path)
                if image_array is None:
                    raise RuntimeError(f"Failed to load NIFTI image for FileID '{file_id}': {image_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load image for FileID '{file_id}' from {image_path}: {e}")
        
        # Convert single-channel float32 array to PIL Image for transform compatibility
        # Scale to 0-255 range and convert to uint8
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image (grayscale)
        image = Image.fromarray(image_uint8, mode='L')
        
        # Apply preprocessing
        if (self.use_official_processor and self.raddino_processor and 
            self.model_arch.lower().replace('_', '').replace('-', '') == 'raddino'):
            # Use official RadDINO processor
            try:
                inputs = self.raddino_processor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            except Exception as e:
                print(f"Warning: RadDINO processor failed for image {idx}: {e}")
                # Fall back to standard transforms
                if self.transform:
                    image = self.transform(image)
        else:
            # Use standard transforms
            if self.transform:
                image = self.transform(image)
        
        # Extract CSI labels (6 zones) as classification targets
        labels = []
        for col in CSI_COLUMNS:
            labels.append(row[col])
        
        # Convert to classification targets (already in range 0-4)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Convert file_id to integer to match CSV format and return for zone masking support
        try:
            file_id_int = int(float(file_id))
            return image, label_tensor, str(file_id_int)
        except (ValueError, TypeError):
            # If conversion fails, return original file_id
            return image, label_tensor, file_id

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 