"""
Heatmap generation for CSI-Predictor.

This module contains heatmap generation functionality using GradCAM
to visualize model attention for each CSI zone.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import random
from torch.utils.data import DataLoader
import warnings

# Try to import grad-cam, fallback to manual implementation if not available
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    warnings.warn("pytorch_grad_cam not available. Heatmap generation will use fallback method.")

from ...utils.logging import logger
from ...models.complete.csi_models import CSIModelWithZoneMasking

# CSI zone names in order
CSI_ZONE_NAMES = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

def create_custom_purple_red_colormap() -> LinearSegmentedColormap:
    """
    Create a custom colormap from purple to blue to green to yellow to red.
    
    Returns:
        Custom matplotlib colormap
    """
    colors = [
        (0.5, 0.0, 0.5),  # Purple
        (0.0, 0.0, 1.0),  # Blue
        (0.0, 1.0, 0.0),  # Green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 0.0)   # Red
    ]
    
    return LinearSegmentedColormap.from_list('custom_purple_red', colors, N=256)

def get_gradcam_targets(model: nn.Module, zone_idx: int) -> List[Any]:
    """
    Get GradCAM targets for a specific CSI zone.
    
    Args:
        model: The trained model
        zone_idx: Index of the CSI zone (0-5)
        
    Returns:
        List of GradCAM targets
    """
    if GRADCAM_AVAILABLE:
        # For pytorch_grad_cam, we need to specify which output to target
        # Since our model has 6 zone outputs, we target the specific zone
        return [ClassifierOutputTarget(zone_idx)]
    else:
        # Fallback: return zone index for manual implementation
        return [zone_idx]

def generate_gradcam_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    zone_idx: int,
    target_layer: Optional[nn.Module] = None
) -> np.ndarray:
    """
    Generate GradCAM heatmap for a specific zone.
    
    Args:
        model: The trained model
        input_tensor: Input image tensor [1, C, H, W]
        zone_idx: Index of the CSI zone (0-5)
        target_layer: Target layer for GradCAM (if None, auto-detect)
        
    Returns:
        Heatmap as numpy array [H, W]
    """
    if not GRADCAM_AVAILABLE:
        logger.warning("GradCAM not available, using fallback attention visualization")
        return generate_fallback_attention_map(model, input_tensor, zone_idx)
    
    # Auto-detect target layer if not provided
    if target_layer is None:
        target_layer = find_target_layer(model)
        if target_layer is None:
            logger.warning("Could not find suitable target layer for GradCAM, using fallback")
            return generate_fallback_attention_map(model, input_tensor, zone_idx)
    
    try:
        # Initialize GradCAM with updated parameters for newer versions
        device = input_tensor.device
        if hasattr(GradCAM, '__init__'):
            # Try newer GradCAM API
            try:
                cam = GradCAM(model=model, target_layers=[target_layer])
            except TypeError:
                # Fallback to older API
                cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')
        else:
            cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')
        
        # Get targets for the specific zone
        targets = get_gradcam_targets(model, zone_idx)
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0, :]  # Remove batch dimension
        
        # Clean up GradCAM object to prevent attribute errors
        try:
            if hasattr(cam, 'activations_and_grads'):
                cam.activations_and_grads.release()
        except:
            pass
        
        return heatmap
        
    except Exception as e:
        logger.warning(f"GradCAM failed for zone {zone_idx}: {e}. Using fallback method.")
        return generate_fallback_attention_map(model, input_tensor, zone_idx)

def find_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Find a suitable target layer for GradCAM.
    
    Args:
        model: The model to analyze
        
    Returns:
        Target layer or None if not found
    """
    # For ResNet models, use the last convolutional layer
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'layer4'):
            # ResNet architecture
            return backbone.layer4[-1]
        elif hasattr(backbone, 'features'):
            # DenseNet architecture
            return backbone.features.norm5
    
    # For other models, try to find the last conv layer
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    
    return None

def generate_fallback_attention_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    zone_idx: int
) -> np.ndarray:
    """
    Generate a fallback attention map when GradCAM is not available.
    
    Args:
        model: The trained model
        input_tensor: Input image tensor [1, C, H, W]
        zone_idx: Index of the CSI zone (0-5)
        
    Returns:
        Simple attention map as numpy array [H, W]
    """
    # This is a simplified fallback that creates a basic attention visualization
    # In practice, you might want to implement a more sophisticated method
    
    # Get model output
    with torch.no_grad():
        output = model(input_tensor)
    
    # Extract the specific zone output
    if output.dim() == 3:  # [batch, zones, classes]
        zone_output = output[0, zone_idx, :]  # [classes]
    else:
        zone_output = output[0, zone_idx]  # [classes]
    
    # Get the predicted class
    predicted_class = torch.argmax(zone_output).item()
    
    # Create a simple attention map based on the prediction confidence
    confidence = torch.softmax(zone_output, dim=0)[predicted_class].item()
    
    # Create a basic heatmap (this is just a placeholder)
    # In a real implementation, you might use feature maps or other techniques
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    heatmap = np.ones((h, w)) * confidence
    
    # Add some spatial variation based on zone position
    zone_height = h // 3
    zone_width = w // 2
    
    row = zone_idx // 2  # 0, 1, 2
    col = zone_idx % 2   # 0, 1
    
    start_row = row * zone_height
    end_row = (row + 1) * zone_height if row < 2 else h
    start_col = col * zone_width
    end_col = (col + 1) * zone_width if col < 1 else w
    
    # Create a gradient within the zone
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            # Distance from zone center
            center_i = (start_row + end_row) // 2
            center_j = (start_col + end_col) // 2
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            max_distance = np.sqrt((end_row - start_row)**2 + (end_col - start_col)**2) / 2
            
            # Normalize distance and apply to heatmap
            normalized_distance = min(distance / max_distance, 1.0)
            heatmap[i, j] = confidence * (1.0 - normalized_distance * 0.5)
    
    return heatmap

def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Overlay heatmap on the original image.
    
    Args:
        image: Original image [H, W, C] or [H, W]
        heatmap: Heatmap [H, W]
        alpha: Transparency factor
        
    Returns:
        Overlaid image [H, W, C]
    """
    # Ensure image is RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    
    # Normalize image to 0-1 range
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Create colormap
    colormap = create_custom_purple_red_colormap()
    
    # Apply colormap to heatmap
    heatmap_colored = colormap(heatmap)
    
    # Remove alpha channel if present
    if heatmap_colored.shape[2] == 4:
        heatmap_colored = heatmap_colored[:, :, :3]
    
    # Overlay heatmap on image
    overlaid = alpha * heatmap_colored + (1 - alpha) * image
    
    # Clip to valid range
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid

def save_heatmap(
    overlaid_image: np.ndarray,
    save_path: str,
    zone_name: str,
    epoch: Optional[int] = None
) -> None:
    """
    Save heatmap image to file with colorbar.
    
    Args:
        overlaid_image: Image with heatmap overlay
        save_path: Directory to save the image
        zone_name: Name of the CSI zone
        epoch: Epoch number (optional, for per-epoch heatmaps)
    """
    # Create filename
    if epoch is not None:
        filename = f"{epoch:03d}_heatmap_{zone_name}.png"
    else:
        filename = f"heatmap_{zone_name}.png"
    
    filepath = os.path.join(save_path, filename)
    
    # Create figure with single subplot and colorbar positioned to the right
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Main image
    im = ax.imshow(overlaid_image)
    ax.set_title(f"CSI Zone: {zone_name}")
    ax.axis('off')
    
    # Add colorbar positioned to the right of the image
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap for zone {zone_name} to {filepath}")

def generate_heatmaps_for_model(
    model: nn.Module,
    val_loader: DataLoader,
    save_dir: str,
    config,
    epoch: Optional[int] = None,
    num_samples: int = 1
) -> None:
    """
    Generate heatmaps for all CSI zones using the trained model.
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        save_dir: Directory to save heatmaps
        config: Configuration object
        epoch: Epoch number (optional, for per-epoch heatmaps)
        num_samples: Number of random samples to use
    """
    if not config.heatmap_enabled:
        logger.info("Heatmap generation is disabled in configuration")
        return
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get random samples from validation set
    val_dataset = val_loader.dataset
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    logger.info(f"Generating heatmaps for {len(sample_indices)} samples from validation set")
    
    for sample_idx in sample_indices:
        # Get sample (dataset returns image, labels, file_id)
        sample_data = val_dataset[sample_idx]
        if len(sample_data) == 3:
            image, labels, file_id = sample_data
        else:
            # Fallback for older dataset formats
            image, labels = sample_data
        
        # Add batch dimension
        input_tensor = image.unsqueeze(0).to(next(model.parameters()).device)
        
        # Convert to numpy for visualization
        if input_tensor.shape[1] == 1:  # Grayscale
            image_np = input_tensor[0, 0].cpu().numpy()
        else:  # RGB
            image_np = input_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        # Generate heatmaps for each zone
        for zone_idx, zone_name in enumerate(CSI_ZONE_NAMES):
            try:
                # Generate heatmap
                heatmap = generate_gradcam_heatmap(model, input_tensor, zone_idx)
                
                # Overlay on image
                overlaid = overlay_heatmap_on_image(image_np, heatmap)
                
                # Save heatmap
                save_heatmap(overlaid, save_dir, zone_name, epoch)
                
            except Exception as e:
                logger.warning(f"Failed to generate heatmap for zone {zone_name}: {e}")
                continue
    
    logger.info(f"Heatmap generation completed. Saved to: {save_dir}")

def generate_heatmaps_for_best_model(
    model: nn.Module,
    val_loader: DataLoader,
    save_dir: str,
    config
) -> None:
    """
    Generate heatmaps using the best model (typically called at end of training).
    
    Args:
        model: The best trained model
        val_loader: Validation data loader
        save_dir: Directory to save heatmaps
        config: Configuration object
    """
    logger.info("Generating heatmaps for best model")
    generate_heatmaps_for_model(
        model=model,
        val_loader=val_loader,
        save_dir=save_dir,
        config=config,
        epoch=None,  # No epoch number for final heatmaps
        num_samples=config.heatmap_samples_per_epoch
    )

def generate_heatmaps_for_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    save_dir: str,
    config,
    epoch: int
) -> None:
    """
    Generate heatmaps for current epoch (typically called during training).
    
    Args:
        model: The current epoch model
        val_loader: Validation data loader
        save_dir: Directory to save heatmaps
        config: Configuration object
        epoch: Current epoch number
    """
    logger.info(f"Generating heatmaps for epoch {epoch}")
    generate_heatmaps_for_model(
        model=model,
        val_loader=val_loader,
        save_dir=save_dir,
        config=config,
        epoch=epoch,
        num_samples=config.heatmap_samples_per_epoch
    )
