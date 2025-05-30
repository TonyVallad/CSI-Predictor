"""
Utility functions and classes for CSI-Predictor.
Contains helper functions for training, evaluation, and general utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track and compute running averages of metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
        
    def update(self, metric_name: str, value: float) -> None:
        """
        Update metric with new value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.metrics[metric_name].append(value)
        
    def get_average(self, metric_name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        return np.mean(self.metrics[metric_name])
    
    def get_averages(self) -> Dict[str, float]:
        """
        Get averages for all metrics.
        
        Returns:
            Dictionary of metric averages
        """
        return {name: self.get_average(name) for name in self.metrics.keys()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize average meter.
        
        Args:
            name: Name of the meter
            fmt: Format string for display
        """
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self) -> None:
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1) -> None:
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self) -> str:
        """String representation."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def calculate_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    weights = total_samples / (num_classes * class_counts)
    
    return torch.FloatTensor(weights)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_model_summary(model: torch.nn.Module, input_size: tuple) -> None:
    """
    Print model summary with parameter counts.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        total_params = count_parameters(model)
        print(f"Model has {total_params:,} trainable parameters")
        print("Install torchsummary for detailed model summary: pip install torchsummary")


def create_dirs(*paths: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Paths to create
    """
    from pathlib import Path
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def show_batch(
    data_loader: DataLoader,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Show a batch of images with their CSI zone predictions side-by-side.
    
    This utility randomly samples a batch and displays the 6 predicted zone masks
    in a single matplotlib figure for debugging and visualization purposes.
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to display
        figsize: Figure size for matplotlib
        save_path: Path to save the figure (optional)
        
    # Unit test stub:
    # def test_show_batch():
    #     # Create mock data loader
    #     mock_dataset = MockCSIDataset()
    #     loader = DataLoader(mock_dataset, batch_size=4)
    #     show_batch(loader, num_samples=4)
    #     # Verify plot was created without errors
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch
    import numpy as np
    from torchvision.utils import make_grid
    
    # Get a batch from the data loader
    data_iter = iter(data_loader)
    try:
        images, labels = next(data_iter)
    except StopIteration:
        print("Data loader is empty")
        return
    
    # Limit to num_samples
    if images.size(0) > num_samples:
        indices = torch.randperm(images.size(0))[:num_samples]
        images = images[indices]
        labels = labels[indices]
    
    batch_size = images.size(0)
    
    # CSI zone names for labeling
    zone_names = ['Right Superior', 'Left Superior', 'Right Middle', 
                  'Left Middle', 'Right Inferior', 'Left Inferior']
    
    # CSI class names
    class_names = ['Normal (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Unknown (4)']
    
    # Create figure with subplots
    fig, axes = plt.subplots(batch_size, 7, figsize=figsize)
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm_images = images * std + mean
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    for i in range(batch_size):
        # Convert image to numpy for display
        img_np = denorm_images[i].permute(1, 2, 0).numpy()
        
        # Display original image in first column
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {i+1}')
        axes[i, 0].axis('off')
        
        # Display CSI zones predictions in remaining columns
        for j, zone_name in enumerate(zone_names):
            col_idx = j + 1
            csi_score = labels[i, j].item()
            
            # Create zone visualization (simplified lung diagram)
            axes[i, col_idx].imshow(img_np)
            
            # Add zone overlay (approximate lung regions)
            h, w = img_np.shape[:2]
            zone_color = ['green', 'yellow', 'orange', 'red', 'gray'][csi_score]
            zone_alpha = 0.3
            
            # Define approximate zone regions (simplified)
            if 'Superior' in zone_name:
                y_start, y_end = 0, h//3
            elif 'Middle' in zone_name:
                y_start, y_end = h//3, 2*h//3
            else:  # Inferior
                y_start, y_end = 2*h//3, h
            
            if 'Right' in zone_name:
                x_start, x_end = w//2, w
            else:  # Left
                x_start, x_end = 0, w//2
            
            # Add colored rectangle for the zone
            rect = patches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor=zone_color, facecolor=zone_color, alpha=zone_alpha
            )
            axes[i, col_idx].add_patch(rect)
            
            # Set title with zone name and score
            axes[i, col_idx].set_title(f'{zone_name}\n{class_names[csi_score]}', fontsize=8)
            axes[i, col_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    plt.show()


def visualize_data_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the distribution of CSI scores across train/val/test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # CSI zone columns
    from .data import CSI_COLUMNS
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Unknown']
    
    for i, zone in enumerate(CSI_COLUMNS):
        # Combine data from all splits
        train_counts = train_df[zone].value_counts().sort_index()
        val_counts = val_df[zone].value_counts().sort_index()
        test_counts = test_df[zone].value_counts().sort_index()
        
        # Ensure all classes are represented
        for class_idx in range(5):
            if class_idx not in train_counts.index:
                train_counts[class_idx] = 0
            if class_idx not in val_counts.index:
                val_counts[class_idx] = 0
            if class_idx not in test_counts.index:
                test_counts[class_idx] = 0
        
        train_counts = train_counts.sort_index()
        val_counts = val_counts.sort_index()
        test_counts = test_counts.sort_index()
        
        # Create stacked bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[i].bar(x - width, train_counts, width, label='Train', alpha=0.8)
        axes[i].bar(x, val_counts, width, label='Val', alpha=0.8)
        axes[i].bar(x + width, test_counts, width, label='Test', alpha=0.8)
        
        axes[i].set_title(f'{zone.replace("_", " ").title()} Zone')
        axes[i].set_xlabel('CSI Score')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('CSI Score Distribution Across Train/Val/Test Sets', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()


def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing data patterns in the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with missing data statistics
    """
    from .data import CSI_COLUMNS, CSI_UNKNOWN_CLASS
    
    analysis = {
        'total_samples': len(df),
        'zone_missing_counts': {},
        'missing_patterns': {},
        'completely_missing_samples': 0
    }
    
    # Count missing values per zone (class 4 = unknown)
    for zone in CSI_COLUMNS:
        if zone in df.columns:
            missing_count = (df[zone] == CSI_UNKNOWN_CLASS).sum()
            analysis['zone_missing_counts'][zone] = missing_count
    
    # Analyze missing patterns
    missing_matrix = df[CSI_COLUMNS] == CSI_UNKNOWN_CLASS
    patterns = missing_matrix.apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
    analysis['missing_patterns'] = patterns.value_counts().to_dict()
    
    # Count samples with all zones missing
    analysis['completely_missing_samples'] = (missing_matrix.all(axis=1)).sum()
    
    return analysis


def create_debug_dataset(
    num_samples: int = 100,
    image_size: Tuple[int, int] = (224, 224),
    output_dir: str = "./debug_data"
) -> None:
    """
    Create a small debug dataset for testing purposes.
    
    Args:
        num_samples: Number of samples to create
        image_size: Size of synthetic images
        output_dir: Output directory for debug data
    """
    import numpy as np
    import pandas as pd
    from PIL import Image
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Creating {num_samples} synthetic images...")
    
    # Generate CSV data
    data = []
    for i in tqdm(range(num_samples), desc="Generating debug data"):
        # Create synthetic image (random lung-like pattern)
        img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        
        # Add some lung-like structure
        center_x, center_y = image_size[1] // 2, image_size[0] // 2
        for x in range(image_size[1]):
            for y in range(image_size[0]):
                # Create lung-like oval shapes
                dist_left = ((x - center_x//2)**2 + (y - center_y)**2) ** 0.5
                dist_right = ((x - 3*center_x//2)**2 + (y - center_y)**2) ** 0.5
                
                if dist_left < center_x//2 or dist_right < center_x//2:
                    img_array[y, x] = img_array[y, x] * 0.7 + 30  # Darker lung regions
        
        # Save image
        filename = f"debug_img_{i:04d}.png"
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(images_dir / filename)
        
        # Generate CSI scores (with some missing values)
        csi_scores = []
        for zone in range(6):
            if np.random.random() < 0.1:  # 10% missing
                csi_scores.append(4)  # Unknown
            else:
                csi_scores.append(np.random.randint(0, 4))  # Normal scores
        
        data.append({
            'FileID': filename,
            'right_sup': csi_scores[0],
            'left_sup': csi_scores[1], 
            'right_mid': csi_scores[2],
            'left_mid': csi_scores[3],
            'right_inf': csi_scores[4],
            'left_inf': csi_scores[5]
        })
    
    # Save CSV
    df = pd.DataFrame(data)
    csv_path = output_path / "debug_labels.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Debug dataset created:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {csv_path}")
    print(f"  Samples: {num_samples}") 