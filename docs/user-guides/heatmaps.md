# Heatmap Generation Guide

This guide explains how to use the heatmap generation functionality in CSI-Predictor to visualize model attention for each CSI zone.

## Overview

The heatmap generation feature creates visualizations showing which parts of chest X-ray images the model focuses on when making predictions for each of the 6 CSI zones:

- `right_sup` - Right Superior
- `left_sup` - Left Superior  
- `right_mid` - Right Middle
- `left_mid` - Left Middle
- `right_inf` - Right Inferior
- `left_inf` - Left Inferior

## Configuration

### Basic Settings

Add these settings to your `config.ini` file:

```ini
[Heatmaps]
heatmap_enabled = true
heatmap_samples_per_epoch = 1
heatmap_generate_per_epoch = false
heatmap_color_map = custom_purple_red
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `heatmap_enabled` | bool | true | Enable/disable heatmap generation |
| `heatmap_samples_per_epoch` | int | 1 | Number of random validation images to use |
| `heatmap_generate_per_epoch` | bool | false | Generate heatmaps after each epoch |
| `heatmap_color_map` | str | custom_purple_red | Color scheme for heatmaps |

## Usage

### Automatic Generation

Heatmaps are automatically generated during training based on your configuration:

1. **End of Training**: By default, heatmaps are generated using the best model after training completes
2. **Per Epoch**: If `heatmap_generate_per_epoch = true`, heatmaps are generated after each epoch

### File Structure

Heatmaps are saved in the run directory:

```
runs/
└── 20250731_130219_resnet50_train/
    └── heatmaps/
        ├── heatmap_right_sup.png
        ├── heatmap_left_sup.png
        ├── heatmap_right_mid.png
        ├── heatmap_left_mid.png
        ├── heatmap_right_inf.png
        └── heatmap_left_inf.png
```

For per-epoch generation, files are prefixed with epoch numbers:

```
heatmaps/
├── 005_heatmap_right_sup.png
├── 005_heatmap_left_sup.png
├── 010_heatmap_right_sup.png
├── 010_heatmap_left_sup.png
└── ...
```

### Color Scheme

The default color scheme progresses from:
- **Purple** (low attention)
- **Blue** 
- **Green**
- **Yellow**
- **Red** (high attention)

## Technical Details

### Implementation

The heatmap generation uses **GradCAM** (Gradient-weighted Class Activation Mapping) to visualize model attention:

1. **Target Layer Detection**: Automatically finds the last convolutional layer
2. **Gradient Computation**: Computes gradients with respect to each zone's output
3. **Attention Mapping**: Creates attention maps showing important regions
4. **Visualization**: Overlays attention maps on original images

### Fallback Method

If GradCAM fails or `pytorch_grad_cam` is not available, a fallback method is used that creates basic attention visualizations based on prediction confidence.

### Dependencies

Required dependency (automatically installed):
```
grad-cam>=1.4.0
```

## Examples

### Basic Training with Heatmaps

```bash
# Train with default heatmap settings (generated at end)
python main.py --mode train

# Train with per-epoch heatmaps
python main.py --mode train --config config_with_per_epoch_heatmaps.ini
```

### Custom Configuration

```ini
[Heatmaps]
heatmap_enabled = true
heatmap_samples_per_epoch = 3
heatmap_generate_per_epoch = true
```

This will:
- Generate heatmaps for 3 random validation images
- Create heatmaps after each epoch
- Save files with epoch prefixes

### Testing Heatmap Generation

Test the heatmap functionality without full training:

```bash
python scripts/test_heatmaps.py
```

## Troubleshooting

### Common Issues

1. **No heatmaps generated**: Check that `heatmap_enabled = true`
2. **GradCAM errors**: Ensure `grad-cam` is installed
3. **Memory issues**: Reduce `heatmap_samples_per_epoch`
4. **Slow generation**: Disable per-epoch generation for faster training

### Error Messages

- `"GradCAM not available"`: Install `grad-cam` package
- `"Could not find suitable target layer"`: Model architecture not supported
- `"Failed to generate heatmap"`: Check model and data compatibility

### Performance Considerations

- **Per-epoch generation** significantly slows training
- **Multiple samples** increase generation time
- **Large models** may require more memory
- **GPU memory** usage increases during heatmap generation

## Advanced Usage

### Custom Color Maps

To add custom color schemes, modify `src/evaluation/visualization/heatmaps.py`:

```python
def create_custom_colormap():
    colors = [
        (0.0, 0.0, 0.0),  # Black
        (1.0, 1.0, 1.0),  # White
    ]
    return plt.cm.LinearSegmentedColormap.from_list('custom', colors, N=256)
```

### Integration with Custom Models

For custom model architectures, ensure the model has:
- A `backbone` attribute with convolutional layers
- Output shape `[batch_size, num_zones, num_classes]`
- Proper gradient flow for GradCAM

## Best Practices

1. **Use sparingly**: Per-epoch generation slows training significantly
2. **Monitor memory**: Large batch sizes may cause OOM errors
3. **Validate results**: Check that heatmaps make anatomical sense
4. **Save selectively**: Only generate heatmaps for important epochs
5. **Test first**: Use test script before full training runs
