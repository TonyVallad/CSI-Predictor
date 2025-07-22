# ArchiMed Images V2.0 - Functions Package

This package contains modular functions for DICOM processing, lung segmentation, and visualization used by the ArchiMed Images V2.0 notebook.

## 📁 Package Structure

```
functions/
├── __init__.py                      # Package initialization and exports
├── functions_archimed.py            # ArchiMed integration and file management
├── functions_conversion.py          # DICOM reading and format conversion
├── functions_pre_processing.py      # Image enhancement and lung segmentation
├── functions_visualisation.py       # Overlay creation and result visualization
└── README.md                        # This documentation file
```

## 📋 Module Descriptions

### `functions_conversion.py`
**DICOM Reading and Format Conversion**
- `read_dicom_file()` - Read and validate DICOM files
- `normalize_image_array()` - Normalize pixel values to different data types
- `resize_with_aspect_ratio()` - Resize images while maintaining aspect ratio
- `save_as_png()` / `save_as_nifti()` - Save images in different formats
- `convert_dicom_to_format()` - Main conversion function
- `get_output_path()` - Generate output file paths
- `validate_dicom_file()` - Quick DICOM validation

### `functions_pre_processing.py`
**Image Enhancement and Lung Segmentation**
- `initialize_segmentation_model()` - Load TorchXRayVision or fallback model
- `enhance_image_preprocessing()` - Apply histogram equalization and blur
- `segment_lungs_separate()` - Separate left/right lung detection
- `calculate_crop_bounds()` - Smart cropping with aspect ratio preservation
- `process_image_with_segmentation()` - Complete processing pipeline
- `keep_largest_component()` - Remove small segmentation artifacts

### `functions_visualisation.py`
**Overlay Creation and Result Visualization**
- `create_separate_lung_overlay()` - Color-coded lung visualization
- `save_visualization_files()` - Save masks and overlays
- `create_progress_tracker()` - Progress bar creation
- `print_processing_summary()` - Comprehensive result reporting
- `print_archimed_connection_status()` - Connection status display
- `display_sample_results()` - Sample file information
- `save_processing_log()` - Detailed JSON logging

### `functions_archimed.py`
**ArchiMed Integration and File Management**
- `initialize_archimed_connection()` - Connect to ArchiMed system
- `load_file_ids_from_csv()` - Parse CSV files for file IDs
- `manage_file_discovery()` - Complete file discovery and download
- `validate_configuration()` - Configuration parameter validation
- `print_configuration_summary()` - Display current settings
- `check_local_files()` / `download_files_from_archimed()` - File management

## 🔧 Usage

### In the notebook:
```python
# Import all functions
from functions import *

# Or import specific modules
from functions.functions_conversion import read_dicom_file
from functions.functions_pre_processing import segment_lungs_separate
```

### Standalone usage:
```python
import sys
sys.path.append('path/to/notebooks')

from functions import initialize_segmentation_model, read_dicom_file
```

## ✨ Key Features

- **Modular Design**: Each module has a specific responsibility
- **Easy Imports**: Package-level imports for convenience
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive error handling and reporting
- **Documentation**: Detailed docstrings for all functions
- **Flexibility**: Support for both PNG and NIFTI output formats

## 🔄 Version History

- **V2.0.0**: Initial modular release with separate function files
- Improved organization and maintainability
- Added format selection and enhanced error handling
- Better visualization and progress tracking

## 💡 Development Notes

- All functions are designed to be stateless and reusable
- Configuration is passed as dictionaries for flexibility
- Progress tracking uses tqdm for better user experience
- Comprehensive logging for debugging and analysis
- Fallback methods ensure compatibility across different environments 