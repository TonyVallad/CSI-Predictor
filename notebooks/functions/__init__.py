"""
ArchiMed Images V2.0 - Functions Package
Modular functions for DICOM processing, lung segmentation, and visualization.
"""

# Import all functions from each module for convenient access
from .functions_conversion import *
from .functions_pre_processing import *
from .functions_visualisation import *
from .functions_archimed import *

__version__ = "2.0.0"
__author__ = "ArchiMed Images Team"

# Package information
__all__ = [
    # Conversion functions
    'read_dicom_file',
    'normalize_image_array', 
    'resize_with_aspect_ratio',
    'save_as_png',
    'save_as_nifti',
    'convert_dicom_to_format',
    'get_output_path',
    'validate_dicom_file',
    'apply_geometric_transforms_preserve_values',
    'test_nifti_conversion',
    
    # Pre-processing functions
    'initialize_segmentation_model',
    'enhance_image_preprocessing',
    'keep_largest_component',
    'segment_lungs_separate',
    'calculate_crop_bounds',
    'process_image_with_segmentation',
    'apply_segmentation_transforms_to_original',
    
    # Visualization functions
    'create_separate_lung_overlay',
    'save_visualization_files',
    'create_progress_tracker',
    'print_processing_summary',
    'print_archimed_connection_status',
    'print_file_discovery_summary',
    'display_sample_results',
    'save_processing_log',
    
    # ArchiMed functions
    'initialize_archimed_connection',
    'load_file_ids_from_csv',
    'manage_file_discovery',
    'validate_configuration',
    'print_configuration_summary',
] 