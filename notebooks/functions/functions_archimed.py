"""
ArchiMed Images V2.0 - ArchiMed Integration Functions
Functions for connecting to ArchiMed, downloading files, and managing CSV data.
"""

import os
import pandas as pd
import glob
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm


def initialize_archimed_connection() -> Tuple[Any, bool]:
    """
    Initialize connection to ArchiMed system.
    
    Returns:
        Tuple[connector, authenticated]: ArchiMed connector and authentication status
    """
    try:
        import ArchiMedConnector.A3_Connector as A3_Conn
        a3conn = A3_Conn.A3_Connector()
        
        # Test the connection
        user_info = a3conn.getUserInfos()
        print("✅ ArchiMed connector initialized and authenticated successfully")
        print(f"👤 User: {user_info}")
        
        return a3conn, True
        
    except Exception as e:
        print(f"⚠️ ArchiMed initialization failed: {e}")
        print("📝 Note: ArchiMed may require specific configuration or credentials")
        print("🔄 Continuing with local file processing only...")
        
        return None, False


def load_file_ids_from_csv(csv_path: str, csv_separator: str = ';') -> List[str]:
    """
    Load file IDs from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        csv_separator (str): CSV separator character
    
    Returns:
        List[str]: List of file IDs
    """
    try:
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return []
        
        df = pd.read_csv(csv_path, sep=csv_separator)
        
        # Find the FileID column
        file_id_column = None
        for col in ['FileID', 'file_id', 'File_ID']:
            if col in df.columns:
                file_id_column = col
                break
        
        if file_id_column is None:
            print(f"❌ FileID column not found in CSV. Available columns: {list(df.columns)}")
            return []
        
        file_ids = df[file_id_column].dropna().unique()
        print(f"📋 Found {len(file_ids)} files in CSV")
        
        return [str(fid) for fid in file_ids]
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return []


def check_local_files(file_ids: List[str], download_path: str) -> Tuple[List[str], List[str]]:
    """
    Check which files exist locally and which need to be downloaded.
    
    Args:
        file_ids (List[str]): List of file IDs to check
        download_path (str): Directory to check for local files
    
    Returns:
        Tuple[local_files, missing_files]: Lists of local and missing file paths
    """
    local_files = []
    missing_file_ids = []
    
    for file_id in file_ids:
        dicom_path = os.path.join(download_path, f"{file_id}.dcm")
        if os.path.exists(dicom_path):
            local_files.append(dicom_path)
        else:
            missing_file_ids.append(file_id)
    
    return local_files, missing_file_ids


def download_files_from_archimed(a3conn, file_ids: List[str], download_path: str) -> Tuple[List[str], List[str]]:
    """
    Download files from ArchiMed system.
    
    Args:
        a3conn: ArchiMed connector object
        file_ids (List[str]): List of file IDs to download
        download_path (str): Directory to save downloaded files
    
    Returns:
        Tuple[downloaded_files, failed_downloads]: Lists of successful and failed downloads
    """
    if not file_ids:
        return [], []
    
    os.makedirs(download_path, exist_ok=True)
    downloaded_files = []
    failed_downloads = []
    
    print(f"🌐 Downloading {len(file_ids)} files from ArchiMed...")
    
    with tqdm(total=len(file_ids), desc="Downloading", unit="file") as pbar:
        for file_id in file_ids:
            try:
                dicom_path = os.path.join(download_path, f"{file_id}.dcm")
                
                result = a3conn.downloadFile(
                    int(file_id),
                    asStream=False,
                    destDir=download_path,
                    filename=f"{file_id}.dcm",
                    inWorklist=False
                )
                
                if result and os.path.exists(dicom_path):
                    downloaded_files.append(dicom_path)
                else:
                    failed_downloads.append(file_id)
                
            except Exception as e:
                print(f"⚠️ Failed to download {file_id}: {e}")
                failed_downloads.append(file_id)
            
            pbar.update(1)
    
    return downloaded_files, failed_downloads


def discover_local_dicom_files(download_path: str) -> List[str]:
    """
    Discover DICOM files in local directory when CSV is not available.
    
    Args:
        download_path (str): Directory to search for DICOM files
    
    Returns:
        List[str]: List of discovered DICOM file paths
    """
    print("📂 Scanning local directory for DICOM files...")
    
    local_patterns = [
        os.path.join(download_path, "*.dcm"),
        os.path.join(download_path, "*.DCM"),
        os.path.join(download_path, "**/*.dcm"),
        os.path.join(download_path, "**/*.DCM"),
    ]
    
    dicom_files = []
    for pattern in local_patterns:
        dicom_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    dicom_files = list(set(dicom_files))
    print(f"📁 Found {len(dicom_files)} local DICOM files")
    
    return dicom_files


def manage_file_discovery(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    """
    Complete file discovery and download management.
    
    Args:
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        Tuple[dicom_files, discovery_stats]: List of DICOM files and discovery statistics
    """
    csv_folder = config.get('CSV_FOLDER')
    csv_labels_file = config.get('CSV_LABELS_FILE')
    csv_separator = config.get('CSV_SEPARATOR', ';')
    download_path = config.get('DOWNLOAD_PATH')
    download_if_missing = config.get('DOWNLOAD_IF_MISSING', True)
    use_archimed = config.get('USE_ARCHIMED', True)
    
    # Initialize statistics
    stats = {
        'csv_files_found': 0,
        'local_files_found': 0,
        'downloaded_files': 0,
        'failed_downloads': 0,
        'total_files': 0
    }
    
    # Step 1: Initialize ArchiMed connection if requested
    a3conn, authenticated = None, False
    if use_archimed:
        a3conn, authenticated = initialize_archimed_connection()
    
    # Step 2: Load file IDs from CSV
    file_ids = []
    if csv_folder and csv_labels_file:
        csv_path = os.path.join(csv_folder, csv_labels_file)
        file_ids = load_file_ids_from_csv(csv_path, csv_separator)
        stats['csv_files_found'] = len(file_ids)
    
    # Step 3: Check local files and download missing ones
    dicom_files = []
    if file_ids:
        print(f"🔍 Checking {len(file_ids)} files...")
        
        # Check which files exist locally
        local_files, missing_file_ids = check_local_files(file_ids, download_path)
        stats['local_files_found'] = len(local_files)
        dicom_files.extend(local_files)
        
        print(f"✅ Found {len(local_files)} files locally")
        print(f"📥 Need to download {len(missing_file_ids)} files")
        
        # Download missing files if ArchiMed is available
        if missing_file_ids and authenticated and download_if_missing:
            downloaded_files, failed_downloads = download_files_from_archimed(
                a3conn, missing_file_ids, download_path
            )
            
            stats['downloaded_files'] = len(downloaded_files)
            stats['failed_downloads'] = len(failed_downloads)
            dicom_files.extend(downloaded_files)
            
            print(f"✅ Successfully downloaded {len(downloaded_files)}/{len(missing_file_ids)} files")
            
            if failed_downloads:
                print(f"❌ Failed to download {len(failed_downloads)} files")
        
        elif missing_file_ids and not authenticated:
            print(f"❌ Cannot download {len(missing_file_ids)} missing files - ArchiMed not authenticated")
            stats['failed_downloads'] = len(missing_file_ids)
    
    else:
        # No CSV available, scan local directory
        dicom_files = discover_local_dicom_files(download_path)
        stats['local_files_found'] = len(dicom_files)
    
    stats['total_files'] = len(dicom_files)
    print(f"🎯 Total DICOM files ready for processing: {len(dicom_files)}")
    
    return dicom_files, stats


def validate_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters for ArchiMed integration.
    
    Args:
        config (Dict[str, Any]): Configuration parameters
    
    Returns:
        Tuple[is_valid, error_messages]: Validation status and error messages
    """
    errors = []
    
    # Check required paths
    download_path = config.get('DOWNLOAD_PATH')
    if not download_path:
        errors.append("DOWNLOAD_PATH is required")
    
    images_path = config.get('IMAGES_PATH')
    if not images_path:
        errors.append("IMAGES_PATH is required")
    
    # Check CSV configuration if specified
    csv_folder = config.get('CSV_FOLDER')
    csv_labels_file = config.get('CSV_LABELS_FILE')
    
    if csv_folder and csv_labels_file:
        csv_path = os.path.join(csv_folder, csv_labels_file)
        if not os.path.exists(csv_path):
            errors.append(f"CSV file not found: {csv_path}")
    
    # Check output format
    output_format = config.get('OUTPUT_FORMAT', 'png').lower()
    if output_format not in ['png', 'nifti']:
        errors.append(f"Invalid OUTPUT_FORMAT: {output_format}. Must be 'png' or 'nifti'")
    
    # Check target size
    target_size = config.get('TARGET_SIZE')
    if target_size and (not isinstance(target_size, (list, tuple)) or len(target_size) != 2):
        errors.append("TARGET_SIZE must be a tuple/list of 2 integers (width, height)")
    
    return len(errors) == 0, errors


def print_configuration_summary(config: Dict[str, Any]):
    """
    Print a summary of the current configuration.
    
    Args:
        config (Dict[str, Any]): Configuration parameters
    """
    print("📋 Configuration Summary:")
    print("=" * 50)
    
    # Data paths
    print(f"📁 Data Paths:")
    print(f"   • CSV Folder: {config.get('CSV_FOLDER', 'Not specified')}")
    print(f"   • CSV File: {config.get('CSV_LABELS_FILE', 'Not specified')}")
    print(f"   • Download Path: {config.get('DOWNLOAD_PATH')}")
    print(f"   • Images Path: {config.get('IMAGES_PATH')}")
    
    if config.get('SAVE_MASKS', False):
        print(f"   • Masks Path: {config.get('MASKS_PATH')}")
    
    # Processing settings
    output_format = config.get('OUTPUT_FORMAT', 'png').upper()
    target_size = config.get('TARGET_SIZE', (518, 518))
    print(f"\n🔧 Processing Settings:")
    print(f"   • Output Format: {output_format}")
    print(f"   • Target Size: {target_size[0]}x{target_size[1]}")
    print(f"   • Save Masks: {config.get('SAVE_MASKS', False)}")
    print(f"   • Use ArchiMed: {config.get('USE_ARCHIMED', True)}")
    print(f"   • Download Missing: {config.get('DOWNLOAD_IF_MISSING', True)}")
    
    # Segmentation settings
    print(f"\n🫁 Segmentation Settings:")
    print(f"   • Sensitivity: {config.get('MODEL_SENSITIVITY', 0.0001)}")
    print(f"   • Crop Margin: {config.get('CROP_MARGIN', 25)}px")
    print(f"   • Histogram Equalization: {config.get('ENABLE_HISTOGRAM_EQUALIZATION', True)}")
    print(f"   • Gaussian Blur: {config.get('ENABLE_GAUSSIAN_BLUR', True)}")
    print(f"   • Multiple Thresholds: {config.get('USE_MULTIPLE_THRESHOLDS', True)}")
    print(f"   • Aggressive Morphology: {config.get('AGGRESSIVE_MORPHOLOGY', True)}")
    
    print("=" * 50) 