#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download ArchiMed Images

This script downloads medical images from the ArchiMed system based on FileIDs in a CSV file,
then optionally converts the DICOM files to PNG format.
"""

import os
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm
import ArchiMedConnector.A3_Connector as A3_Conn

# ANSI escape codes for colored output
ANSI = {
    'R': '\033[91m',  # Red
    'G': '\033[92m',  # Green
    'B': '\033[94m',  # Blue
    'Y': '\033[93m',  # Yellow
    'W': '\033[0m',  # White
}


class ArchiMedDownloader:
    def __init__(self, 
                 csv_folder="../../data/Paradise_CSV/",
                 csv_labels_file="Labeled_Data_RAW_Sample.csv",
                 csv_separator=";",
                 import_columns=None,
                 chunk_size=50000,
                 download_path='../../data/Paradise_DICOMs',
                 images_path='../../data/Paradise_Images',
                 export_metadata=True,
                 convert=True,
                 batch_size=10,
                 bit_depth=8,
                 create_subfolders=False,
                 delete_dicom=False,
                 monochrome=1,
                 resize_y=750):
        """
        Initialize the ArchiMed downloader with configuration parameters.
        
        Args:
            csv_folder (str): Folder path containing the CSV files
            csv_labels_file (str): Filename of the CSV file with labeled data
            csv_separator (str): Separator character used in the CSV file
            import_columns (list): Columns to import from CSV (if empty, import all)
            chunk_size (int): Number of rows to read at a time when importing CSV
            download_path (str): Path where to save downloaded DICOM files
            images_path (str): Path where to save converted PNG images
            export_metadata (bool): Whether to export DICOM metadata
            convert (bool): Whether to convert DICOM files to PNG
            batch_size (int): Number of files to process in each batch
            bit_depth (int): Bit depth for output images (8, 12, or 16)
            create_subfolders (bool): Create subfolders named after ExamCode for output files
            delete_dicom (bool): Delete DICOM files after conversion
            monochrome (int): Monochrome type (1 or 2) to use for converted images
            resize_y (int): Target height for resizing images
        """
        self.csv_folder = csv_folder
        self.csv_labels_file = csv_labels_file
        self.csv_separator = csv_separator
        self.import_columns = import_columns if import_columns else []
        self.chunk_size = chunk_size
        self.download_path = download_path
        self.images_path = images_path
        self.export_metadata = export_metadata
        self.convert = convert
        self.batch_size = batch_size
        self.bit_depth = bit_depth
        self.create_subfolders = create_subfolders
        self.delete_dicom = delete_dicom
        self.monochrome = monochrome
        self.resize_y = resize_y
        
        # Initialize the ArchiMed connector
        self.a3conn = A3_Conn.A3_Connector()
        
    def import_csv_to_dataframe(self, file_path=None, separator=None, columns=None, chunk_size=None):
        """
        Import CSV file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file
            separator (str): CSV separator character
            columns (list): List of columns to import (if None, import all)
            chunk_size (int): Number of rows to read at a time (if None, read all at once)
            
        Returns:
            pandas.DataFrame: The imported data
        """
        if file_path is None:
            file_path = os.path.join(self.csv_folder, self.csv_labels_file)
        
        if separator is None:
            separator = self.csv_separator
            
        if columns is None:
            columns = self.import_columns
            
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        try:
            # Determine which columns to use
            usecols = columns if columns and len(columns) > 0 else None
            
            if chunk_size:
                # Read in chunks and concatenate
                chunks = []
                for chunk in pd.read_csv(file_path, sep=separator, usecols=usecols, chunksize=chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                # Read all at once
                return pd.read_csv(file_path, sep=separator, usecols=usecols)
        except Exception as e:
            print(f"{ANSI['R']}Error importing CSV: {e}{ANSI['W']}")
            return None

    def collect_metadata(self, file_ids):
        """
        Collect metadata from DICOM file headers for one or more FileIDs.
        
        Args:
            file_ids (str or list): A single FileID or a list of FileIDs to process
        
        Returns:
            pandas.DataFrame: DataFrame containing the metadata from the DICOM headers
        """
        # Convert single FileID to list for consistent processing
        if isinstance(file_ids, str):
            file_ids = [file_ids]
        elif isinstance(file_ids, int):
            file_ids = [str(file_ids)]
        
        metadata_list = []
        
        for file_id in file_ids:
            file_id = str(file_id)
            subfolder_path = os.path.join(self.download_path, file_id)
            dicom_file_path = os.path.join(subfolder_path, f"{file_id}.dcm")
            
            try:
                if os.path.exists(dicom_file_path):
                    dicom_data = pydicom.dcmread(dicom_file_path)
                    metadata = {'FileID': file_id}
                    
                    for attr in dir(dicom_data):
                        # Skip internal/private attributes and PixelData
                        if attr.startswith('_') or attr == 'PixelData':
                            continue
                        try:
                            value = getattr(dicom_data, attr)
                            if not callable(value) and not isinstance(value, pydicom.sequence.Sequence):
                                metadata[attr] = str(value)
                        except Exception as e:
                            metadata[attr] = f"Error: {str(e)}"
                    
                    metadata_list.append(metadata)
                else:
                    print(f"{ANSI['R']}DICOM file not found: {dicom_file_path}{ANSI['W']}")
            except Exception as e:
                print(f"{ANSI['R']}Error processing {file_id}: {str(e)}{ANSI['W']}")
        
        if metadata_list:
            return pd.DataFrame(metadata_list)
        else:
            print(f"{ANSI['Y']}No metadata collected.{ANSI['W']}")
            return pd.DataFrame()

    def convert_dicom_to_png(self, import_folder=None, export_folder=None, bit_depth=None,
                            create_subfolders=None, resize_y=None, monochrome=None,
                            delete_dicom=None):
        """
        Convert all DICOM files in import_folder (including subfolders) to PNG format.
        
        Args:
            import_folder (str): Path to folder containing DICOM files to convert
            export_folder (str): Path to folder where PNG files will be saved
            bit_depth (int): Bit depth for output images (8, 12, or 16)
            create_subfolders (bool): If True, create subfolders named after ExamCode
            resize_y (int): Height to resize images to
            monochrome (int): Default monochrome type (1 or 2)
            delete_dicom (bool): If True, delete the DICOM file after conversion
        
        Returns:
            dict: Summary of conversion process with counts
        """
        # Use instance variables if parameters not provided
        import_folder = import_folder or self.download_path
        export_folder = export_folder or self.images_path
        bit_depth = bit_depth or self.bit_depth
        create_subfolders = create_subfolders if create_subfolders is not None else self.create_subfolders
        resize_y = resize_y or self.resize_y
        monochrome = monochrome or self.monochrome
        delete_dicom = delete_dicom if delete_dicom is not None else self.delete_dicom
        
        # Validate bit depth
        if bit_depth not in [8, 12, 16]:
            raise ValueError("bit_depth must be 8, 12, or 16")
        
        # Validate monochrome
        if monochrome not in [1, 2]:
            raise ValueError("monochrome must be 1 or 2")
        
        # Create export folder if it doesn't exist
        os.makedirs(export_folder, exist_ok=True)
        
        # Find all DICOM files recursively
        dicom_files = []
        for ext in ['.dcm', '.DCM']:  # Common DICOM extensions
            dicom_files.extend(glob.glob(os.path.join(import_folder, '**/*' + ext), recursive=True))
        
        # Initialize counters
        successful = 0
        failed = 0
        skipped = 0
        
        # Suppress specific pydicom warnings about character sets
        warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.charset")
        
        # Process each DICOM file
        for dicom_path in tqdm(dicom_files, desc="Converting DICOM files to PNG", total=len(dicom_files)):
            try:
                # Try to read as DICOM
                try:
                    ds = pydicom.dcmread(dicom_path)
                    pixel_array = ds.pixel_array
                except Exception as e:
                    skipped += 1
                    continue  # Skip if not a valid DICOM file
                
                # Get metadata for subfolder creation if needed
                exam_code = str(getattr(ds, 'StudyDescription', os.path.basename(os.path.dirname(dicom_path))))
                
                # Get file ID from the filename
                file_id = os.path.splitext(os.path.basename(dicom_path))[0]
                if file_id.endswith('.dcm'):
                    file_id = file_id[:-4]  # Remove .dcm if present
                
                # Check the PhotometricInterpretation from DICOM header
                dicom_monochrome = monochrome  # Default value
                
                if hasattr(ds, 'PhotometricInterpretation'):
                    if ds.PhotometricInterpretation == 'MONOCHROME1':
                        dicom_monochrome = 1
                    elif ds.PhotometricInterpretation == 'MONOCHROME2':
                        dicom_monochrome = 2
                
                # Get bit depth information from DICOM header
                bits_allocated = getattr(ds, 'BitsAllocated', 14)  # Default to 14 if not present
                bits_stored = getattr(ds, 'BitsStored', bits_allocated)  # Default to bits_allocated if not present
                high_bit = getattr(ds, 'HighBit', bits_stored - 1)  # Default to bits_stored-1 if not present
                max_pixel_value = pixel_array.max()
                
                # Calculate the maximum possible value based on bits_stored
                max_possible_value = (2 ** bits_stored) - 1
                
                # Normalize pixel values based on bit depth
                output_max_value = (2 ** bit_depth) - 1  # Maximum value for the output bit depth
                
                # Scale to the appropriate range based on the output bit depth
                if max_pixel_value > 0:
                    # Use the actual bit depth for scaling
                    pixel_array = ((pixel_array / min(max_pixel_value, max_possible_value)) * output_max_value)
                
                # Convert to appropriate data type based on bit depth
                if bit_depth <= 8:
                    pixel_array = pixel_array.astype(np.uint8)
                else:
                    pixel_array = pixel_array.astype(np.uint16)
                
                # Invert pixel values if needed to match the desired monochrome type
                # If DICOM is MONOCHROME1 and we want MONOCHROME2, or vice versa, we need to invert
                if dicom_monochrome != monochrome and dicom_monochrome in [1, 2] and monochrome in [1, 2]:
                    pixel_array = output_max_value - pixel_array
                
                # Convert to PIL Image
                img = Image.fromarray(pixel_array)
                
                # Resize if specified, maintaining aspect ratio
                if resize_y is not None:
                    original_width, original_height = img.size
                    # Calculate width to maintain aspect ratio
                    aspect_ratio = original_width / original_height
                    new_size = (int(resize_y * aspect_ratio), resize_y)
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Determine output path
                base_filename = os.path.splitext(os.path.basename(dicom_path))[0]
                if create_subfolders:
                    subfolder_path = os.path.join(export_folder, exam_code)
                    os.makedirs(subfolder_path, exist_ok=True)
                    output_path = os.path.join(subfolder_path, f"{base_filename}.png")
                else:
                    output_path = os.path.join(export_folder, f"{base_filename}.png")
                
                # Save as PNG
                img.save(output_path)
                successful += 1
                
                # Delete DICOM file and its containing folder if requested
                if delete_dicom:
                    # Delete the DICOM file
                    os.remove(dicom_path)
                    
                    # Delete the containing subfolder if it's empty
                    dicom_folder = os.path.dirname(dicom_path)
                    if dicom_folder != import_folder:  # Don't delete the main import folder
                        try:
                            # Check if folder is empty
                            if not os.listdir(dicom_folder):
                                shutil.rmtree(dicom_folder)
                        except Exception as e:
                            print(f"{ANSI['Y']}Warning: Could not delete folder {dicom_folder}: {str(e)}{ANSI['W']}")
                
            except Exception as e:
                print(f"{ANSI['R']}Error converting {dicom_path}: {str(e)}{ANSI['W']}")
                failed += 1
        
        # Summary
        summary = {
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "total": len(dicom_files)
        }
        
        return summary

    def download_archimed_files(self, dataframe=None, download_path=None, file_id_column='FileID', batch_size=None, convert=None):
        """
        Downloads files from ArchiMed based on FileIDs in the dataframe.
        
        Args:
            dataframe (pandas.DataFrame): DataFrame containing FileIDs
            download_path (str): Path where to save downloaded files
            file_id_column (str): Name of the column containing FileIDs
            batch_size (int): Number of files to process in each batch for progress reporting
            convert (bool): If True, convert downloaded DICOM files to PNG after each batch
            
        Returns:
            pandas.DataFrame: DataFrame with metadata of converted files
        """
        if dataframe is None:
            csv_path = os.path.join(self.csv_folder, self.csv_labels_file)
            print(f"{ANSI['B']}Importing labeled data from: {csv_path}{ANSI['W']}")
            dataframe = self.import_csv_to_dataframe()
            
            if dataframe is None:
                print(f"{ANSI['R']}Failed to import labeled data{ANSI['W']}")
                return pd.DataFrame()
                
        download_path = download_path or self.download_path
        batch_size = batch_size or self.batch_size
        convert = convert if convert is not None else self.convert
        
        # Create download directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        # Get user info for verification
        user_info = self.a3conn.getUserInfos()
        print(f"{ANSI['G']}ArchiMed Authentication Info")
        print(f"{ANSI['B']}Username:{ANSI['W']} {user_info.get('userInfos', {}).get('login', 'Unknown')}")
        print(f"{ANSI['B']}User level:{ANSI['W']} {user_info.get('userInfos', {}).get('level', 'Unknown')}")
        
        # Fix for native groups display
        native_groups = user_info.get('userInfos', {}).get('nativeGroups', ['None'])
        native_groups_str = ', '.join(native_groups) if native_groups else 'None'
        print(f"{ANSI['B']}Native Groups:{ANSI['W']} {native_groups_str}")
        
        print(f"{ANSI['B']}Authorized studies:{ANSI['W']} {', '.join(user_info.get('authorizedStudies', ['None']))}")
        print(f"{ANSI['B']}Authorized temporary storages:{ANSI['W']} {', '.join(user_info.get('authorizedTmpStorages', ['None']))}")
        
        # Check if the FileID column exists
        if file_id_column not in dataframe.columns:
            print(f"{ANSI['R']}Error: Column '{file_id_column}' not found in dataframe{ANSI['W']}")
            return pd.DataFrame()
        
        # Get unique FileIDs to avoid downloading duplicates
        file_ids = dataframe[file_id_column].unique()
        total_files = len(file_ids)
        
        print(f"\n{ANSI['B']}Starting download of {ANSI['W']}{total_files}{ANSI['B']} files to{ANSI['W']} {download_path}\n")
        
        failed_files = []
        batch_files = []
        all_metadata = pd.DataFrame()  # Store all metadata records
        downloaded_count = 0
        skipped_count = 0
        
        # Process files in batches to show progress
        for i, file_id in enumerate(file_ids):
            if pd.isna(file_id):
                continue
                
            try:
                # Convert to integer if needed
                file_id = int(file_id)
                
                # Define output path for this file
                file_output_path = os.path.join(download_path, f"{file_id}")
                dicom_file_path = os.path.join(file_output_path, f"{file_id}.dcm")
                
                # Check if the file already exists
                if os.path.exists(dicom_file_path):
                    print(f"{ANSI['Y']}File {ANSI['W']}{file_id}{ANSI['Y']} already exists, skipping download (Progress: {ANSI['W']}{((i+1)/total_files)*100:.1f}%{ANSI['Y']} - {ANSI['W']}{i+1}/{total_files}{ANSI['Y']}){ANSI['W']}")
                    batch_files.append(dicom_file_path)
                    skipped_count += 1
                else:
                    print(f"{ANSI['B']}Downloading file {ANSI['W']}{file_id}{ANSI['B']} (Progress: {ANSI['W']}{((i+1)/total_files)*100:.1f}%{ANSI['B']} - {ANSI['W']}{i+1}/{total_files}{ANSI['B']}) from{ANSI['W']} ArchiMed")
                    
                    # Download the file
                    result = self.a3conn.downloadFile(
                        file_id,
                        asStream=False,
                        destDir=file_output_path,
                        filename=f"{file_id}.dcm",
                        inWorklist=False
                    )
                    
                    downloaded_count += 1
                    batch_files.append(dicom_file_path)  # Store the path, not the result
                
                # Collect metadata for this file if needed
                if self.export_metadata:
                    try:
                        file_metadata = self.collect_metadata(file_id)
                        
                        # Add metadata to the collection if not already present
                        if not file_metadata.empty and (all_metadata.empty or not (all_metadata['FileID'] == file_id).any()):
                            all_metadata = pd.concat([all_metadata, file_metadata], ignore_index=True)
                    except Exception as e:
                        print(f"{ANSI['Y']}Warning: Could not collect metadata for file {file_id}: {str(e)}{ANSI['W']}")
                
                # Show progress every batch_size files
                if (i + 1) % batch_size == 0 or (i + 1) == total_files:
                    
                    # Convert batch if requested
                    if convert and batch_files:
                        try:
                            summary = self.convert_dicom_to_png()
                            print(f"{ANSI['G']}Conversion summary: {summary}{ANSI['W']}")
                        except Exception as e:
                            print(f"{ANSI['R']}Error during conversion: {str(e)}{ANSI['W']}")
                    
                    batch_files = []
                    print(f"{ANSI['Y']}Progress:{ANSI['W']} {i + 1}/{total_files} {ANSI['B']}files processed {ANSI['W']}({ANSI['B']}{((i + 1) / total_files * 100):.1f}%{ANSI['W']})\n")
                    
            except Exception as e:
                failed_files.append(file_id)
                print(f"{ANSI['R']}Error downloading file ID {file_id}: {str(e)}{ANSI['W']}")
        
        # Summary
        print(f"\n{ANSI['G']}Download complete: {downloaded_count} files downloaded successfully{ANSI['W']}")
        if skipped_count > 0:
            print(f"{ANSI['Y']}Skipped {skipped_count} files (already downloaded){ANSI['W']}")
        if failed_files:
            print(f"{ANSI['R']}Failed to download {len(failed_files)} files{ANSI['W']}")
        
        # Save metadata to CSV if requested and available
        if self.export_metadata and not all_metadata.empty:
            metadata_csv_path = os.path.join(download_path, "dicom_metadata.csv")
            all_metadata.to_csv(metadata_csv_path, index=False)
            print(f"{ANSI['G']}Metadata saved to {metadata_csv_path}{ANSI['W']}")
            
        return all_metadata

    def run(self):
        """
        Execute the full workflow:
        1. Import CSV data
        2. Download files from ArchiMed
        3. Convert DICOM files to PNG
        4. Export metadata
        """
        # Import CSV data
        csv_path = os.path.join(self.csv_folder, self.csv_labels_file)
        print(f"{ANSI['B']}Importing labeled data from: {csv_path}{ANSI['W']}")
        
        df_labeled_data = self.import_csv_to_dataframe()
        
        if df_labeled_data is not None:
            print(f"{ANSI['G']}Successfully imported {len(df_labeled_data)} rows of labeled data{ANSI['W']}")
            print(df_labeled_data.head())
            
            # Download files from ArchiMed
            metadata_df = self.download_archimed_files(dataframe=df_labeled_data)
            
            if not metadata_df.empty:
                print(f"{ANSI['G']}Downloaded files successfully to {ANSI['W']}{self.download_path}")
                print(f"{ANSI['B']}Metadata collected for {ANSI['W']}{len(metadata_df)}{ANSI['B']} images{ANSI['W']}")
            else:
                print(f"{ANSI['Y']}Files downloaded but no metadata was collected{ANSI['W']}")
        else:
            print(f"{ANSI['R']}Cannot download files: No labeled data available{ANSI['W']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and convert medical images from ArchiMed")
    
    # CSV parameters
    parser.add_argument("--csv-folder", default="../../data/Paradise_CSV/", help="Folder containing CSV files")
    parser.add_argument("--csv-file", default="Labeled_Data_RAW_Sample.csv", help="CSV file with labeled data")
    parser.add_argument("--csv-separator", default=";", help="CSV separator character")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of rows per chunk for CSV import")
    
    # Download parameters
    parser.add_argument("--download-path", default="../../data/Paradise_DICOMs", help="Path for downloaded DICOM files")
    parser.add_argument("--images-path", default="../../data/Paradise_Images", help="Path for converted PNG images")
    parser.add_argument("--no-metadata", action="store_false", dest="export_metadata", help="Skip exporting metadata")
    parser.add_argument("--no-convert", action="store_false", dest="convert", help="Skip conversion to PNG")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of files per batch")
    
    # Conversion parameters
    parser.add_argument("--bit-depth", type=int, default=8, choices=[8, 12, 16], help="Bit depth for PNG images")
    parser.add_argument("--create-subfolders", action="store_true", help="Create subfolders by ExamCode")
    parser.add_argument("--delete-dicom", action="store_true", help="Delete DICOM files after conversion")
    parser.add_argument("--monochrome", type=int, default=1, choices=[1, 2], help="Monochrome type")
    parser.add_argument("--resize-y", type=int, default=750, help="Target height for resized images")
    
    args = parser.parse_args()
    
    # Initialize and run the downloader
    downloader = ArchiMedDownloader(
        csv_folder=args.csv_folder,
        csv_labels_file=args.csv_file,
        csv_separator=args.csv_separator,
        chunk_size=args.chunk_size,
        download_path=args.download_path,
        images_path=args.images_path,
        export_metadata=args.export_metadata,
        convert=args.convert,
        batch_size=args.batch_size,
        bit_depth=args.bit_depth,
        create_subfolders=args.create_subfolders,
        delete_dicom=args.delete_dicom,
        monochrome=args.monochrome,
        resize_y=args.resize_y
    )
    
    downloader.run() 