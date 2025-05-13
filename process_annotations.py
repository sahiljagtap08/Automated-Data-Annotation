#!/usr/bin/env python3
"""
Process Annotations - Utility for Wearable Device Data Annotation

This script integrates the video annotation process with raw accelerometer data:
1. Reads raw CSV with unix timestamps
2. Creates a working copy of the data
3. Adds a readable time column
4. Processes video for activity annotations
5. Merges annotations with accelerometer data by timestamp matching
6. Saves the annotated copy as output
"""

import os
import sys
import pandas as pd
import argparse
import subprocess
import tempfile
from datetime import datetime
import shutil

def unix_to_readable(unix_time_ms):
    """Convert unix timestamp (ms) to human-readable time"""
    try:
        unix_time_s = unix_time_ms / 1000.0  # Convert to seconds
        dt = datetime.fromtimestamp(unix_time_s)
        return dt.strftime('%H:%M:%S.%f')[:-3]  # Format as HH:MM:SS.mmm
    except Exception as e:
        print(f"Error converting timestamp {unix_time_ms}: {str(e)}")
        return None

def ensure_unix_timestamp(df, column_name="unixTimestampInMs"):
    """Ensure the dataframe has a unix timestamp column"""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found. Please enter the column name for unix timestamps:")
        timestamp_column = input("> ")
        if timestamp_column and timestamp_column in df.columns:
            df = df.rename(columns={timestamp_column: column_name})
        else:
            print(f"Error: Could not identify timestamp column. Available columns: {df.columns.tolist()}")
            sys.exit(1)
    return df

def process_raw_data(input_csv, output_csv):
    """Process raw CSV data to create a copy with readable time"""
    try:
        # Read the raw CSV file
        print(f"Reading raw data from: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Ensure we have unix timestamp column
        df = ensure_unix_timestamp(df)
        
        # Create a copy of the dataframe
        print(f"Creating working copy with readable time...")
        df_copy = df.copy()
        
        # Add readable time column
        df_copy['readableTime'] = df_copy['unixTimestampInMs'].apply(unix_to_readable)
        
        # Save the copy with readable time
        df_copy.to_csv(output_csv, index=False)
        print(f"Processed data saved to: {output_csv}")
        
        return df_copy
    except Exception as e:
        print(f"Error processing raw data: {str(e)}")
        return None

def run_video_annotation(video_path, output_annotation_path, sampling_rate=25):
    """Run the video annotation process to generate activity labels"""
    try:
        print(f"Processing video for annotations: {video_path}")
        print(f"This may take a while depending on video length...")
        
        # Build command to run the main annotation script
        cmd = [
            "python", "main.py",
            "--video", video_path,
            "--output", output_annotation_path,
            "--rate", str(sampling_rate),
            "--real-time",
            "--debug"
        ]
        
        # Run the annotation process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running video annotation: {stderr}")
            return False
        
        print(f"Video annotation completed successfully!")
        print(f"Annotations saved to: {output_annotation_path}")
        return True
    except Exception as e:
        print(f"Error running video annotation: {str(e)}")
        return False

def merge_annotations(processed_csv, annotations_csv, final_output_csv):
    """Merge processed accelerometer data with activity annotations by timestamp matching"""
    try:
        print(f"Merging accelerometer data with annotations...")
        
        # Read the processed data and annotations
        df_data = pd.read_csv(processed_csv)
        df_annotations = pd.read_csv(annotations_csv)
        
        # Ensure both dataframes have 'unixTimestampInMs' column
        if 'unixTimestampInMs' not in df_data.columns or 'unixTimestampInMs' not in df_annotations.columns:
            print("Error: Missing timestamp column in one of the files")
            return False
        
        # Create a copy of data for the final output
        df_final = df_data.copy()
        
        # Add a 'label' column initialized to 'Unknown'
        df_final['label'] = 'Unknown'
        
        # Get timestamps from both datasets
        data_timestamps = df_final['unixTimestampInMs'].values
        annotation_timestamps = df_annotations['unixTimestampInMs'].values
        annotation_labels = df_annotations['label'].values
        
        # Track statistics for reporting
        matches = 0
        total_rows = len(df_final)
        
        print(f"Matching {len(annotation_timestamps)} annotations to {total_rows} accelerometer data points...")
        
        # Efficient timestamp matching using nearest neighbor approach
        for i, ts in enumerate(data_timestamps):
            # Find closest annotation timestamp (naive approach - can be optimized for very large datasets)
            closest_idx = None
            min_diff = float('inf')
            
            for j, ann_ts in enumerate(annotation_timestamps):
                diff = abs(ts - ann_ts)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j
            
            # If closest timestamp is within 200ms, use that annotation
            if min_diff <= 200:  # 200ms threshold for matching
                df_final.at[i, 'label'] = annotation_labels[closest_idx]
                matches += 1
            
            # Print progress every 5%
            if i % max(1, total_rows // 20) == 0:
                print(f"Progress: {i/total_rows*100:.1f}% ({matches} matches so far)")
        
        # Save the final annotated data
        df_final.to_csv(final_output_csv, index=False)
        
        print(f"\nAnnotation merging complete!")
        print(f"Total data points: {total_rows}")
        print(f"Matched annotations: {matches} ({matches/total_rows*100:.2f}%)")
        print(f"Final annotated data saved to: {final_output_csv}")
        
        return True
    except Exception as e:
        print(f"Error merging annotations: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process and annotate wearable device data')
    parser.add_argument('--raw', type=str, required=True, help='Path to raw CSV with accelerometer data')
    parser.add_argument('--video', type=str, required=True, help='Path to video file for annotation')
    parser.add_argument('--output', type=str, help='Path to final output CSV file (default: <raw>_annotated.csv)')
    parser.add_argument('--rate', type=int, default=25, help='Sampling rate for video annotation (Hz)')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary files')
    args = parser.parse_args()
    
    # Set up file paths
    raw_csv = args.raw
    video_path = args.video
    
    if not args.output:
        base_name = os.path.splitext(raw_csv)[0]
        output_csv = f"{base_name}_annotated.csv"
    else:
        output_csv = args.output
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define temporary file paths
        processed_csv = os.path.join(temp_dir, "processed_data.csv")
        annotations_csv = os.path.join(temp_dir, "video_annotations.csv")
        
        # Step 1: Process raw data
        df_processed = process_raw_data(raw_csv, processed_csv)
        if df_processed is None:
            print("Error processing raw data. Exiting.")
            return
        
        # Step 2: Run video annotation
        success = run_video_annotation(video_path, annotations_csv, args.rate)
        if not success:
            print("Error running video annotation. Exiting.")
            return
        
        # Step 3: Merge annotations with processed data
        success = merge_annotations(processed_csv, annotations_csv, output_csv)
        if not success:
            print("Error merging annotations. Exiting.")
            return
        
        # Step 4: Save intermediate files if requested
        if args.keep_temp:
            base_dir = os.path.dirname(output_csv)
            processed_copy = os.path.join(base_dir, "processed_data.csv")
            annotations_copy = os.path.join(base_dir, "video_annotations.csv")
            
            shutil.copy(processed_csv, processed_copy)
            shutil.copy(annotations_csv, annotations_copy)
            
            print(f"\nTemporary files saved:")
            print(f"  - Processed data: {processed_copy}")
            print(f"  - Video annotations: {annotations_copy}")
    
    print("\nComplete annotation process finished successfully!")
    print(f"Final annotated data is available at: {output_csv}")

if __name__ == "__main__":
    main() 