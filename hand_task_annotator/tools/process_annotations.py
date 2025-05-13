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

from hand_task_annotator.core.annotator import VideoAnnotator
from hand_task_annotator.core.utils import get_readable_time, find_matching_timestamp


def unix_to_readable(unix_time_ms):
    """Convert unix timestamp (ms) to human-readable time"""
    return get_readable_time(unix_time_ms)


def create_working_copy(input_file, output_dir=None):
    """Create a working copy of the input file"""
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        
    # Derive filename
    base_name = os.path.basename(input_file)
    name_parts = os.path.splitext(base_name)
    working_copy = os.path.join(output_dir, f"{name_parts[0]}_working{name_parts[1]}")
    
    # Copy file
    shutil.copy(input_file, working_copy)
    print(f"Created working copy: {working_copy}")
    return working_copy


def add_readable_time(csv_file, timestamp_column="unixTimestampInMs"):
    """Add a human-readable time column to the CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Check if timestamp column exists
        if timestamp_column not in df.columns:
            print(f"Error: Timestamp column '{timestamp_column}' not found in {csv_file}")
            print(f"Available columns: {', '.join(df.columns)}")
            return False
            
        # Add readable time column
        df['readableTime'] = df[timestamp_column].apply(unix_to_readable)
        
        # Save updated file
        df.to_csv(csv_file, index=False)
        print(f"Added readable time column to {csv_file}")
        return True
    except Exception as e:
        print(f"Error adding readable time: {str(e)}")
        return False


def process_video_for_annotations(video_file, output_csv, sampling_rate=25, debug=False):
    """Process video to generate activity annotations"""
    try:
        # Create annotator
        annotator = VideoAnnotator(
            video_path=video_file,
            sampling_rate=sampling_rate,
            output_path=output_csv,
            real_time_output=True,
            debug_mode=debug
        )
        
        # Process video
        print(f"Processing video: {video_file}")
        result = annotator.process_video()
        
        if result:
            print(f"Video annotations saved to {output_csv}")
            return True
        else:
            print("Video annotation failed")
            return False
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False


def merge_annotations(accel_file, annotation_file, output_file):
    """Merge accelerometer data with activity annotations based on timestamps"""
    try:
        # Read files
        df_accel = pd.read_csv(accel_file)
        df_annotations = pd.read_csv(annotation_file)
        
        # Create output DataFrame (copy of accelerometer data)
        df_final = df_accel.copy()
        
        # Add label column if not exists
        if 'label' not in df_final.columns:
            df_final['label'] = "Unknown"
            
        # Get timestamp and label arrays for faster lookup
        annotation_timestamps = df_annotations['unixTimestampInMs'].values
        annotation_labels = df_annotations['label'].values
        
        # Match each accelerometer timestamp with nearest annotation
        matches = 0
        for i, row in df_final.iterrows():
            accel_ts = row['unixTimestampInMs']
            
            # Find closest timestamp in annotations
            closest_idx = None
            min_diff = float('inf')
            
            for j, ts in enumerate(annotation_timestamps):
                diff = abs(ts - accel_ts)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j
            
            # If closest timestamp is within 200ms, use that annotation
            if closest_idx is not None and min_diff <= 200:  # 200ms threshold
                df_final.at[i, 'label'] = annotation_labels[closest_idx]
                matches += 1
        
        # Save merged file
        df_final.to_csv(output_file, index=False)
        
        # Report stats
        total_rows = len(df_final)
        match_percent = (matches / total_rows) * 100 if total_rows > 0 else 0
        print(f"Merged annotations with accelerometer data: {matches}/{total_rows} rows matched ({match_percent:.1f}%)")
        print(f"Final annotated data saved to {output_file}")
        
        # Activity distribution
        activity_counts = df_final['label'].value_counts()
        print("\nActivity Distribution:")
        for activity, count in activity_counts.items():
            percent = (count / total_rows) * 100
            print(f"  {activity}: {count} ({percent:.1f}%)")
            
        return True
    except Exception as e:
        print(f"Error merging annotations: {str(e)}")
        return False


def process_annotations(raw_file, video_file, output_file=None, sampling_rate=25, 
                      keep_temp=False, debug=False):
    """Main processing function that performs the complete workflow"""
    try:
        # Create temp directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
        
        # Determine output filename if not provided
        if output_file is None:
            base_name = os.path.basename(raw_file)
            name_parts = os.path.splitext(base_name)
            output_file = f"{name_parts[0]}_annotated{name_parts[1]}"
            
        # Step 1: Create working copy
        working_copy = create_working_copy(raw_file, temp_dir)
        
        # Step 2: Add readable time column
        if not add_readable_time(working_copy):
            print("Failed to add readable time, aborting")
            return False
            
        # Step 3: Process video for annotations
        annotations_file = os.path.join(temp_dir, "video_annotations.csv")
        if not process_video_for_annotations(video_file, annotations_file, sampling_rate, debug):
            print("Failed to process video, aborting")
            return False
            
        # Step 4: Merge annotations with accelerometer data
        if not merge_annotations(working_copy, annotations_file, output_file):
            print("Failed to merge annotations, aborting")
            return False
            
        # Clean up temp directory unless keep_temp is True
        if not keep_temp:
            try:
                shutil.rmtree(temp_dir)
                print(f"Temporary directory removed: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temp directory: {str(e)}")
        else:
            print(f"Keeping temporary directory: {temp_dir}")
            
        print(f"Processing complete! Final output saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error in processing workflow: {str(e)}")
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process wearable device annotations')
    
    parser.add_argument('--raw', '-r', type=str, required=True,
                      help='Path to raw CSV with accelerometer data')
    parser.add_argument('--video', '-v', type=str, required=True,
                      help='Path to video file for annotation')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Path to final output CSV file')
    parser.add_argument('--rate', type=int, default=25,
                      help='Sampling rate for video annotation in Hz (default: 25)')
    parser.add_argument('--keep-temp', action='store_true',
                      help='Keep temporary processing files')
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Enable debug mode with verbose output')
                      
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate input files
    if not os.path.isfile(args.raw):
        print(f"Error: Raw data file not found: {args.raw}")
        return 1
        
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
        
    # Process annotations
    success = process_annotations(
        raw_file=args.raw,
        video_file=args.video,
        output_file=args.output,
        sampling_rate=args.rate,
        keep_temp=args.keep_temp,
        debug=args.debug
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 