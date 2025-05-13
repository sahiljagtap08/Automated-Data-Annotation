#!/usr/bin/env python3
"""
Hand Task Annotator - Main entry point

This script serves as the main entry point for the hand task annotation system.
It processes video files to automatically detect and annotate hand activities
in task-based experiments.
"""

import sys
import argparse
import tkinter as tk

from hand_task_annotator.core.annotator import VideoAnnotator
from hand_task_annotator.gui.annotation_gui import AnnotationGUI, run_gui


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hand Task Video Annotator')
    
    # Input/output options
    parser.add_argument('--video', '-v', type=str, help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, help='Path to output CSV file')
    
    # Processing options
    parser.add_argument('--rate', '-r', type=int, default=25,
                       help='Sampling rate in Hz (default: 25)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--log-file', '-l', type=str,
                       help='Path to log file (if not specified, logs only to console)')
    parser.add_argument('--real-time', '-rt', action='store_true',
                       help='Enable real-time output to CSV file')
    
    # GUI mode
    parser.add_argument('--gui', '-g', action='store_true',
                       help='Launch in GUI mode')
                       
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Check if GUI mode is requested
    if args.gui or (len(sys.argv) == 1):  # Default to GUI if no args provided
        print("Starting in GUI mode...")
        run_gui()
        return
    
    # Command-line mode
    if not args.video:
        print("Error: Input video file is required in command-line mode")
        print("Use --gui flag to launch in GUI mode or specify --video")
        return
    
    if not args.output:
        print("Error: Output CSV file is required in command-line mode")
        print("Specify output file with --output")
        return
    
    # Create annotator instance
    annotator = VideoAnnotator(
        video_path=args.video,
        sampling_rate=args.rate,
        output_path=args.output,
        real_time_output=args.real_time,
        debug_mode=args.debug,
        log_file=args.log_file
    )
    
    # Process video
    print(f"Processing video: {args.video}")
    print(f"Output will be saved to: {args.output}")
    print(f"Sampling rate: {args.rate}Hz")
    
    result = annotator.process_video()
    
    if result:
        print("Annotation completed successfully!")
    else:
        print("Annotation failed. Check logs for details.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 