#!/usr/bin/env python3
"""
Demo utility for the hand task annotation system.

This script provides examples of how to use the annotation system
with various configurations and test scenarios.
"""

import os
import sys
import argparse
import time
from datetime import datetime

from hand_task_annotator.core.annotator import VideoAnnotator
from hand_task_annotator.gui.annotation_gui import run_gui


def setup_demo_environment():
    """Create a demo directory structure and log file"""
    # Create a demo directory if it doesn't exist
    demo_dir = os.path.expanduser("~/hand_task_annotator_demo")
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(demo_dir, f"demo_log_{timestamp}.txt")
    
    # Write header to log
    with open(log_path, "w") as f:
        f.write(f"Hand Task Annotator Demo - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
    
    return demo_dir, log_path


def run_cli_demo(video_path, output_dir, log_path):
    """Run demo in command-line mode"""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    print(f"Running CLI demo with video: {video_path}")
    print(f"Logs will be saved to: {log_path}")
    
    # Create output path
    output_path = os.path.join(
        output_dir, 
        f"annotations_{os.path.basename(video_path).split('.')[0]}.csv"
    )
    
    # Create annotator with debug mode
    annotator = VideoAnnotator(
        video_path=video_path,
        sampling_rate=25,
        output_path=output_path,
        real_time_output=True,
        debug_mode=True,
        log_file=log_path
    )
    
    # Process video
    print("Starting video processing...")
    start_time = time.time()
    result = annotator.process_video()
    elapsed_time = time.time() - start_time
    
    if result:
        print(f"Demo completed successfully in {elapsed_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
        return True
    else:
        print("Demo failed. Check logs for details.")
        return False


def run_gui_demo():
    """Run demo in GUI mode"""
    print("Starting GUI demo...")
    run_gui()
    return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hand Task Annotator Demo')
    
    parser.add_argument('--video', '-v', type=str,
                       help='Path to demo video file')
    parser.add_argument('--gui', '-g', action='store_true',
                       help='Launch demo in GUI mode')
    
    return parser.parse_args()


def main():
    """Main entry point for demo"""
    args = parse_args()
    
    # Create demo environment
    demo_dir, log_path = setup_demo_environment()
    
    # Run in GUI mode if requested or if no video provided
    if args.gui or not args.video:
        return run_gui_demo()
    
    # Run in CLI mode with specified video
    return run_cli_demo(args.video, demo_dir, log_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 