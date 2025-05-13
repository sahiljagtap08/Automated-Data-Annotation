#!/usr/bin/env python3
"""
Hand Task Annotation System - Demo Script
This script demonstrates the main features of the annotation system.
"""

import sys
import os
import cv2
import argparse
import numpy as np
from main import VideoAnnotator, EnhancedHandTaskDetector

def demo_visualization():
    """Demonstrate visualization capabilities with sample video or webcam"""
    parser = argparse.ArgumentParser(description='Hand Task Annotation System Demo')
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to input video file (default: use webcam)')
    parser.add_argument('--output', type=str, default="demo_annotation.csv", 
                        help='Output CSV file (default: demo_annotation.csv)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode with verbose logging')
    parser.add_argument('--delay', type=int, default=30,
                        help='Frame delay in ms (higher = slower playback)')
    parser.add_argument('--rate', type=int, default=25,
                        help='Sampling rate in Hz (default: 25)')
    args = parser.parse_args()
    
    # Check for video file or use webcam
    if args.video and os.path.exists(args.video):
        video_source = args.video
        print(f"Using video file: {video_source}")
    else:
        video_source = 0  # Use webcam
        print("No valid video file specified, using webcam (press 'q' to quit)")
    
    # Create the annotator
    annotator = VideoAnnotator(
        video_path=video_source,
        sampling_rate=args.rate,
        output_path=args.output,
        real_time_output=True,
        debug_mode=args.debug,
        log_file="demo_log.txt"
    )
    
    # Set initial frame delay
    annotator.frame_delay = args.delay
    
    # Add demo message handler
    demo_messages = [
        "Welcome to the Hand Task Annotation Demo!",
        "Press 'p' to pause/resume playback",
        "Press 'f' to fast-forward 10 seconds",
        "Press '+'/'-' to adjust playback speed",
        "Press 'q' to quit",
        "The system is tracking your dominant hand",
        "Try moving your hand to different regions",
        "Make a pinch gesture to simulate grabbing objects",
        "Move between regions to simulate activities",
        "Annotations are being saved to CSV in real-time"
    ]
    
    def demo_callback(progress, activity, timestamp):
        """Callback to show demo messages"""
        # Show a different message every few seconds
        message_idx = int(timestamp) % len(demo_messages)
        print(f"\r[DEMO] {demo_messages[message_idx]}", end="")
        return True
    
    # Start processing
    try:
        print("\nStarting demo - watch the visualization window")
        print("=" * 50)
        annotator.process_video(gui_callback=demo_callback)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"\nError in demo: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDemo completed!")
    print(f"Annotations saved to {args.output}")
    print("You can view the results using:")
    print(f"  - CSV file: {args.output}")
    print(f"  - Log file: demo_log.txt")

def demo_detector_visualization():
    """Demo showing just the detector visualizations with a webcam"""
    print("Starting detector visualization demo...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create detector with debug mode
    detector = EnhancedHandTaskDetector(debug_mode=True)
    timestamp = 0.0
    
    # Process frames from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update ROIs for the first few frames
        if timestamp < 1.0:
            detector.update_dynamic_rois(frame, timestamp)
        
        # Detect activity
        activity = detector.detect_activity(frame, timestamp)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw ROIs
        for roi_name, roi, color in [
            ("Chip Source", detector.chip_source_roi, (0, 0, 255)),
            ("Chip Dest", detector.chip_dest_roi, (0, 255, 0)),
            ("Box Source", detector.box_source_roi, (255, 0, 0)),
            ("Box Dest", detector.box_dest_roi, (0, 255, 255))
        ]:
            if roi:
                cv2.rectangle(vis_frame, 
                             (roi['x1'], roi['y1']),
                             (roi['x2'], roi['y2']),
                             color, 2)
                cv2.putText(vis_frame, roi_name, 
                           (roi['x1'], roi['y1'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw current activity label with prominent background
        label_size, _ = cv2.getTextSize(activity, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(vis_frame, (5, 5), (15 + label_size[0], 35), (0, 0, 0), -1)
        
        # Use color coding for activity
        if activity == "Off Task":
            color = (0, 0, 255)  # Red for off task
        elif "Taking" in activity:
            color = (255, 165, 0)  # Orange for taking
        else:
            color = (0, 255, 0)  # Green for placing
        
        cv2.putText(vis_frame, activity, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw confidence scores
        info_text = [
            f"Holding chip: {detector.is_holding_chip} (conf: {detector.chip_holding_confidence:.2f})",
            f"Holding box: {detector.is_holding_box} (conf: {detector.box_holding_confidence:.2f})",
            f"Watch detected: {detector.watch_detected}"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 60 + i * 25
            cv2.putText(vis_frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw help text
        help_text = "Press 'q' to quit demo"
        cv2.putText(vis_frame, help_text, 
                   (10, vis_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Detector Demo', vis_frame)
        
        # Increment timestamp (simulate 25 fps)
        timestamp += 0.04
        
        # Check for quit
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detector demo completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--detector-only":
        demo_detector_visualization()
    else:
        demo_visualization() 