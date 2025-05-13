"""
VideoAnnotator class for processing videos and generating annotations
"""

import cv2
import pandas as pd
import subprocess
from datetime import datetime

from hand_task_annotator.core.detector import EnhancedHandTaskDetector


class VideoAnnotator:
    """
    Process videos and generate annotations of hand activities.
    
    This class handles video frame extraction, visualization, and
    CSV output generation with configurable sampling rate.
    """
    
    def __init__(self, video_path=None, sampling_rate=25, output_path=None, real_time_output=False, debug_mode=False, log_file=None, headless_mode=False):
        self.video_path = video_path
        self.detector = EnhancedHandTaskDetector(debug_mode=debug_mode)
        if log_file:
            self.detector.set_debug_mode(debug_mode, log_file)
        self.annotations = []
        self.sampling_rate = sampling_rate  # Hz
        self.output_path = output_path
        self.real_time_output = real_time_output
        self.temp_csv_path = output_path.replace('.csv', '_temp.csv') if output_path else None
        self.debug_mode = debug_mode
        self.log_file = log_file
        self.paused = False  # For pausing
        self.step_mode = False  # For step-by-step processing
        self.frame_delay = 1   # Milliseconds between frames (1 for fastest, higher for slower)
        self.headless_mode = headless_mode  # Skip visualization when true
        
        # For OCR timestamp extraction
        try:
            import pytesseract
            self.ocr_available = True
        except ImportError:
            print("Warning: pytesseract not available. Timestamp OCR disabled.")
            self.ocr_available = False
    
    def log(self, message, level="INFO"):
        """Forward log messages to detector"""
        if hasattr(self.detector, 'log'):
            self.detector.log(message, level)
        elif self.debug_mode or level == "ERROR":
            print(f"[{level}] {message}")
    
    def set_video_path(self, video_path):
        """Set the video path after initialization"""
        self.video_path = video_path
        
    def set_output_path(self, output_path):
        """Set the output path after initialization"""
        self.output_path = output_path
        self.temp_csv_path = output_path.replace('.csv', '_temp.csv') if output_path else None
        
    def process_video(self, output_path=None, gui_callback=None):
        """Process video and generate annotations at specified sampling rate with real-time output"""
        if output_path:
            self.output_path = output_path
            
        if not self.video_path:
            self.log("No video path specified", "ERROR")
            return False
            
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            self.log(f"Error: Could not open video file: {self.video_path}", "ERROR")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        self.log(f"Video FPS: {fps}", "INFO")
        self.log(f"Total frames: {frame_count}", "INFO")
        self.log(f"Duration: {duration:.2f} seconds", "INFO")
        
        # Calculate sample interval based on sampling rate
        sample_interval = int(fps / self.sampling_rate)
        if sample_interval < 1:
            sample_interval = 1
            
        self.log(f"Sampling every {sample_interval} frames to achieve {self.sampling_rate}Hz", "INFO")
        
        # Set up CSV writer for real-time output
        if self.real_time_output and self.output_path:
            import csv
            temp_csv_path = self.output_path.replace('.csv', '_temp.csv')
            self.temp_csv_path = temp_csv_path
            temp_csv_file = open(temp_csv_path, 'w', newline='')
            csv_writer = csv.writer(temp_csv_file)
            csv_writer.writerow(['unixTimestampInMs', 'readableTime', 'label', 'x', 'y', 'z'])
            last_save_time = 0
            save_interval = 5.0  # Save every 5 seconds
        
        # Read first frame to configure ROIs
        ret, frame = cap.read()
        if not ret:
            self.log("Error: Could not read first frame", "ERROR")
            return False
        
        # Process frames at sampling rate
        frame_index = 0
        batch_size = 20  # Process in smaller batches for more frequent updates
        batch_annotations = []
        
        # Define the timestamp OCR region (top-right corner usually)
        if self.ocr_available:
            timestamp_roi = {'x1': int(frame.shape[1] * 0.7), 
                           'y1': int(frame.shape[0] * 0.05),
                           'x2': int(frame.shape[1] * 0.95),
                           'y2': int(frame.shape[0] * 0.15)}
        
        # Try visualization first frame - if it fails, switch to headless mode
        try:
            if not self.headless_mode:
                cv2.namedWindow('Video Annotation', cv2.WINDOW_NORMAL)
                cv2.imshow('Video Annotation', frame)
                cv2.waitKey(1)  # Test if window works
        except Exception as e:
            self.log(f"Warning: Unable to create visualization window: {str(e)}", "WARNING")
            self.headless_mode = True
            self.log("Switching to headless mode (no visualization)", "WARNING")
        
        while cap.isOpened():
            # Handle paused state
            if self.paused and not self.step_mode and not self.headless_mode:
                # Show the current frame with "PAUSED" overlay
                paused_frame = frame.copy()
                cv2.putText(paused_frame, "PAUSED - Press 'P' to resume, 'S' for step", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Video Annotation', paused_frame)
                
                # Handle keypresses while paused
                key = cv2.waitKey(100) & 0xFF
                
                # Handle keypresses while paused
                if key == ord('p'):  # Resume
                    self.paused = False
                elif key == ord('s'):  # Single step
                    self.step_mode = True
                elif key == ord('q'):  # Quit
                    break
                elif key == ord('+') or key == ord('='):  # Speed up visualization
                    self.frame_delay = max(1, self.frame_delay - 10)
                    self.log(f"Frame delay: {self.frame_delay}ms", "INFO")
                elif key == ord('-'):  # Slow down visualization
                    self.frame_delay = min(500, self.frame_delay + 10)
                    self.log(f"Frame delay: {self.frame_delay}ms", "INFO")
                    
                continue
                
            # If in step mode, process one frame then pause again
            if self.step_mode and not self.headless_mode:
                self.step_mode = False
                self.paused = True
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process frames at the specified sampling rate
            if frame_index % sample_interval == 0:
                # Calculate timestamp in seconds
                timestamp = frame_index / fps
                
                # Try to extract timestamp from video frame
                video_timestamp = None
                if self.ocr_available:
                    video_timestamp = self._extract_timestamp_from_frame(frame, timestamp_roi)
                
                # Use video timestamp if available, otherwise calculated timestamp
                if video_timestamp:
                    self.log(f"OCR detected timestamp: {video_timestamp}", "DEBUG")
                    timestamp = video_timestamp
                
                # Detect activity
                activity = self.detector.detect_activity(frame, timestamp)
                
                # Create annotation
                annotation = {
                    'unixTimestampInMs': int(timestamp * 1000),
                    'readableTime': self._format_timestamp(timestamp),
                    'label': activity,
                    'x': 0,  # Dummy values for accelerometer data
                    'y': 0,
                    'z': 0
                }
                
                # Add to batch
                batch_annotations.append(annotation)
                
                # Update GUI if callback provided
                if gui_callback:
                    progress = (frame_index / frame_count) * 100
                    gui_callback(progress, activity, timestamp)
                
                # Draw visualization
                self._draw_visualization(frame, activity, timestamp)
                
                # Add artificial delay for slower viewing if needed
                if self.frame_delay > 1 and not self.headless_mode:
                    cv2.waitKey(self.frame_delay)
                
                # Write batch to file when it reaches batch size
                if len(batch_annotations) >= batch_size:
                    self.annotations.extend(batch_annotations)
                    
                    # If real-time output is enabled, write to CSV immediately
                    if self.real_time_output and self.output_path:
                        for ann in batch_annotations:
                            csv_writer.writerow([
                                ann['unixTimestampInMs'],
                                ann['readableTime'],
                                ann['label'],
                                ann['x'], ann['y'], ann['z']
                            ])
                        temp_csv_file.flush()  # Ensure data is written to disk
                        
                        # Periodically rename temp file to final output file
                        if timestamp - last_save_time >= save_interval:
                            temp_csv_file.close()
                            import shutil
                            shutil.copy(temp_csv_path, self.output_path)
                            temp_csv_file = open(temp_csv_path, 'a', newline='')  # Reopen in append mode
                            csv_writer = csv.writer(temp_csv_file)
                            last_save_time = timestamp
                            self.log(f"Interim results saved to {self.output_path} at {annotation['readableTime']}", "SUCCESS")
                    
                    batch_annotations = []
                
                # Show progress every second
                if frame_index % int(fps) == 0:
                    self.log(f"Processing: {frame_index}/{frame_count} frames ({timestamp:.2f}s)", "INFO")
                    self.log(f"Current activity: {activity}", "INFO")
            
            # Increment frame counter
            frame_index += 1
            
            # Check for quit
            if not self.headless_mode:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):  # Fast-forward option
                    # Skip ahead by 10 seconds
                    skip_frames = int(fps * 10)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + skip_frames)
                    frame_index += skip_frames
                    self.log(f"Fast-forwarded 10 seconds to frame {frame_index}", "INFO")
                elif key == ord('p'):  # Pause/resume
                    self.paused = not self.paused
                    if self.paused:
                        self.log("Playback paused", "INFO")
                    else:
                        self.log("Playback resumed", "INFO")
                elif key == ord('+') or key == ord('='):  # Speed up visualization
                    self.frame_delay = max(1, self.frame_delay - 10)
                    self.log(f"Frame delay: {self.frame_delay}ms", "INFO")
                elif key == ord('-'):  # Slow down visualization
                    self.frame_delay = min(500, self.frame_delay + 10)
                    self.log(f"Frame delay: {self.frame_delay}ms", "INFO")
        
        # Add any remaining annotations
        if batch_annotations:
            self.annotations.extend(batch_annotations)
            
            # Write remaining annotations if real-time output is enabled
            if self.real_time_output and self.output_path:
                for ann in batch_annotations:
                    csv_writer.writerow([
                        ann['unixTimestampInMs'],
                        ann['readableTime'],
                        ann['label'],
                        ann['x'], ann['y'], ann['z']
                    ])
                temp_csv_file.close()
                
                # Final copy of temp file to output file
                import shutil
                shutil.copy(temp_csv_path, self.output_path)
                self.log(f"Final results saved to {self.output_path}", "SUCCESS")
        
        cap.release()
        if not self.headless_mode:
            cv2.destroyAllWindows()
        
        # Save final annotations if not already saved
        if not (self.real_time_output and self.output_path):
            self._save_annotations(self.output_path)
            
        return True
    
    def _draw_visualization(self, frame, activity, timestamp):
        """Draw visualization on frame for debugging with enhanced information"""
        # Skip visualization in headless mode
        if self.headless_mode:
            return
            
        try:
            # Make a copy of the frame to avoid modifying the original
            vis_frame = frame.copy()
            
            # Draw ROIs
            self._draw_roi(vis_frame, self.detector.chip_source_roi, (0, 0, 255), "Chip Source")
            self._draw_roi(vis_frame, self.detector.chip_dest_roi, (0, 255, 0), "Chip Dest")
            self._draw_roi(vis_frame, self.detector.box_source_roi, (255, 0, 0), "Box Source")
            self._draw_roi(vis_frame, self.detector.box_dest_roi, (0, 255, 255), "Box Dest")
            
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
            
            # Draw timestamp
            time_str = self._format_timestamp(timestamp)
            cv2.putText(vis_frame, time_str, (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw hand position history if available
            if len(self.detector.hand_positions) > 1:
                for i in range(1, len(self.detector.hand_positions)):
                    pt1 = self.detector.hand_positions[i-1]
                    pt2 = self.detector.hand_positions[i]
                    cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 2)
            
            # Add state information with confidence scores
            info_text = []
            chip_conf = f"{self.detector.chip_holding_confidence:.2f}"
            box_conf = f"{self.detector.box_holding_confidence:.2f}"
            
            info_text.append(f"Holding chip: {self.detector.is_holding_chip} (conf: {chip_conf})")
            info_text.append(f"Holding box: {self.detector.is_holding_box} (conf: {box_conf})")
            info_text.append(f"Watch detected: {self.detector.watch_detected}")
            
            for i, text in enumerate(info_text):
                y_pos = 60 + i * 25
                cv2.putText(vis_frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Instructions for keyboard controls
            controls = "Controls: 'q'-quit | 'f'-fast forward | 'p'-pause | '+'/'-'-speed"
            cv2.putText(vis_frame, controls, 
                       (10, vis_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show speed indicator
            cv2.putText(vis_frame, f"Delay: {self.frame_delay}ms", 
                       (vis_frame.shape[1] - 150, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Video Annotation', vis_frame)
        except Exception as e:
            self.log(f"Warning: Visualization failed - {str(e)}", "WARNING")
            self.headless_mode = True  # Switch to headless mode if visualization fails
            self.log("Switching to headless mode (no visualization)", "WARNING")
    
    def _draw_roi(self, frame, roi, color, label):
        """Draw a region of interest on the frame"""
        if roi:
            cv2.rectangle(frame, 
                         (roi['x1'], roi['y1']),
                         (roi['x2'], roi['y2']),
                         color, 2)
            cv2.putText(frame, label, (roi['x1'], roi['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _extract_timestamp_from_frame(self, frame, roi):
        """Extract timestamp from video frame using OCR"""
        try:
            import pytesseract
            
            # Extract timestamp region
            timestamp_region = frame[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(timestamp_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Use OCR to extract text
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            
            # Try to parse timestamp (handle different formats)
            import re
            
            # Look for HH:MM:SS or HH:MM:SS.mmm format
            time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)', text)
            if time_match:
                time_str = time_match.group(1)
                
                # Convert to seconds
                parts = time_str.split(':')
                if len(parts) == 3:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    
                    # Handle seconds with possible milliseconds
                    if '.' in parts[2]:
                        sec_parts = parts[2].split('.')
                        seconds = int(sec_parts[0])
                        ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
                        # Scale milliseconds based on digits
                        if len(sec_parts[1]) == 1:
                            ms *= 100
                        elif len(sec_parts[1]) == 2:
                            ms *= 10
                    else:
                        seconds = int(parts[2])
                        ms = 0
                    
                    # Calculate total seconds
                    total_seconds = hours * 3600 + minutes * 60 + seconds + ms / 1000
                    return total_seconds
            
            return None
        except Exception as e:
            self.log(f"Error in OCR timestamp extraction: {str(e)}", "ERROR")
            return None
        
    def _format_timestamp(self, seconds):
        """Format seconds into HH:MM:SS"""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _save_annotations(self, output_path):
        """Save annotations to CSV file"""
        if not self.annotations:
            self.log("Warning: No annotations to save", "WARNING")
            return
            
        if not output_path:
            self.log("Warning: No output path specified", "WARNING")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.annotations)
        
        # Add dummy accelerometer data to match format if needed
        if 'x' not in df.columns:
            df['x'] = 0
            df['y'] = 0
            df['z'] = 0
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        self.log(f"Annotations saved to {output_path}", "SUCCESS")
        self.log(f"Total annotations: {len(df)}", "INFO")
        
        # Also save in readable format with activities grouped
        activity_summary = df.groupby('label').size().reset_index(name='count')
        self.log("\nActivity Summary:", "INFO")
        for _, row in activity_summary.iterrows():
            self.log(f"{row['label']}: {row['count']} frames ({row['count']/len(df)*100:.1f}%)", "INFO") 