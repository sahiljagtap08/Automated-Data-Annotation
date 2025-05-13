import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import argparse
import sys

class EnhancedHandTaskDetector:
    def __init__(self, debug_mode=False):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track up to 2 hands to identify the watch-wearing hand
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Drawing
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Debug mode for verbose logging
        self.debug_mode = debug_mode
        
        # Define regions of interest (ROI) - will be dynamically updated
        self.chip_source_roi = None  # Red box area with chips
        self.chip_dest_roi = None    # Area where chips are placed
        self.box_source_roi = None   # Where boxes are taken from
        self.box_dest_roi = None     # Where boxes are placed
        
        # Track if ROIs need updating
        self.last_roi_update_time = 0
        self.roi_update_interval = 2.0  # Update ROIs every 2 seconds
        
        # Track motion history
        self.hand_positions = deque(maxlen=60)
        self.state_memory = deque(maxlen=15)
        self.action_start_time = None
        self.current_action = "Off Task"
        self.last_position = None
        self.dominant_hand_idx = None  # Index of the watch-wearing dominant hand
        
        # Activity detection thresholds
        self.grabbing_threshold = 0.2
        self.releasing_threshold = 0.3
        self.movement_threshold = 30
        
        # State tracking for complex actions
        self.is_holding_chip = False
        self.is_holding_box = False
        self.time_in_state = 0
        self.time_in_current_activity = 0
        self.time_without_activity = 0
        self.consecutive_off_task_frames = 0  # Count consecutive off-task frames
        
        # Add confidence scores for better accuracy and debugging
        self.chip_holding_confidence = 0.0  # 0.0-1.0 confidence score 
        self.box_holding_confidence = 0.0   # 0.0-1.0 confidence score
        self.frames_with_chip = 0          # Count frames showing chip
        self.frames_without_chip = 0       # Count frames not showing chip
        self.chip_memory_frames = 10       # Frames to remember
        
        # Color detection parameters with increased specificity
        self.chip_colors = {
            'red': ([0, 120, 120], [10, 255, 255]),  # More saturated red
            'green': ([45, 100, 100], [75, 255, 255]),  # More specific green
            'blue': ([100, 120, 100], [130, 255, 255])  # More specific blue
        }
        
        self.box_colors = {
            'red': ([0, 120, 120], [10, 255, 255]),
            'green': ([45, 100, 100], [75, 255, 255]),
            'blue': ([100, 120, 100], [130, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255])
        }
        
        # For watch detection
        self.watch_detected = False
        self.watch_hand_idx = None
        self.watch_detection_threshold = 5  # Frames to confirm watch detection
        self.watch_frames_detected = 0
        self.watch_color_lower = np.array([0, 0, 0])  # Black/dark color for watch
        self.watch_color_upper = np.array([180, 255, 50])  # Dark objects
        
        # For logging
        self.log_file = None
        
    def set_debug_mode(self, debug_mode, log_file=None):
        """Enable or disable debug mode with optional log file"""
        self.debug_mode = debug_mode
        self.log_file = log_file
        
    def log(self, message, level="INFO"):
        """Log a message to console and/or file if in debug mode"""
        if not self.debug_mode and level != "ERROR":
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        # Always print errors regardless of debug mode
        if level == "ERROR":
            print(f"\033[91m{formatted_msg}\033[0m")  # Red color for errors
        elif level == "WARNING":
            print(f"\033[93m{formatted_msg}\033[0m")  # Yellow for warnings
        elif level == "SUCCESS":
            print(f"\033[92m{formatted_msg}\033[0m")  # Green for success
        elif level == "DEBUG" and self.debug_mode:
            print(f"\033[94m{formatted_msg}\033[0m")  # Blue for debug
        elif self.debug_mode:
            print(formatted_msg)
            
        # Write to log file if specified
        if self.log_file and self.debug_mode:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(formatted_msg + "\n")
            except Exception as e:
                print(f"Error writing to log file: {str(e)}")

    def detect_watch(self, frame, hand_landmarks, idx):
        """Detect if this hand is wearing a watch by looking for dark band near wrist"""
        h, w = frame.shape[:2]
        
        # Get wrist and palm positions
        wrist_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * w)
        wrist_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * h)
        
        # Look for watch in area around and slightly above wrist (watches usually worn above the wrist)
        watch_region_x1 = max(0, wrist_x - 60)
        watch_region_y1 = max(0, wrist_y - 60)  # Check slightly above wrist
        watch_region_x2 = min(w, wrist_x + 60)
        watch_region_y2 = min(h, wrist_y + 10)  # Mostly above the wrist
        
        # Extract region and convert to HSV
        watch_region = frame[watch_region_y1:watch_region_y2, watch_region_x1:watch_region_x2]
        if watch_region.size == 0:
            return False
            
        hsv = cv2.cvtColor(watch_region, cv2.COLOR_BGR2HSV)
        
        # Look for dark band (watch) using color thresholding
        mask = cv2.inRange(hsv, self.watch_color_lower, self.watch_color_upper)
        watch_pixels = np.sum(mask) / 255
        
        # Calculate percentage of dark pixels in the region
        total_pixels = mask.shape[0] * mask.shape[1]
        if total_pixels == 0:
            return False
            
        dark_percentage = watch_pixels / total_pixels
        
        # Debug visualization
        cv2.rectangle(frame, (watch_region_x1, watch_region_y1), 
                     (watch_region_x2, watch_region_y2), (255, 0, 255), 1)
        
        # If enough dark pixels are found, likely a watch
        return dark_percentage > 0.15  # Threshold for watch detection
        
    def update_dynamic_rois(self, frame, timestamp):
        """Update ROIs dynamically based on box positions"""
        # Only update periodically to save processing time
        if timestamp - self.last_roi_update_time < self.roi_update_interval:
            return
            
        self.last_roi_update_time = timestamp
        height, width = frame.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Update chip destination (box) region by detecting box position
        # Look for boxes by color
        combined_box_mask = np.zeros((height, width), dtype=np.uint8)
        for color_name, (lower, upper) in self.box_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_box_mask = cv2.bitwise_or(combined_box_mask, mask)
        
        # Find contours of boxes
        contours, _ = cv2.findContours(combined_box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find large contours that are likely boxes
            box_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
            
            if box_contours:
                # Create bounding boxes around detected boxes
                box_rects = [cv2.boundingRect(cnt) for cnt in box_contours]
                
                # Update chip destination ROI (with margin)
                if box_rects:
                    # Sort by area
                    box_rects.sort(key=lambda r: r[2]*r[3], reverse=True)
                    x, y, w, h = box_rects[0]
                    
                    # Update chip destination ROI (with margin)
                    self.chip_dest_roi = {
                        'x1': max(0, x - 20),
                        'y1': max(0, y - 20),
                        'x2': min(width, x + w + 20),
                        'y2': min(height, y + h + 20)
                    }
                    
                    # Update box destination ROI to be near the detected box
                    self.box_dest_roi = {
                        'x1': max(0, x - 100),
                        'y1': max(0, y - 50),
                        'x2': min(width, x + w + 100),
                        'y2': min(height, y + h + 100)
                    }
        
        # Keep chip source as bottom area with some adjustments if needed
        self.chip_source_roi = {
            'x1': int(width * 0.05),
            'x2': int(width * 0.95),
            'y1': int(height * 0.7),
            'y2': int(height * 0.95)
        }
        
        # Box source typically in upper right region
        if not self.box_source_roi:
            self.box_source_roi = {
                'x1': int(width * 0.7),
                'x2': int(width * 0.95),
                'y1': int(height * 0.1),
                'y2': int(height * 0.4)
            }
        
        print("Dynamic ROIs updated")

    def detect_activity(self, frame, timestamp):
        """Analyze frame to detect participant's activity focusing on dominant hand with watch"""
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Update ROIs dynamically based on object positions
        self.update_dynamic_rois(frame, timestamp)
        
        # Debug frame dimensions
        height, width = frame.shape[:2]
        
        # If no hands detected
        if not results.multi_hand_landmarks:
            self.time_in_state += 1
            
            # Draw all hand landmarks for debugging even if no hands
            self._draw_debug_landmarks(frame, results)
            
            # Check for objects in hand by looking at pixel colors
            if len(self.hand_positions) > 0:
                last_hand_pos = self.hand_positions[-1]
                hand_region = frame[
                    max(0, last_hand_pos[1]-50):min(height, last_hand_pos[1]+50), 
                    max(0, last_hand_pos[0]-50):min(width, last_hand_pos[0]+50)
                ]
                if hand_region.size > 0 and self._detect_object_in_region(hand_region):
                    # Maintain current state if holding something
                    if self.current_action in ["Taking the chip", "Taking the box"]:
                        return self.current_action
            
            # Consider off-task if hands not detected for enough frames
            self.consecutive_off_task_frames += 1
            if self.consecutive_off_task_frames > 5:  # Just 5 frames (0.2 seconds at 25Hz)
                self.is_holding_chip = False
                self.is_holding_box = False
                self.state_memory.append("Off Task")
                return "Off Task"
                
            # Maintain current state briefly to handle detection gaps
            if self.current_action != "Off Task":
                return self.current_action
            
            return "Off Task"
        
        # Reset off-task counter since hands are detected
        self.consecutive_off_task_frames = 0
        self.time_in_state = 0
        
        # Draw all hand landmarks for debugging
        self._draw_debug_landmarks(frame, results)
        
        # Try to identify the watch-wearing (dominant) hand
        dominant_hand = None
        dominant_hand_idx = None
        
        # If we haven't confirmed the watch hand yet, or occasionally recheck
        if not self.watch_detected or (timestamp % 10 < 0.1):  # Recheck every 10 seconds
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if self.detect_watch(frame, hand_landmarks, idx):
                    self.watch_frames_detected += 1
                    self.watch_hand_idx = idx
                    
                    # Confirm watch detection after enough consistent frames
                    if self.watch_frames_detected >= self.watch_detection_threshold:
                        self.watch_detected = True
                        print(f"Watch confirmed on hand {idx}")
                    break
        
        # Use the confirmed watch hand or try to find it
        if self.watch_detected and len(results.multi_hand_landmarks) > self.watch_hand_idx:
            dominant_hand = results.multi_hand_landmarks[self.watch_hand_idx]
            dominant_hand_idx = self.watch_hand_idx
        else:
            # If watch not detected yet, use the first hand
            dominant_hand = results.multi_hand_landmarks[0]
            dominant_hand_idx = 0
        
        if dominant_hand:
            # Track hand center and fingertips
            hand_center = self._get_hand_center(dominant_hand, frame)
            thumb_tip, index_tip = self._get_fingertips(dominant_hand, frame)
            pinch_distance = self._calculate_pinch_distance(thumb_tip, index_tip)
            
            # Add current position to history
            self.hand_positions.append(hand_center)
            
            # Determine current activity based on hand position and gesture
            activity = self._determine_activity(hand_center, pinch_distance, frame)
            
            # Update state memory
            if activity != self.current_action:
                print(f"Activity changed: {self.current_action} -> {activity}")
                self.state_memory.append(activity)
                self.current_action = activity
            
            return activity
        
        # Default to off task if no hands were successfully analyzed
        return "Off Task"
        
    def _draw_debug_landmarks(self, frame, results):
        """Draw hand landmarks for debugging"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)
                    
    def _detect_object_in_region(self, region):
        """Simple detection of objects in region by color analysis"""
        if region.size == 0:
            return False
            
        # Check for chip colors (red, green, blue, white)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Red chips
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Green chips
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Blue chips
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combined mask
        combined_mask = red_mask | green_mask | blue_mask
        
        # Check if enough pixels match chip colors
        return np.sum(combined_mask) > 100
    
    def _get_hand_center(self, hand_landmarks, frame):
        """Calculate the center point of the hand"""
        h, w = frame.shape[:2]
        cx = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * w)
        cy = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * h)
        return (cx, cy)
    
    def _get_fingertips(self, hand_landmarks, frame):
        """Get positions of thumb and index fingertips"""
        h, w = frame.shape[:2]
        
        thumb_tip = (
            int(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * w),
            int(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * h)
        )
        
        index_tip = (
            int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
            int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
        )
        
        return thumb_tip, index_tip
    
    def _calculate_pinch_distance(self, thumb_tip, index_tip):
        """Calculate normalized distance between thumb and index finger"""
        return np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2) / 100
        
    def _check_chip_in_hand(self, frame, hand_center):
        """Check if there's actually a chip in the hand region with improved accuracy"""
        h, w = frame.shape[:2]
        
        # Define hand region (area around and below the hand center)
        hand_region_x1 = max(0, hand_center[0] - 100)
        hand_region_y1 = max(0, hand_center[1] - 30)
        hand_region_x2 = min(w, hand_center[0] + 100)
        hand_region_y2 = min(h, hand_center[1] + 80)
        
        # Extract hand region
        hand_region = frame[hand_region_y1:hand_region_y2, hand_region_x1:hand_region_x2]
        
        if hand_region.size == 0:
            return False
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Check for poker chip colors with blob detection
        chip_detected = False
        chip_area = 0
        color_detected = None
        
        for color_name, (lower, upper) in self.chip_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any contours are large enough and have a circular shape
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area for a chip
                    # Check if the contour is circular (chips are circular)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:  # Value close to 1 indicates circle
                            chip_detected = True
                            chip_area = max(chip_area, area)
                            color_detected = color_name
                            
                            # Draw contour for debugging
                            cv2.drawContours(hand_region, [contour], -1, (0, 255, 0), 2)
        
        # Update confidence based on detection
        if chip_detected:
            self.frames_with_chip += 1
            self.frames_without_chip = 0
            # Set confidence based on area and detection history
            self.chip_holding_confidence = min(1.0, self.chip_holding_confidence + 0.2)
            self.log(f"Detected {color_detected} chip in hand (area: {chip_area:.1f}, confidence: {self.chip_holding_confidence:.2f})", "DEBUG")
        else:
            self.frames_without_chip += 1
            self.frames_with_chip = 0
            # Gradually decrease confidence when chip not detected
            self.chip_holding_confidence = max(0.0, self.chip_holding_confidence - 0.1)
        
        # Debug - draw the hand region
        cv2.rectangle(frame, (hand_region_x1, hand_region_y1), 
                     (hand_region_x2, hand_region_y2), (255, 0, 255), 1)
                
        # Only return true if confidence is high enough
        return self.chip_holding_confidence > 0.6
        
    def _check_box_in_hand(self, frame, hand_center):
        """Check if there's a box in the hand region with improved accuracy"""
        h, w = frame.shape[:2]
        
        # Define broader hand region for box detection
        hand_region_x1 = max(0, hand_center[0] - 150)
        hand_region_y1 = max(0, hand_center[1] - 50)
        hand_region_x2 = min(w, hand_center[0] + 150)
        hand_region_y2 = min(h, hand_center[1] + 150)
        
        # Extract region
        hand_region = frame[hand_region_y1:hand_region_y2, hand_region_x1:hand_region_x2]
        
        if hand_region.size == 0:
            return False
            
        # Convert to HSV
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Check for box colors with shape analysis
        box_detected = False
        box_area = 0
        color_detected = None
        
        for color_name, (lower, upper) in self.box_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any contours are box-like
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Boxes are larger than chips
                    # Check if the contour is rectangular (boxes are rectangular)
                    epsilon = 0.05 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # If contour has 4-8 sides, it could be a box
                    if 4 <= len(approx) <= 8:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        
                        # Check aspect ratio for box-like shape
                        if 0.5 <= aspect_ratio <= 2.0:
                            box_detected = True
                            box_area = max(box_area, area)
                            color_detected = color_name
                            
                            # Draw contour for debugging
                            cv2.drawContours(hand_region, [contour], -1, (0, 0, 255), 2)
        
        # Update confidence based on detection
        if box_detected:
            self.box_holding_confidence = min(1.0, self.box_holding_confidence + 0.15)
            self.log(f"Detected {color_detected} box near hand (area: {box_area:.1f}, confidence: {self.box_holding_confidence:.2f})", "DEBUG")
        else:
            self.box_holding_confidence = max(0.0, self.box_holding_confidence - 0.1)
        
        # Debug - draw the detection region
        cv2.rectangle(frame, (hand_region_x1, hand_region_y1), 
                     (hand_region_x2, hand_region_y2), (0, 0, 255), 1)
                
        # Only return true if confidence is high enough
        return self.box_holding_confidence > 0.5
        
    def _color_near_hand(self, frame, hand_center, object_type="chip"):
        """Check if specific object color is near the hand"""
        h, w = frame.shape[:2]
        
        # Define search region
        region_x1 = max(0, hand_center[0] - 80)
        region_y1 = max(0, hand_center[1] - 80)
        region_x2 = min(w, hand_center[0] + 80)
        region_y2 = min(h, hand_center[1] + 80)
        
        # Extract region
        region = frame[region_y1:region_y2, region_x1:region_x2]
        
        if region.size == 0:
            return False
            
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Choose appropriate color set
        colors = self.chip_colors if object_type == "chip" else self.box_colors
        
        # Check for colors
        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 500:  # Low threshold just to detect presence
                return True
                
        return False
    
    def _determine_activity(self, hand_center, pinch_distance, frame):
        """Determine the current activity based on hand position and gesture"""
        # Check if position is in any of the ROIs
        in_chip_source = self._point_in_roi(hand_center, self.chip_source_roi)
        in_chip_dest = self._point_in_roi(hand_center, self.chip_dest_roi)
        in_box_source = self._point_in_roi(hand_center, self.box_source_roi)
        in_box_dest = self._point_in_roi(hand_center, self.box_dest_roi)
        
        # Debug information
        self.log(f"Hand position: {hand_center}, Pinch distance: {pinch_distance:.2f}", "DEBUG")
        self.log(f"In ROIs - Chip source: {in_chip_source}, Chip dest: {in_chip_dest}, Box source: {in_box_source}, Box dest: {in_box_dest}", "DEBUG")
        
        # Check if actually holding a chip by analyzing hand region
        actually_holding_chip = self._check_chip_in_hand(frame, hand_center)
        actually_holding_box = self._check_box_in_hand(frame, hand_center)
        
        # Print actual holding state
        self.log(f"Actually holding chip: {actually_holding_chip} (conf: {self.chip_holding_confidence:.2f}), Actually holding box: {actually_holding_box} (conf: {self.box_holding_confidence:.2f})", "INFO")
        
        # Update holding state based on visual confirmation
        if actually_holding_chip:
            self.is_holding_chip = True
        elif self.chip_holding_confidence < 0.2:  # Reset if confidence is very low
            self.is_holding_chip = False
            
        if actually_holding_box:
            self.is_holding_box = True
        elif self.box_holding_confidence < 0.2:  # Reset if confidence is very low
            self.is_holding_box = False
        
        # More precise grab/release detection
        is_grabbing = pinch_distance < 0.25  # Stricter grabbing threshold
        is_releasing = pinch_distance > 0.35
        
        # If hand is in chip source area
        if in_chip_source:
            # Only set holding chip if we actually see grabbing motion AND chip is visually detected
            if is_grabbing and self._color_near_hand(frame, hand_center, "chip"):
                self.is_holding_chip = True
                self.time_in_current_activity = 0  # Reset activity timer
                self.log("Taking chip from source area", "SUCCESS")
                return "Taking the chip"
            # If already holding a chip, maintain that state
            elif self.is_holding_chip and actually_holding_chip:
                return "Taking the chip"
            # Otherwise just being in the chip source area
            return "Taking the chip"
            
        # If hand is in chip destination area (box)
        elif in_chip_dest:
            # If releasing while holding a chip, mark as placed
            if is_releasing and self.is_holding_chip:
                self.is_holding_chip = False
                self.chip_holding_confidence = 0.0  # Reset confidence
                self.time_in_current_activity = 0  # Reset activity timer
                self.log("Placing chip in destination", "SUCCESS")
                return "Placing the chip"
            # If still holding a chip in the destination area
            elif self.is_holding_chip and actually_holding_chip:
                return "Placing the chip"
            # Even without holding, being in dest area suggests placing
            return "Placing the chip"
            
        # If hand is in box source area
        elif in_box_source:
            # Only set holding box if we actually see grabbing motion AND box color is detected
            if is_grabbing and self._color_near_hand(frame, hand_center, "box"):
                self.is_holding_box = True
                self.time_in_current_activity = 0  # Reset activity timer
                self.log("Taking box from source area", "SUCCESS")
                return "Taking the box"
            # If we're already tracking box holding
            elif self.is_holding_box and actually_holding_box:
                return "Taking the box"
            # Otherwise being in box area suggests intent
            return "Taking the box"
            
        # If hand is in box destination area
        elif in_box_dest:
            # If releasing while holding a box, mark as placed
            if is_releasing and self.is_holding_box:
                self.is_holding_box = False
                self.box_holding_confidence = 0.0  # Reset confidence
                self.time_in_current_activity = 0  # Reset activity timer
                self.log("Placing box in destination", "SUCCESS")
                return "Placing the box"
            # If still holding box in destination area
            elif self.is_holding_box and actually_holding_box:
                return "Placing the box"
            # Being in box dest area suggests placing intent
            return "Placing the box"
            
        # If we're holding objects outside specific regions
        if self.is_holding_chip and actually_holding_chip:
            # If we confirm visually that a chip is being held, maintain chip taking state
            return "Taking the chip"
        elif self.is_holding_box and actually_holding_box:
            # If we confirm visually that a box is being held, maintain box taking state
            return "Taking the box"
            
        # Check if hand is moving significantly (even if not in an ROI)
        if len(self.hand_positions) >= 5:  # Need more points for better movement detection
            # Calculate recent movement
            recent_positions = list(self.hand_positions)[-5:]
            movement_vector = np.array(recent_positions[-1]) - np.array(recent_positions[0])
            movement = np.linalg.norm(movement_vector)
            
            # If hand is moving significantly, it's not off-task
            if movement > 20:  # Lower threshold to be less sensitive
                self.log(f"Significant hand movement detected: {movement:.2f} pixels", "DEBUG")
                
                # If we were previously in a task, maintain it briefly
                if self.current_action != "Off Task":
                    return self.current_action
                    
                # If coming from off-task, but moving toward chip area
                angle = np.arctan2(movement_vector[1], movement_vector[0]) * 180 / np.pi
                if -45 < angle < 45 and hand_center[0] < self.chip_source_roi['x1']:
                    return "Taking the chip"
        
        # More aggressive Off Task detection - any pauses are considered Off Task
        self.time_without_activity += 1
        
        # Even brief pauses (0.2 seconds at 25Hz) now count as "Off Task"
        if self.time_without_activity > 5:
            if self.current_action != "Off Task":
                self.log("Detected Off Task - participant paused or inactive", "WARNING")
            return "Off Task"
        
        # Otherwise maintain previous activity for a brief period
        return self.current_action
    
    def _point_in_roi(self, point, roi):
        """Check if a point is inside a region of interest"""
        x, y = point
        return (roi['x1'] <= x <= roi['x2'] and roi['y1'] <= y <= roi['y2'])
    
    def _calculate_movement(self, start_pos, end_pos):
        """Calculate movement magnitude between two positions"""
        if start_pos and end_pos:
            return np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        return 0

class VideoAnnotator:
    def __init__(self, video_path=None, sampling_rate=25, output_path=None, real_time_output=False, debug_mode=False, log_file=None):
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
        
        while cap.isOpened():
            # Handle paused state
            if self.paused and not self.step_mode:
                # Show the current frame with "PAUSED" overlay
                paused_frame = frame.copy()
                cv2.putText(paused_frame, "PAUSED - Press 'P' to resume, 'S' for step", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Video Annotation', paused_frame)
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
            if self.step_mode:
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
                if self.frame_delay > 1:
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
        cv2.destroyAllWindows()
        
        # Save final annotations if not already saved
        if not (self.real_time_output and self.output_path):
            self._save_annotations(self.output_path)
            
        return True
    
    def _draw_visualization(self, frame, activity, timestamp):
        """Draw visualization on frame for debugging with enhanced information"""
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

class AnnotationGUI:
    """GUI for selecting video files and configuring annotation parameters"""
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        
        self.root = root
        self.root.title("Hand Task Video Annotator")
        self.root.geometry("700x600")
        
        # Create style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=6)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        # Video file selection
        ttk.Label(file_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        self.video_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_video).grid(row=0, column=2)
        
        # Output file selection
        ttk.Label(file_frame, text="Output CSV:").grid(row=1, column=0, sticky=tk.W)
        self.output_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2)
        
        # Log file selection
        ttk.Label(file_frame, text="Log File:").grid(row=2, column=0, sticky=tk.W)
        self.log_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.log_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_log).grid(row=2, column=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Annotation Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        
        # Sampling rate
        ttk.Label(params_frame, text="Sampling Rate (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.sampling_rate = tk.IntVar(value=25)
        ttk.Spinbox(params_frame, from_=1, to=60, textvariable=self.sampling_rate, width=5).grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Playback speed
        ttk.Label(params_frame, text="Initial Delay (ms):").grid(row=0, column=2, sticky=tk.W)
        self.frame_delay = tk.IntVar(value=30)  # Default slower for better visibility
        ttk.Spinbox(params_frame, from_=1, to=500, textvariable=self.frame_delay, width=5).grid(row=0, column=3, padx=5, sticky=tk.W)
        
        # Checkbox options
        self.real_time_output = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Real-time Output", variable=self.real_time_output).grid(row=1, column=0, sticky=tk.W)
        
        self.debug_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Debug Mode", variable=self.debug_mode).grid(row=1, column=1, sticky=tk.W)
        
        self.start_paused = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Start Paused", variable=self.start_paused).grid(row=1, column=2, sticky=tk.W)
        
        # Status and progress frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        # Progress bar
        ttk.Label(status_frame, text="Progress:").grid(row=0, column=0, sticky=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=500)
        self.progress_bar.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Current activity and time
        ttk.Label(status_frame, text="Current Activity:").grid(row=1, column=0, sticky=tk.W)
        self.current_activity = tk.StringVar(value="Not started")
        ttk.Label(status_frame, textvariable=self.current_activity).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(status_frame, text="Current Time:").grid(row=2, column=0, sticky=tk.W)
        self.current_time = tk.StringVar(value="00:00:00")
        ttk.Label(status_frame, textvariable=self.current_time).grid(row=2, column=1, sticky=tk.W)
        
        # Console output frame (with scrollbar)
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(console_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add Text widget
        self.console = tk.Text(console_frame, wrap=tk.WORD, height=10, 
                              yscrollcommand=scrollbar.set)
        self.console.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.console.yview)
        
        # Redirect stdout to the console
        import sys
        self.original_stdout = sys.stdout
        sys.stdout = self
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Add buttons
        ttk.Button(button_frame, text="Start Annotation", command=self.start_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Console", command=self.clear_console).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.exit_application).pack(side=tk.RIGHT, padx=5)
        
        # Annotation process
        self.annotator = None
        self.running = False
        self.stop_flag = False
        
    def write(self, text):
        """Redirect stdout to the console"""
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.root.update_idletasks()
        
    def flush(self):
        """Required for stdout redirection"""
        pass
        
    def browse_video(self):
        """Browse for video file"""
        filename = self.filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if filename:
            self.video_path.set(filename)
            # Auto-suggest output filename
            if not self.output_path.get():
                output = filename.rsplit(".", 1)[0] + "_annotations.csv"
                self.output_path.set(output)
            # Auto-suggest log filename
            if not self.log_path.get():
                log = filename.rsplit(".", 1)[0] + "_log.txt"
                self.log_path.set(log)
                
    def browse_output(self):
        """Browse for output CSV file"""
        filename = self.filedialog.asksaveasfilename(
            title="Select Output CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            defaultextension=".csv"
        )
        if filename:
            self.output_path.set(filename)
            
    def browse_log(self):
        """Browse for log file"""
        filename = self.filedialog.asksaveasfilename(
            title="Select Log File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            defaultextension=".txt"
        )
        if filename:
            self.log_path.set(filename)
            
    def update_progress(self, progress, activity, timestamp):
        """Update progress bar and status"""
        self.progress_var.set(progress)
        self.current_activity.set(activity)
        
        # Format timestamp
        minutes, seconds = divmod(int(timestamp), 60)
        hours, minutes = divmod(minutes, 60)
        self.current_time.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update UI
        self.root.update_idletasks()
        
    def start_annotation(self):
        """Start the annotation process"""
        if self.running:
            self.messagebox.showwarning("Warning", "Annotation is already running!")
            return
            
        # Validate inputs
        if not self.video_path.get():
            self.messagebox.showerror("Error", "Please select a video file!")
            return
            
        if not self.output_path.get():
            self.messagebox.showerror("Error", "Please specify an output CSV file!")
            return
            
        # Create annotator
        self.annotator = VideoAnnotator(
            video_path=self.video_path.get(),
            sampling_rate=self.sampling_rate.get(),
            output_path=self.output_path.get(),
            real_time_output=self.real_time_output.get(),
            debug_mode=self.debug_mode.get(),
            log_file=self.log_path.get() if self.log_path.get() else None
        )
        
        # Set initial frame delay
        self.annotator.frame_delay = self.frame_delay.get()
        
        # Start paused if selected
        self.annotator.paused = self.start_paused.get()
        
        # Clear console
        self.clear_console()
        
        # Start annotation process in a separate thread
        import threading
        self.running = True
        self.stop_flag = False
        threading.Thread(target=self.run_annotation).start()
        
    def run_annotation(self):
        """Run annotation in background thread"""
        try:
            # Run the annotation process
            success = self.annotator.process_video(gui_callback=self.update_progress)
            
            if self.stop_flag:
                self.write("\n[INFO] Annotation stopped by user\n")
            elif success:
                self.write("\n[INFO] Annotation completed successfully!\n")
            else:
                self.write("\n[ERROR] Annotation failed\n")
                
        except Exception as e:
            import traceback
            self.write(f"\n[ERROR] Exception during annotation: {str(e)}\n")
            self.write(traceback.format_exc())
            
        finally:
            self.running = False
            self.progress_var.set(0)
            # Reset stdout
            import sys
            sys.stdout = self.original_stdout
            
    def stop_annotation(self):
        """Stop the annotation process"""
        if not self.running:
            self.messagebox.showinfo("Info", "No annotation process is running")
            return
            
        self.stop_flag = True
        # If annotator exists, set paused to True to help the process notice it should stop
        if self.annotator:
            self.annotator.paused = True
            
    def clear_console(self):
        """Clear the console output"""
        self.console.delete(1.0, tk.END)
        
    def exit_application(self):
        """Exit the application"""
        if self.running:
            if self.messagebox.askyesno("Warning", "Annotation is still running. Are you sure you want to exit?"):
                self.stop_annotation()
                self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()


def test_roi_configuration():
    """Test function to verify ROI configuration with a single frame"""
    video_path = "P19.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Create detector with debug mode
    detector = EnhancedHandTaskDetector(debug_mode=True)
    
    # Update ROIs dynamically
    detector.update_dynamic_rois(frame, 0)
    
    # Draw ROIs on frame
    draw_frame = frame.copy()
    
    # Draw chip source ROI
    cv2.rectangle(draw_frame, 
                (detector.chip_source_roi['x1'], detector.chip_source_roi['y1']),
                (detector.chip_source_roi['x2'], detector.chip_source_roi['y2']),
                (0, 0, 255), 2)  # Red
    cv2.putText(draw_frame, "Chip Source", 
               (detector.chip_source_roi['x1'], detector.chip_source_roi['y1'] - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw chip destination ROI if detected
    if detector.chip_dest_roi:
        cv2.rectangle(draw_frame, 
                    (detector.chip_dest_roi['x1'], detector.chip_dest_roi['y1']),
                    (detector.chip_dest_roi['x2'], detector.chip_dest_roi['y2']),
                    (0, 255, 0), 2)  # Green
        cv2.putText(draw_frame, "Chip Dest", 
                   (detector.chip_dest_roi['x1'], detector.chip_dest_roi['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw box source ROI
    cv2.rectangle(draw_frame, 
                (detector.box_source_roi['x1'], detector.box_source_roi['y1']),
                (detector.box_source_roi['x2'], detector.box_source_roi['y2']),
                (255, 0, 0), 2)  # Blue
    cv2.putText(draw_frame, "Box Source", 
               (detector.box_source_roi['x1'], detector.box_source_roi['y1'] - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw box destination ROI if detected
    if detector.box_dest_roi:
        cv2.rectangle(draw_frame, 
                    (detector.box_dest_roi['x1'], detector.box_dest_roi['y1']),
                    (detector.box_dest_roi['x2'], detector.box_dest_roi['y2']),
                    (0, 255, 255), 2)  # Yellow
        cv2.putText(draw_frame, "Box Dest", 
                   (detector.box_dest_roi['x1'], detector.box_dest_roi['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Show instructions
    cv2.putText(draw_frame, "Press any key to close", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('ROI Configuration', draw_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()
    print("ROI configuration test complete!")

def main():
    # Check if GUI mode is requested (default)
    if '--no-gui' not in sys.argv:
        try:
            import tkinter as tk
            root = tk.Tk()
            app = AnnotationGUI(root)
            root.mainloop()
            return
        except ImportError:
            print("Tkinter not available. Falling back to command line mode.")
            # Fall through to command-line mode
    
    # Command-line mode
    parser = argparse.ArgumentParser(description='Automated video annotation for task analysis')
    parser.add_argument('--video', type=str, default="P19.mp4", help='Path to input video file')
    parser.add_argument('--output', type=str, default="P19_annotations.csv", help='Path to output CSV file')
    parser.add_argument('--rate', type=int, default=25, help='Sampling rate in Hz')
    parser.add_argument('--real-time', action='store_true', help='Enable real-time output to CSV')
    parser.add_argument('--test-roi', action='store_true', help='Test ROI configuration on first frame')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--log-file', type=str, help='Path to log file for debug output')
    parser.add_argument('--delay', type=int, default=1, help='Frame delay in ms (higher = slower playback)')
    parser.add_argument('--no-gui', action='store_true', help='Force command-line mode (no GUI)')
    args = parser.parse_args()
    
    # Test ROI configuration if requested
    if args.test_roi:
        test_roi_configuration()
        return
        
    print(f"Processing video: {args.video}")
    print(f"Output file: {args.output}")
    print(f"Sampling rate: {args.rate}Hz")
    print(f"Real-time output: {'Enabled' if args.real_time else 'Disabled'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    # Create and run the annotator
    annotator = VideoAnnotator(
        video_path=args.video, 
        sampling_rate=args.rate,
        output_path=args.output,
        real_time_output=args.real_time,
        debug_mode=args.debug,
        log_file=args.log_file
    )
    
    # Set frame delay
    annotator.frame_delay = args.delay
    
    try:
        annotator.process_video()
        print("Annotation complete!")
    except Exception as e:
        print(f"Error during annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\nKeyboard shortcuts while running:")
    print("  'q' - Quit the annotation process")
    print("  'f' - Fast-forward 10 seconds (useful to skip inactive segments)")
    print("  'p' - Pause/resume processing")
    print("  '+'/'-' - Increase/decrease playback speed")

if __name__ == "__main__":
    main()