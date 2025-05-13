"""
Hand activity detector using MediaPipe and computer vision techniques
"""

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from collections import deque


class EnhancedHandTaskDetector:
    """
    Detects hand activities in video frames using MediaPipe hand tracking.
    
    This class implements watch detection to identify the dominant hand,
    dynamic ROI tracking, and confidence-based object detection to classify
    activities like taking/placing chips and boxes.
    """
    
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
        
        self.log("Dynamic ROIs updated", "DEBUG")

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
                        self.log(f"Watch confirmed on hand {idx}", "SUCCESS")
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
                self.log(f"Activity changed: {self.current_action} -> {activity}", "INFO")
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
        if not roi:
            return False
        x, y = point
        return (roi['x1'] <= x <= roi['x2'] and roi['y1'] <= y <= roi['y2'])
    
    def _calculate_movement(self, start_pos, end_pos):
        """Calculate movement magnitude between two positions"""
        if start_pos and end_pos:
            return np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        return 0 