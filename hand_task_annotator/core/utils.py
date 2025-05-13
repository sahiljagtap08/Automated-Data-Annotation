"""
Utility functions for the hand task annotation system
"""

import cv2
import numpy as np
import os
import re
from datetime import datetime


def format_timestamp(seconds):
    """Format seconds into HH:MM:SS.mmm format"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def get_readable_time(unix_timestamp_ms=None):
    """
    Convert unix timestamp in milliseconds to human-readable time format
    
    Args:
        unix_timestamp_ms: Unix timestamp in milliseconds
        
    Returns:
        String in format "HH:MM:SS.mmm"
    """
    if unix_timestamp_ms is None:
        # Use current time if no timestamp provided
        dt = datetime.now()
    else:
        # Convert unix timestamp to datetime
        dt = datetime.fromtimestamp(unix_timestamp_ms / 1000.0)
        
    return dt.strftime("%H:%M:%S.%f")[:-3]


def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from a filename that contains timestamp information
    
    Args:
        filename: Filename string potentially containing timestamp
        
    Returns:
        Unix timestamp in milliseconds if found, None otherwise
    """
    # Try to find timestamp patterns
    # Look for unix timestamp (epoch)
    unix_match = re.search(r'_(\d{10,13})_?', filename)
    if unix_match:
        unix_time = int(unix_match.group(1))
        # Check if milliseconds or seconds
        if len(unix_match.group(1)) <= 10:
            unix_time *= 1000  # Convert seconds to milliseconds
        return unix_time
        
    # Look for YYYYMMDD_HHMMSS format
    date_match = re.search(r'(\d{8})_(\d{6})', filename)
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
            
    return None


def find_matching_timestamp(target_time, timestamp_list, tolerance_ms=500):
    """
    Find closest timestamp in a list within tolerance
    
    Args:
        target_time: Target timestamp in milliseconds
        timestamp_list: List of available timestamps in milliseconds
        tolerance_ms: Maximum allowed difference in milliseconds
        
    Returns:
        Closest matching timestamp, or None if no match within tolerance
    """
    closest_time = None
    min_diff = tolerance_ms
    
    for ts in timestamp_list:
        diff = abs(ts - target_time)
        if diff < min_diff:
            min_diff = diff
            closest_time = ts
            
    return closest_time


def create_roi(x1, y1, x2, y2):
    """Create a Region of Interest dictionary"""
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


def detect_objects_by_color(frame, color_lower, color_upper, min_area=500, max_area=None, 
                           circularity_threshold=None, aspect_ratio_range=None):
    """
    Detect objects in frame based on color and shape characteristics
    
    Args:
        frame: Input image frame
        color_lower: Lower HSV bounds for color detection
        color_upper: Upper HSV bounds for color detection
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider (optional)
        circularity_threshold: If set, filter for circular objects (0-1, where 1 is perfect circle)
        aspect_ratio_range: If set, tuple of (min, max) aspect ratio values
        
    Returns:
        List of contours meeting the criteria
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask based on color
    mask = cv2.inRange(hsv, color_lower, color_upper)
    
    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on criteria
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip if area is too small or too large
        if area < min_area:
            continue
            
        if max_area is not None and area > max_area:
            continue
            
        # Check circularity if threshold provided
        if circularity_threshold is not None:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < circularity_threshold:
                    continue
                    
        # Check aspect ratio if range provided
        if aspect_ratio_range is not None:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                aspect_ratio = float(w) / h
                min_ratio, max_ratio = aspect_ratio_range
                if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                    continue
                    
        # All criteria passed
        filtered_contours.append(contour)
        
    return filtered_contours


def draw_contours_with_info(frame, contours, color=(0, 255, 0), thickness=2, show_area=True, show_center=True):
    """
    Draw contours on frame with optional information about each contour
    
    Args:
        frame: Input image frame
        contours: List of contours to draw
        color: BGR color tuple for contour drawing
        thickness: Line thickness
        show_area: Whether to display contour area
        show_center: Whether to display center point
    """
    for i, contour in enumerate(contours):
        # Draw the contour
        cv2.drawContours(frame, [contour], -1, color, thickness)
        
        # Get contour properties
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        
        # Get and mark center point
        if M["m00"] != 0 and show_center:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
        # Show area if requested
        if show_area:
            x, y, w, h = cv2.boundingRect(contour)
            label = f"#{i}: {area:.0f}"
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def apply_text_with_background(frame, text, position, font_scale=1.0, thickness=2, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Draw text with a contrasting background for better visibility
    
    Args:
        frame: Input image frame
        text: Text string to display
        position: (x, y) position of text
        font_scale: Font scale factor
        thickness: Line thickness
        text_color: BGR color tuple for text
        bg_color: BGR color tuple for background
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle
    padding = 5
    x, y = position
    cv2.rectangle(frame, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + padding), 
                 bg_color, -1)
    
    # Draw text over background
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness) 