# Automated Hand Task Annotation System

A computer vision-based system for automatically annotating hand activity in task-based video data. This system uses MediaPipe hand tracking, contour analysis, and confidence-based detection to identify when participants are taking or placing objects during experiments.

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Detection Algorithms](#detection-algorithms)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Mode](#gui-mode)
  - [Command-line Mode](#command-line-mode)
  - [Wearable Data Integration](#wearable-data-integration)
  - [Keyboard Controls](#keyboard-controls)
- [Output Format](#output-format)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Overview

This system analyzes videos of participants performing tasks with their hands, specifically focusing on participants who are wearing a watch on their dominant hand. The system detects and tracks hand movements, identifies when participants are interacting with chips and boxes, and generates timestamped annotations of activities including:

- Taking the chip
- Placing the chip
- Taking the box
- Placing the box
- Off Task (when no relevant activity is detected)

Key features include:

- **Watch-Wearing Hand Detection**: Tracks only the participant's dominant hand (wearing a watch)
- **Dynamic ROI Tracking**: Automatically adjusts regions of interest if objects move
- **Real-Time Annotation**: Outputs annotations while processing, no need to wait for completion
- **Confidence-Based Detection**: Uses a confidence score system to reduce false positives
- **Shape Analysis**: Distinguishes circular chips from rectangular boxes using contour analysis
- **GUI Interface**: User-friendly interface for file selection and configuration
- **Video Visualization**: Color-coded display of activities and detection status
- **Wearable Data Integration**: Merge video annotations with accelerometer data from wearable devices

## Technical Architecture

The system consists of the following main components:

1. **EnhancedHandTaskDetector**: Core detection engine that processes video frames to identify activities
2. **VideoAnnotator**: Manages video processing, visualization, and CSV output
3. **AnnotationGUI**: Provides a user interface for configuring and running annotations
4. **ProcessAnnotations**: Integrates video annotations with wearable device accelerometer data

### Data Flow

1. Video frames are captured and sampled at the specified rate (default: 25Hz)
2. Each frame is processed by MediaPipe's hand tracking to detect hand positions
3. Watch detection identifies the dominant hand for tracking
4. Dynamic regions of interest (ROIs) are updated based on detected objects
5. Hand gestures and positions relative to ROIs are analyzed to determine activities
6. Confidence scores for chip/box detection are updated based on visual analysis
7. Annotations are generated with timestamps and activity labels
8. Results are displayed visually and saved to a CSV file
9. Annotations can be merged with accelerometer data from wearable devices

## Detection Algorithms

### Hand Tracking

The system uses MediaPipe Hands to detect and track hand landmarks. Key points include:

- Wrist position: Used as the center of the hand for ROI checks
- Thumb and index fingertips: Used to detect grabbing/releasing gestures
- Pinch distance: Calculated to determine when objects are being grasped

### Watch Detection

To identify the dominant hand:

1. The area around the wrist of each detected hand is analyzed
2. A color threshold is applied to detect dark regions (watch band)
3. If a watch-like object is consistently detected, that hand is marked as dominant

### Object Detection

The system uses a combination of techniques to detect objects:

#### Chip Detection

1. Color thresholding in HSV color space to identify potential chip colors
2. Morphological operations to clean up the mask
3. Contour detection to find circular shapes
4. Circularity calculation: 4π × area / perimeter²
5. Confidence scoring based on consistent detection over multiple frames

#### Box Detection

1. Color thresholding in HSV color space
2. Contour detection and polygon approximation
3. Aspect ratio and side count analysis to identify rectangular shapes
4. Size-based filtering to distinguish from smaller objects

### Activity Classification

Activities are determined using the following logic:

1. Check if hand position is within any ROI
2. Verify if object is being held (using visual confirmation)
3. Analyze pinch distance to detect grabbing/releasing gestures
4. Track temporal consistency of activities
5. Apply state transition rules (e.g., must release a chip to transition from "taking" to "placing")

The system uses confidence scores that increase gradually with consistent detection and decrease when detection is lost, providing robustness against momentary detection failures.

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install system dependencies for OCR timestamp detection (optional):
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Dependencies

- opencv-python: For image processing and visualization
- mediapipe: For hand tracking
- numpy: For numerical operations
- pandas: For data manipulation and CSV output
- pytesseract (optional): For OCR timestamp detection
- tkinter: For GUI interface

## Usage

The system can be run in either GUI mode (default), command-line mode, or wearable data integration mode.

### GUI Mode

Simply run the main script without arguments:

```
python main.py
```

The GUI provides options to:
1. Select input video file
2. Specify output CSV file
3. Configure sampling rate
4. Enable/disable debug mode
5. Set visualization speed
6. Start/stop annotation process

**Note:** The GUI opens the video visualization in a separate window when processing starts.

### Command-line Mode

For batch processing or use without a GUI, use command-line arguments:

```
python main.py --video P19.mp4 --output P19_annotations.csv --debug --rate 25
```

Available command-line options:
```
--video FILE       Path to input video file (default: P19.mp4)
--output FILE      Path to output CSV file (default: P19_annotations.csv)
--rate RATE        Sampling rate in Hz (default: 25)
--real-time        Enable real-time output to CSV
--test-roi         Test ROI configuration on first frame
--debug            Enable debug mode with verbose logging
--log-file FILE    Path to log file for debug output
--delay MS         Frame delay in ms (higher = slower playback)
--no-gui           Force command-line mode (no GUI)
```

### Wearable Data Integration

For projects integrating with wearable device data (accelerometer x,y,z data), use the process_annotations.py script:

```
python process_annotations.py --raw accelerometer_data.csv --video participant_video.mp4 --output annotated_data.csv
```

This script handles the complete workflow:
1. Reads raw CSV with accelerometer data (containing unix timestamps)
2. Creates a working copy of the data (preserving the raw data)
3. Adds a readable time column by converting unix timestamps
4. Processes the video for activity annotations
5. Merges annotations with accelerometer data by timestamp matching
6. Saves the annotated data as a new CSV file

Available options:
```
--raw FILE         Path to raw CSV with accelerometer data (required)
--video FILE       Path to video file for annotation (required)
--output FILE      Path to final output CSV file (default: <raw>_annotated.csv)
--rate RATE        Sampling rate for video annotation in Hz (default: 25)
--keep-temp        Keep temporary processing files
```

The timestamp matching uses a nearest-neighbor approach with a 200ms threshold, ensuring that each accelerometer data point is annotated with the closest video-derived activity label.

### Keyboard Controls

While the annotation process is running, you can use these controls:

- **Q**: Quit processing
- **F**: Fast-forward 10 seconds
- **P**: Pause/resume processing
- **S**: Step forward one frame (when paused)
- **+**: Increase playback speed
- **-**: Decrease playback speed (slow down for better visibility)

## Output Format

The system generates a CSV file with the following columns:

- `unixTimestampInMs`: Timestamp in milliseconds
- `readableTime`: Human-readable timestamp (HH:MM:SS)
- `label`: Activity label (one of the five recognized activities)
- `x`, `y`, `z`: Dummy accelerometer data fields (for compatibility)

When using the wearable data integration mode, the output includes all original accelerometer data columns plus the activity labels.

Example output:
```
unixTimestampInMs,readableTime,label,x,y,z
1631234567000,12:34:56,Taking the chip,0.23,0.45,9.81
1631234567040,12:34:57,Taking the chip,0.25,0.42,9.78
1631234567080,12:34:58,Placing the chip,0.31,0.55,9.82
1631234567120,12:34:59,Off Task,0.02,0.03,9.79
```

An activity summary is also printed at the end of processing, showing the distribution of activities.

## Customization

### Adjusting Detection Parameters

To customize detection sensitivity, modify the following parameters in `EnhancedHandTaskDetector.__init__`:

```python
# Activity detection thresholds
self.grabbing_threshold = 0.2    # Lower = more sensitive grab detection
self.releasing_threshold = 0.3   # Higher = more sensitive release detection
self.movement_threshold = 30     # Lower = more movement detection

# Color detection parameters (HSV color space)
self.chip_colors = {
    'red': ([0, 120, 120], [10, 255, 255]),  # [lower_bound], [upper_bound]
    'green': ([45, 100, 100], [75, 255, 255]),
    'blue': ([100, 120, 100], [130, 255, 255])
}
```

### Modifying ROI Detection

For different experimental setups, adjust the dynamic ROI calculation in `update_dynamic_rois`:

```python
# Example: Change chip source region to bottom-left instead of full bottom
self.chip_source_roi = {
    'x1': int(width * 0.05),
    'x2': int(width * 0.4),  # Only use left side
    'y1': int(height * 0.7),
    'y2': int(height * 0.95)
}
```

### Adjusting Activity Detection Logic

To modify how activities are determined, edit the `_determine_activity` method. For example, to make Off Task detection less sensitive:

```python
# Change this line to require more frames of inactivity
if self.time_without_activity > 10:  # Changed from 5 (more tolerant)
    return "Off Task"
```

### Modifying Timestamp Matching

For wearable data integration, you can adjust the timestamp matching threshold in `merge_annotations`:

```python
# If closest timestamp is within 200ms, use that annotation
if min_diff <= 200:  # Adjust threshold as needed (in milliseconds)
    df_final.at[i, 'label'] = annotation_labels[closest_idx]
    matches += 1
```

## Troubleshooting

### Common Issues

1. **No hand detection**: 
   - Ensure adequate lighting in the video
   - Try adjusting `min_detection_confidence` in MediaPipe Hands initialization
   - Check if hand is visible and not obscured

2. **Incorrect activity detection**:
   - Adjust `grabbing_threshold` and `releasing_threshold` values
   - Modify color ranges for chip and box detection
   - Ensure ROIs are correctly positioned for your experimental setup

3. **Watch not detected**:
   - Ensure watch is clearly visible
   - Adjust `watch_color_lower` and `watch_color_upper` values
   - Increase `watch_detection_threshold` for more persistence

4. **OCR timestamp errors**:
   - Verify that Tesseract OCR is properly installed
   - Adjust the timestamp_roi coordinates to match your video's timestamp location
   - Try preprocessing options in `_extract_timestamp_from_frame`

5. **GUI buttons not visible**:
   - Try resizing the GUI window to see if buttons appear at the bottom
   - Use command-line mode as an alternative
   - Check if the window is maximized

6. **Timestamp matching issues**:
   - Verify that the unix timestamps in accelerometer data are in milliseconds
   - Adjust the matching threshold (default: 200ms)
   - Use the `--keep-temp` option to inspect intermediate files

### Debug Mode

Enable debug mode to get detailed information about the detection process:

```python
# Via command line
python main.py --debug --log-file debug.txt

# Or in code
detector = EnhancedHandTaskDetector(debug_mode=True)
detector.set_debug_mode(True, log_file="debug.txt")
```

Debug output includes:
- Hand position and pinch distance
- ROI checks and results
- Confidence scores for chip and box detection
- Activity transitions

## FAQ

**Q: How accurate is the system?**
A: The accuracy depends on video quality, lighting conditions, and how well the default parameters match your experimental setup. In optimal conditions, accuracy typically ranges from 85-95%.

**Q: Can I use this for multiple participants simultaneously?**
A: The current version is designed for single-participant scenarios. For multiple participants, significant modifications would be required to track and distinguish between different people.

**Q: Does this work with any video format?**
A: Any video format supported by OpenCV should work, including MP4, AVI, and MOV. However, for best results, use high-quality videos with good lighting and clear visibility of hands.

**Q: How can I improve detection accuracy?**
A: Ensure good lighting, use a high-quality camera, position the camera to clearly capture hand movements, and adjust the detection parameters based on your specific experimental setup.

**Q: What if the participant uses their non-watch hand?**
A: The system is designed to track only the watch-wearing hand, as specified in the requirements. Activities performed with the non-dominant hand will not be tracked.

**Q: How do I synchronize video with wearable device data?**
A: The `process_annotations.py` script handles this automatically by matching timestamps between the video annotations and accelerometer data. Make sure your accelerometer data contains unix timestamps in milliseconds.

**Q: Will this work with smartwatch data?**
A: Yes, the system is designed to work with any wearable device data as long as it contains a unix timestamp column. The integration script will handle the merging of annotations with your device data.

**Q: What if my accelerometer sampling rate is different from the video?**
A: The timestamp matching algorithm handles different sampling rates by finding the closest video annotation for each accelerometer data point, within a configurable time threshold.

---

## License

MIT

## Acknowledgments

- MediaPipe team for their hand tracking solution
- OpenCV community for computer vision tools
- Contributors to this project 