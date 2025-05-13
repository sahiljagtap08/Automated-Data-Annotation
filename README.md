# Automated Hand Task Annotation System

A computer vision-based system for automatically annotating hand activity in task-based video data. This system uses MediaPipe hand tracking, contour analysis, and confidence-based detection to identify when participants are taking or placing objects during experiments.

## Project Structure

The project has been refactored into a modular structure:

```
hand_task_annotator/
├── core/              # Core detection and annotation components
│   ├── detector.py    # Hand activity detector class
│   ├── annotator.py   # Video annotation processor
│   └── utils.py       # Common utility functions
├── gui/               # GUI components
│   └── annotation_gui.py # Tkinter-based GUI
├── tools/             # Utility tools
│   └── demo.py        # Demo functionality
└── main.py            # Main application entry point
```

## Technical Architecture

This system integrates several computer vision techniques:

1. **MediaPipe Hand Tracking**: Detects hand positions and landmarks
2. **Watch Detection**: Identifies the dominant hand by detecting a watch
3. **Dynamic ROI Tracking**: Adapts to different setups by tracking regions of interest
4. **Confidence-Based Detection**: Reduces false positives through confidence scoring
5. **Shape Analysis**: Distinguishes objects by shape (circularity, aspect ratio)

For more technical details, see [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md).

## Detection Algorithms

The system is designed to detect the following activities:
- **Taking the chip**: When the hand grabs a chip
- **Placing the chip**: When the hand places a chip into a container
- **Taking the box**: When the hand grabs a box
- **Placing the box**: When the hand places a box
- **Off Task**: When no specified activity is detected

## Installation

### Prerequisites
- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy
- Pandas

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hand-task-annotator.git
   cd hand-task-annotator
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### GUI Mode

The easiest way to use the system is through the GUI:

```
python main.py
```

or 

```
python -m hand_task_annotator.main --gui
```

### Command-line Mode

For batch processing or integration into other workflows:

```
python -m hand_task_annotator.main --video path/to/video.mp4 --output path/to/output.csv --rate 25 --debug
```

Options:
- `--video`, `-v`: Path to input video file
- `--output`, `-o`: Path to output CSV file
- `--rate`, `-r`: Sampling rate in Hz (default: 25)
- `--debug`, `-d`: Enable debug mode with detailed logging
- `--log-file`, `-l`: Path to log file (optional)
- `--real-time`, `-rt`: Enable real-time output to CSV file

### Wearable Data Integration

For annotating wearable sensor data using video annotations:

```
python -m hand_task_annotator.tools.process_annotations --video path/to/video.mp4 --accelerometer path/to/accel_data.csv --output path/to/annotated_data.csv
```

This will:
1. Convert Unix timestamps to readable time format
2. Create a working copy of accelerometer data
3. Process the video to generate activity annotations
4. Merge annotations with accelerometer data based on timestamps
5. Save the final annotated dataset

### Keyboard Controls

During video playback:
- `p`: Pause/resume playback
- `f`: Fast-forward 10 seconds
- `s`: Step forward one frame (while paused)
- `+`/`-`: Increase/decrease playback speed
- `q`: Quit

## Output Format

The system generates CSV files with the following columns:
- `unixTimestampInMs`: Unix timestamp in milliseconds
- `readableTime`: Human-readable time (HH:MM:SS.mmm)
- `label`: Detected activity label
- `x`, `y`, `z`: Accelerometer data (if integrated with wearable data)

## Customization

The system can be customized by modifying the following parameters:

### Region of Interest (ROI) Configuration
Modify `detector.py` to adjust the default ROI positions if your experimental setup differs.

### Detection Thresholds
Adjust confidence thresholds in `detector.py` to make detection more or less sensitive.

### Color Ranges
Update the color detection ranges in `detector.py` to match your experimental objects:

```python
self.chip_colors = {
    'red': ([0, 120, 120], [10, 255, 255]),  # HSV color range
    'green': ([45, 100, 100], [75, 255, 255]),
    'blue': ([100, 120, 100], [130, 255, 255]),
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 