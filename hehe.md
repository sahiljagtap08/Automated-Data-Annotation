# Automated Hand Task Annotation System

A computer vision system for automatically annotating task-based video data with hand activity labels.

## Features

- **Watch-Wearing Hand Detection**: Tracks only the dominant hand wearing a watch
- **Dynamic ROI Tracking**: Automatically adjusts regions of interest if objects move
- **Real-Time Annotation**: Outputs annotations while processing, no need to wait for completion
- **Fast-Forward Capability**: Skip inactive portions of video during processing
- **OCR Timestamp Sync**: Reads timestamps directly from video frames (optional)
- **Activity Recognition**: Detects and labels activities:
  - Taking the chip
  - Placing the chip
  - Taking the box
  - Placing the box
  - Off Task

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
4. Install system dependencies (for OCR timestamp detection):
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Basic usage:
```
python main.py --video P19.mp4 --output P19_annotations.csv
```

Available command-line options:
```
--video FILE       Path to input video file (default: P19.mp4)
--output FILE      Path to output CSV file (default: P19_annotations.csv)
--rate RATE        Sampling rate in Hz (default: 25)
--real-time        Enable real-time output to CSV
--test-roi         Test ROI configuration on first frame
```

### Keyboard Controls During Processing

- **Q**: Quit processing
- **F**: Fast-forward 10 seconds (useful to skip inactive segments)

## Output Format

The system generates a CSV file with the following columns:

- `unixTimestampInMs`: Timestamp in milliseconds
- `readableTime`: Human-readable timestamp (HH:MM:SS)
- `label`: Activity label (one of the five recognized activities)
- `x`, `y`, `z`: Dummy accelerometer data (for compatibility)

## Customization

- Adjust detection thresholds in `EnhancedHandTaskDetector.__init__`
- Modify ROI detection in `update_dynamic_rois` method
- Change activity detection logic in `_determine_activity` method

## Troubleshooting

If you encounter issues:

1. Run `python test_imports.py` to verify all dependencies are installed
2. Check if your video file is in a supported format (try converting to MP4)
3. Ensure your Python environment meets the requirements (Python 3.8+)
4. For OCR issues, verify Tesseract is installed correctly

## License

MIT 