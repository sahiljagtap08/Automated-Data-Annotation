# Technical Overview - Automated Hand Task Annotation System

This document provides a technical deep-dive into the hand activity annotation system, explaining the algorithms, processing pipeline, and detection methodology.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Video Input    │────▶│  Frame Sampler  │────▶│  Hand Detector  │
│                 │     │                 │     │   (MediaPipe)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  CSV Generator  │◀────│ Activity Tracker│◀────│ Watch Detector  │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │  ROI Detector   │────▶│ Object Detector │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

### Component Responsibilities

1. **Frame Sampler**: Extracts frames at the specified sampling rate (default: 25Hz)
2. **Hand Detector**: Uses MediaPipe to identify hand landmarks
3. **Watch Detector**: Identifies the watch-wearing hand (dominant hand)
4. **ROI Detector**: Dynamically identifies regions of interest (chip source, chip destination, etc.)
5. **Object Detector**: Detects chips and boxes using contour and shape analysis
6. **Activity Tracker**: Classifies activities based on hand position and object detection
7. **CSV Generator**: Creates timestamped annotations in CSV format

## Detection Pipeline

### 1. Hand Landmark Detection

The system uses MediaPipe Hands to detect 21 hand landmarks:

```
    8   12  16  20
    |   |   |   |
    7   11  15  19
    |   |   |   |
    6   10  14  18
    |   |   |   |
    5---9---13--17
    |
    |
    4
   /|\
  3 2 1
     \
      0
```

Key landmarks used:
- **0**: Wrist (hand center position)
- **4**: Thumb tip
- **8**: Index finger tip
- **5-9**: For grabbing gesture detection

### 2. Watch Detection

The system identifies a participant's watch-wearing (dominant) hand by analyzing the wrist area:

```
┌───────────────────────────┐
│                           │
│          Video            │
│          Frame            │
│                           │
│                           │
│      ┌─────────────┐      │
│      │ Wrist Area  │      │
│      └─────────────┘      │
│                           │
└───────────────────────────┘
```

Process:
1. Extract region around wrist landmark
2. Convert to HSV color space
3. Apply color thresholding for dark colors (black watch band)
4. Calculate percentage of dark pixels
5. If sufficient dark pixels are found consistently, mark as watch hand

### 3. Dynamic ROI Detection

The system automatically identifies four key regions:

```
┌─────────────────────────────────────────────┐
│                                             │
│  ┌────────────────┐      ┌────────────────┐ │
│  │                │      │                │ │
│  │  Box Source    │      │  Box Dest      │ │
│  │                │      │                │ │
│  └────────────────┘      └────────────────┘ │
│                                             │
│               ┌────────────────┐            │
│               │                │            │
│               │  Chip Dest     │            │
│               │  (box)         │            │
│               │                │            │
│               └────────────────┘            │
│                                             │
│  ┌────────────────────────────────────────┐ │
│  │                                        │ │
│  │  Chip Source                           │ │
│  │                                        │ │
│  └────────────────────────────────────────┘ │
│                                             │
└─────────────────────────────────────────────┘
```

Detection method:
1. Chip source region is fixed at bottom of frame
2. Box source region is typically in upper right
3. Chip destination (box area) is detected using contour analysis:
   - Apply color thresholding for box colors
   - Find large contours
   - Identify rectangular shapes
   - Update ROI based on box position
4. Box destination is defined as area near detected box

### 4. Object Detection

#### Chip Detection

Poker chips are identified through shape and color analysis:

```
              ┌────────────────┐
              │ Frame Region   │
              │                │
              │    ┌─────┐     │
              │    │     │     │
              │    │     │     │
              │    └─────┘     │
              │                │
              └────────────────┘
                      │
                      ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ HSV        │  │ Contour    │  │ Shape      │
│ Thresholding│─▶│ Detection │─▶│ Analysis   │
└────────────┘  └────────────┘  └────────────┘
                                      │
                                      ▼
                               ┌────────────┐
                               │ Confidence │
                               │ Scoring    │
                               └────────────┘
```

Process:
1. Convert region to HSV color space
2. Apply color thresholds for red, green, blue chips
3. Clean up mask with morphological operations
4. Find contours in the mask
5. For each contour:
   - Calculate area and perimeter
   - Compute circularity = 4π × area / perimeter²
   - If circularity > 0.5, likely a chip (circles have circularity of 1.0)
6. Update chip holding confidence score based on detection history

#### Box Detection

Boxes are identified using similar techniques but with different shape criteria:

```
              ┌────────────────┐
              │ Frame Region   │
              │                │
              │   ┌────────┐   │
              │   │        │   │
              │   │        │   │
              │   └────────┘   │
              │                │
              └────────────────┘
                      │
                      ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ HSV        │  │ Polygon    │  │ Aspect     │
│ Thresholding│─▶│ Approx.   │─▶│ Ratio Check│
└────────────┘  └────────────┘  └────────────┘
                                      │
                                      ▼
                               ┌────────────┐
                               │ Size       │
                               │ Filtering  │
                               └────────────┘
```

Process:
1. Convert region to HSV color space
2. Apply color thresholds
3. Find contours in the mask
4. For each contour:
   - Approximate polygon using Douglas-Peucker algorithm
   - Check if polygon has 4-8 sides (rectangular shapes)
   - Calculate aspect ratio (width/height)
   - Accept if aspect ratio is between 0.5 and 2.0 (rectangular)
   - Filter by size (boxes are larger than chips)
5. Update box holding confidence score

### 5. Activity Classification

The system determines activities based on multiple factors:

```
┌───────────────┐
│ Hand Position │
│ (in ROI?)     │
└───────┬───────┘
        │
        ▼
┌───────────────┐    ┌───────────────┐
│ Object        │    │ Gesture       │
│ Detection     │◀───┤ (pinch        │
│ (holding?)    │    │  distance)    │
└───────┬───────┘    └───────────────┘
        │
        ▼
┌───────────────┐
│ Temporal      │
│ Consistency   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Activity      │
│ Label         │
└───────────────┘
```

Decision process:
1. Check if hand is in any ROI
   - If in chip source: Potential "Taking the chip"
   - If in chip dest: Potential "Placing the chip"
   - If in box source: Potential "Taking the box"
   - If in box dest: Potential "Placing the box"
2. Check pinch distance
   - Small distance (<0.25): Grabbing gesture
   - Large distance (>0.35): Releasing gesture
3. Verify object detection
   - If chip detected with high confidence, confirm chip holding
   - If box detected with high confidence, confirm box holding
4. Apply state transition rules
   - Must be holding chip to place chip
   - Must release chip when placing
5. Apply temporal consistency
   - Brief object disappearance shouldn't change state
   - Gradual confidence decay prevents flickering
6. Default to "Off Task" when no activity detected for several frames

## Confidence Scoring System

The system uses a confidence-based approach to improve robustness:

```
          Detection          No Detection
              │                   │
              ▼                   ▼
     ┌────────────────┐  ┌────────────────┐
     │ Confidence += n│  │ Confidence -= m│
     └────────────────┘  └────────────────┘
              │                   │
              ▼                   ▼
     ┌────────────────┐  ┌────────────────┐
     │ Cap at 1.0     │  │ Floor at 0.0   │
     └────────────────┘  └────────────────┘
              │                   │
              └────────┬──────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Compare to     │
              │ Threshold      │
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Binary         │
              │ Decision       │
              └────────────────┘
```

Benefits:
1. **Reduced false positives**: Requires consistent detection to reach high confidence
2. **Reduced false negatives**: Temporary loss of detection doesn't immediately change state
3. **Hysteresis effect**: Different thresholds for start/stop of activities prevents oscillation
4. **Decay memory**: Confidence decreases gradually rather than binary state changes

## Visualization System

The system provides real-time visual feedback:

```
┌────────────────────────────────────────────────┐
│ Hand Task Video Annotator                  [X] │
├────────────────────────────────────────────────┤
│                                                │
│ [Current Activity: Taking the chip]            │
│                                                │
│ ┌─────────────────────────────────────────────┐│
││                                              ││
││                                              ││
││                 Video Display                ││
││                                              ││
││                                              ││
│└─────────────────────────────────────────────┘│
│                                                │
│ Holding chip: True (conf: 0.85)                │
│ Holding box: False (conf: 0.12)                │
│ Watch detected: True                           │
│                                                │
│ Controls: 'q'-quit | 'f'-fast forward |        │
│           'p'-pause | '+'/'-'-speed            │
└────────────────────────────────────────────────┘
```

Features:
1. Color-coded activity labels (red/green/orange)
2. Visualization of confidence scores
3. Real-time ROI highlighting
4. Hand trajectory tracking
5. Object contour visualization
6. Control instructions

## Key Algorithms

### Pinch Distance Calculation

```python
def _calculate_pinch_distance(self, thumb_tip, index_tip):
    """Calculate normalized distance between thumb and index finger"""
    return np.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                   (thumb_tip[1] - index_tip[1])**2) / 100
```

### Circularity Calculation

```python
# For chip detection
circularity = 4 * np.pi * area / (perimeter * perimeter)
if circularity > 0.5:  # Value close to 1 indicates circle
    chip_detected = True
```

### Confidence Update Logic

```python
# When chip detected
self.chip_holding_confidence = min(1.0, self.chip_holding_confidence + 0.2)

# When not detected
self.chip_holding_confidence = max(0.0, self.chip_holding_confidence - 0.1)

# Decision based on threshold
actually_holding_chip = self.chip_holding_confidence > 0.6
```

### Watch Detection

```python
mask = cv2.inRange(hsv, self.watch_color_lower, self.watch_color_upper)
watch_pixels = np.sum(mask) / 255
dark_percentage = watch_pixels / total_pixels
return dark_percentage > 0.15  # Threshold for watch detection
```

## Performance Considerations

The system balances accuracy with real-time performance:

1. **Sampling rate**: Default 25Hz provides good balance of detail and performance
2. **Batch processing**: Annotations are processed in batches for efficiency
3. **ROI updates**: ROIs updated every 2 seconds instead of every frame
4. **Visualization delay**: Configurable frame delay for slower display without affecting processing
5. **Fast-forward**: Option to skip inactive segments
6. **Dynamic color thresholding**: Balances detection accuracy with processing speed

## Ground Truth Comparison

For validation, the system's annotations can be compared to manually labeled ground truth:

```
Time    Ground Truth    System Output    Match?
00:15   Taking chip     Taking chip      ✓
00:18   Taking chip     Taking chip      ✓
00:23   Placing chip    Placing chip     ✓
00:30   Off Task        Off Task         ✓
00:35   Taking box      Off Task         ✗
00:40   Taking box      Taking box       ✓
```

Typical error patterns:
- Detection lag at activity transitions
- Occasional false negatives during quick movements
- Confusion between similar-colored objects
- Missed detections during hand occlusion

## Future Improvements

Potential enhancements to the system:

1. **Deep learning object detection**: Replace color/contour analysis with trained models
2. **Hand pose estimation**: Better gesture recognition beyond pinch detection
3. **Temporal modeling**: Use sequence models (LSTM/HMM) for activity transitions
4. **Multi-hand tracking**: Support for participants using both hands
5. **3D hand tracking**: Improve depth perception for occlusion handling
6. **Auto-parameter tuning**: Self-adjust thresholds based on video conditions

---

## References

1. MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
2. OpenCV Contour Analysis: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
3. HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV 