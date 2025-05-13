#!/usr/bin/env python3
"""
Test imports for the hand_task_annotator package.

This script attempts to import all necessary dependencies and package modules
to verify that the environment is set up correctly.
"""

import sys

def test_imports():
    """Test importing all required packages and modules"""
    print("Testing imports...")
    
    # Core Python libraries
    print("Importing core Python libraries...")
    try:
        import os
        import sys
        import time
        import json
        import argparse
        import tempfile
        import shutil
        from datetime import datetime
        print("✓ Core Python libraries imported successfully")
    except ImportError as e:
        print(f"✗ Error importing core Python libraries: {str(e)}")
        return False
    
    # Third-party dependencies
    print("\nImporting third-party dependencies...")
    
    # OpenCV
    try:
        import cv2
        print(f"✓ OpenCV version {cv2.__version__} imported successfully")
    except ImportError:
        print("✗ Error importing OpenCV (cv2). Please install: pip install opencv-python")
        return False
    
    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy version {np.__version__} imported successfully")
    except ImportError:
        print("✗ Error importing NumPy. Please install: pip install numpy")
        return False
    
    # MediaPipe
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version {mp.__version__} imported successfully")
    except ImportError:
        print("✗ Error importing MediaPipe. Please install: pip install mediapipe")
        return False
    
    # Pandas
    try:
        import pandas as pd
        print(f"✓ Pandas version {pd.__version__} imported successfully")
    except ImportError:
        print("✗ Error importing Pandas. Please install: pip install pandas")
        return False
    
    # Optional: PyTesseract
    try:
        import pytesseract
        print(f"✓ PyTesseract imported successfully")
    except ImportError:
        print("⚠ PyTesseract not found. OCR timestamp detection will be disabled.")
        print("  To install: pip install pytesseract")
    
    # Package modules
    print("\nImporting hand_task_annotator package modules...")
    try:
        print("Core modules:")
        from hand_task_annotator.core.detector import EnhancedHandTaskDetector
        print("✓ EnhancedHandTaskDetector imported successfully")
        
        from hand_task_annotator.core.annotator import VideoAnnotator
        print("✓ VideoAnnotator imported successfully")
        
        from hand_task_annotator.core.utils import format_timestamp, get_readable_time
        print("✓ Utility functions imported successfully")
        
        print("\nGUI modules:")
        from hand_task_annotator.gui.annotation_gui import AnnotationGUI, run_gui
        print("✓ GUI modules imported successfully")
        
        print("\nTools modules:")
        from hand_task_annotator.tools.demo import run_gui_demo, run_cli_demo
        print("✓ Demo tools imported successfully")
        
        from hand_task_annotator.tools.process_annotations import process_annotations
        print("✓ Annotation processing tools imported successfully")
        
        print("\n✓ All package modules imported successfully")
    except ImportError as e:
        print(f"✗ Error importing package modules: {str(e)}")
        return False
    
    print("\nAll imports successful! The environment is set up correctly.")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)