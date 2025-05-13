import sys
import platform

def test_imports():
    """Test if all required imports are available."""
    print(f"Python version: {sys.version}")
    
    # Test core dependencies
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Failed to import cv2: No module named 'cv2'")
    
    try:
        import mediapipe
        print(f"MediaPipe version: {mediapipe.__version__}")
    except ImportError:
        print("Failed to import mediapipe: No module named 'mediapipe'")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("Failed to import numpy")
    
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("Failed to import pandas")

    # Optional dependencies
    try:
        import pytesseract
        print(f"PyTesseract version: {pytesseract.get_tesseract_version()}")
    except ImportError:
        print("Failed to import pytesseract (optional for timestamp OCR)")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Failed to import torch: No module named 'torch'")
    
    try:
        import torchvision
        print(f"TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("Failed to import torchvision: No module named 'torchvision'")
    
    print("\nAll import tests completed")

if __name__ == "__main__":
    test_imports()