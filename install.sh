#!/bin/bash
# Installation script for Hand Task Annotator

echo "Installing Hand Task Annotator..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    echo "Error: Python version $PYTHON_VERSION is not supported. Please install Python 3.7 or higher."
    exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv venv || { echo "Error: Failed to create virtual environment"; exit 1; }

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip || { echo "Warning: Failed to upgrade pip"; }

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt || { echo "Error: Failed to install dependencies"; exit 1; }

# Install the package in development mode
echo "Installing Hand Task Annotator..."
pip install -e . || { echo "Error: Failed to install package"; exit 1; }

# Check for additional system dependencies
echo "Checking for additional system dependencies..."
if ! command -v tesseract &>/dev/null; then
    echo "Warning: Tesseract OCR not found. OCR timestamp extraction will be disabled."
    echo "To install Tesseract OCR:"
    echo "  - macOS: brew install tesseract"
    echo "  - Ubuntu: sudo apt-get install tesseract-ocr"
    echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

echo "Installation complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the GUI:"
echo "  python main.py"
echo ""
echo "For command-line usage:"
echo "  python main.py --help"
echo "" 