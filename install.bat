@echo off
REM Installation script for Hand Task Annotator on Windows

echo Installing Hand Task Annotator...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.7 or higher.
    exit /b 1
)

REM Check Python version
for /f "tokens=* USEBACKQ" %%F in (`python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`) do (
    set PYTHON_VERSION=%%F
)

echo Using Python %PYTHON_VERSION%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment activation script not found.
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies
    exit /b 1
)

REM Install the package in development mode
echo Installing Hand Task Annotator...
pip install -e .
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install package
    exit /b 1
)

REM Check for additional system dependencies
echo Checking for additional system dependencies...
where tesseract >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Tesseract OCR not found. OCR timestamp extraction will be disabled.
    echo To install Tesseract OCR on Windows:
    echo   - Download from https://github.com/UB-Mannheim/tesseract/wiki
    echo   - Add the installation directory to your PATH
)

echo.
echo Installation complete!
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To start the GUI:
echo   python main.py
echo.
echo For command-line usage:
echo   python main.py --help
echo.

pause 