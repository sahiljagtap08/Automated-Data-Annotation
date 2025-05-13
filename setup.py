#!/usr/bin/env python3
"""
Setup script for hand_task_annotator package
"""

from setuptools import setup, find_packages

setup(
    name="hand_task_annotator",
    version="1.0.0",
    description="Automated annotation system for hand activities in video data",
    author="Dr. Motti Research Group",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.8"],
        "gui": ["tk"],
    },
    entry_points={
        "console_scripts": [
            "hand-task-annotator=hand_task_annotator.main:main",
            "hand-task-demo=hand_task_annotator.tools.demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
) 