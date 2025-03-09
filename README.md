# VCT-Scoreboard-Matrix

## Overview
This project performs real-time OCR (Optical Character Recognition) and object detection on VALORANT esports UI elements. It captures and analyzes the scoreboard, player information, round timers, and weapon icons during matches.

## Features
- **Team Name Detection**: Automatically identifies team names from the scoreboard
- **Score Tracking**: Captures and records the score for both teams
- **Round Timer**: Detects and tracks the round timer
- **Weapon Detection**: Uses template matching to identify specific weapons (e.g., Vandal)
- **Player Kill Feed Analysis**: Processes the kill feed to track player eliminations
- **Automated Screenshot Capture**: Takes screenshots when specific templates are detected
- **Data Export**: Saves all detected information to CSV files for further analysis

## Requirements
- Python 3.x
- OpenCV
- EasyOCR
- NumPy
- MSS (screen capture)
- FuzzyWuzzy (for text similarity matching)

## Installation
To install all required dependencies, run:

```
pip install -r requirements.txt
```

This will install all the necessary packages listed in the requirements.txt file.

## Configuration
The script uses predefined Regions of Interest (ROIs) for different UI elements:
- Team name areas
- Score display areas
- Round timer
- Kill feed area

## Output Files
- `detected_text_all.csv`: Contains all detected text from the screen
- `filtered_text.csv`: Contains filtered text with team and player information
- `detected_screenshots/`: Directory containing screenshots of detected templates

## Usage
The current working version is `main3 copy.py`. It is messy, but its a WIP. Run the script to start capturing and analyzing the VALORANT UI:

```
python "main3 copy.py"
```

The script will run for the configured duration (default: 0.5 minutes) and generate CSV files with the captured data.

## How It Works
1. Detects team names at startup
2. Continuously captures screenshots at defined intervals
3. Performs OCR on specific regions to extract text information
4. Uses template matching to detect weapon icons
5. Processes and filters the data to extract meaningful information
6. Exports the data to CSV files for analysis
7. Removes duplicates and organizes player information by team

## Notes
- The script requires proper configuration of screen coordinates for accurate detection
- Performance depends on the quality of the screen capture and OCR accuracy
- Template images (e.g., "Vandal_icon.png") must be present in the working directory


I am moving this project to OmniParser as of now, and will update the repository once I have a working version.