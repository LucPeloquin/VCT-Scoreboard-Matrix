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

Currently updating logic to work for all weapons.
4. Uses template matching to detect weapon icons
5. Processes and filters the data to extract meaningful information
6. Exports the data to CSV files for analysis
7. Removes duplicates and organizes player information by team

## Notes
- The script requires proper configuration of screen coordinates for accurate detection
- Performance depends on the quality of the screen capture and OCR accuracy
- Template images (e.g., "Vandal_icon.png") must be present in the working directory


I am moving this project to OmniParser as of now, and will update the repository once I have a working version.