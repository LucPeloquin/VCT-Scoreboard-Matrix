import cv2
import easyocr
import csv
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import ImageGrab
import re

# Set UTF-8 encoding for console
sys.stdout.reconfigure(encoding='utf-8')

# ROIs for Team areas
team1_roi = {"left": 8, "top": 540, "width": 399, "height": 534}
team2_roi = {"left": 1518, "top": 546, "width": 397, "height": 525}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Parameters
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line

def capture_screenshot():
    print("Capturing screen in 3 seconds...")
    time.sleep(3)
    screenshot = ImageGrab.grab()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def is_valid_player_name(text):
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Ignore "READY"
    if text.upper() == "READY":
        return False
    
    # Check for spaces
    if ' ' in text:
        return False
        
    # Check length (must be 3 or more characters)
    if len(text) < 3:
        return False
        
    # Check if only alphanumeric characters
    if not text.isalnum():
        return False
        
    # Check if there's at least one letter
    if not any(c.isalpha() for c in text):
        return False
        
    # Check if any number is standalone (not adjacent to a letter)
    parts = re.split(r'([0-9]+)', text)
    for i, part in enumerate(parts):
        if part.isdigit():
            # Check if neither previous nor next part contains letters
            prev_has_letter = i > 0 and any(c.isalpha() for c in parts[i-1])
            next_has_letter = i < len(parts)-1 and any(c.isalpha() for c in parts[i+1])
            if not (prev_has_letter or next_has_letter):
                return False
                
    return True

def find_team_players(frame, roi):
    # Crop the frame to the team's ROI
    team_frame = frame[roi["top"]:roi["top"] + roi["height"],
                      roi["left"]:roi["left"] + roi["width"]]
    
    # Perform OCR on the ROI
    results = reader.readtext(team_frame)

    # Group text into horizontal lines
    line_map = defaultdict(list)
    for result in results:
        bbox = result[0]  # Bounding box
        text = result[1]  # Detected text
        
        # Skip invalid player names
        if not is_valid_player_name(text):
            continue

        ymin = min(point[1] for point in bbox)
        ymax = max(point[1] for point in bbox)
        y_center = (ymin + ymax) // 2

        # Assign text to the closest horizontal line
        added_to_line = False
        for line_center in line_map:
            if abs(line_center - y_center) <= line_tolerance:
                line_map[line_center].append((bbox, text))
                added_to_line = True
                break

        if not added_to_line:
            line_map[y_center].append((bbox, text))

    # Process lines to find player names
    players = []
    for _, line in sorted(line_map.items()):
        line = sorted(line, key=lambda item: item[0][0][0])  # Sort by x-coordinate
        for _, text in line:
            if len(players) < 5:  # Only collect up to 5 players
                players.append(text.strip())

    return players[:5]  # Ensure we return exactly 5 or fewer players

def main():
    output_file = 'team_rosters.csv'
    
    print("Starting team roster detection...")
    
    # Capture screen after 3-second delay
    frame = capture_screenshot()
    
    # Find players for each team using their specific ROIs
    team1_players = find_team_players(frame, team1_roi)
    team2_players = find_team_players(frame, team2_roi)

    # Pad player lists to exactly 5 entries
    team1_players.extend([''] * (5 - len(team1_players)))
    team2_players.extend([''] * (5 - len(team2_players)))

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Team', 'Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5'])
        writer.writerow(['Team 1'] + team1_players)
        writer.writerow(['Team 2'] + team2_players)

    print(f"Team rosters saved to {output_file}")

if __name__ == "__main__":
    main() 
    main() 