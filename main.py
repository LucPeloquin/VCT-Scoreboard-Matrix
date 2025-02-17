import cv2
import os
import easyocr
import csv
import sys
import time
import re
import pandas as pd
from mss import mss
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

# Set UTF-8 encoding for console
sys.stdout.reconfigure(encoding='utf-8')

# Static ROI for main text detection
static_roi = {"left": 1275, "top": 89, "width": 645, "height": 248}

# ROIs for Team 1, Team 2, and the two number detection areas
team1_roi = {"left": 631, "top": 26, "width": 79, "height": 40}
team2_roi = {"left": 1216, "top": 26, "width": 79, "height": 40}
number1_roi = {"left": 791, "top": 1, "width": 886 - 791, "height": 66 - 1}
number2_roi = {"left": 1030, "top": 3, "width": 1121 - 1030, "height": 69 - 3}
time_roi = {"left": 909, "top": 19, "width": 1009 - 909, "height": 74 - 19}

# Paths
output_csv_all = 'detected_text_all.csv'  # Output CSV for all detected text
output_csv_filtered = 'filtered_text.csv'  # Output CSV for filtered lines with team names

# Parameters
capture_interval = 0.05  # Time interval (in seconds) between captures
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line
duration_minutes = 2  # Run for 1.5 minutes
min_confidence = 0.9  # Minimum confidence for OCR

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Get list of icon templates from icons folder
icon_templates = [os.path.join("icons", f) for f in os.listdir("icons") if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Loaded {len(icon_templates)} icon templates: {icon_templates}")

# Initialize time tracking
last_detected_time = None
time_countdown = 40

# Add a dictionary to store detections per round
detections_dict = {}

# Function to detect an image within the static ROI
def detect_image_in_roi(sct, template_paths, round_number, threshold=0.7):
    """
    Detect if any template images exist within the static ROI at any scale.
    """
    detections = []
    
    # Only capture the static ROI
    screen = sct.grab(static_roi)
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2GRAY)

    for template_path in template_paths:
        if not template_path.startswith("icons/"):
            continue
            
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            template_name = os.path.basename(template_path)
            detections.append((template_name, max_loc, max_val))
            # Store the detection for this round
            detections_dict[round_number] = template_name

    return detections

# Detect Team 1 and Team 2 names from the first screenshot
def detect_team_names(sct):
    screen = sct.grab(sct.monitors[1])  # Use the first monitor
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

    # Crop Team 1 and Team 2 areas
    team1_frame = screen_image[team1_roi["top"]:team1_roi["top"] + team1_roi["height"],
                               team1_roi["left"]:team1_roi["left"] + team1_roi["width"]]
    team2_frame = screen_image[team2_roi["top"]:team2_roi["top"] + team2_roi["height"],
                               team2_roi["left"]:team2_roi["left"] + team2_roi["width"]]

    # Perform OCR
    team1_results = reader.readtext(team1_frame, detail=0)
    team2_results = reader.readtext(team2_frame, detail=0)

    team1_name = team1_results[0].strip() if team1_results else "Unknown"
    team2_name = team2_results[0].strip() if team2_results else "Unknown"

    return team1_name, team2_name

# Detect numbers from the two number ROIs
def detect_numbers(sct):
    screen = sct.grab(sct.monitors[1])
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

    # Crop number areas
    number1_frame = screen_image[number1_roi["top"]:number1_roi["top"] + number1_roi["height"],
                                 number1_roi["left"]:number1_roi["left"] + number1_roi["width"]]
    number2_frame = screen_image[number2_roi["top"]:number2_roi["top"] + number2_roi["height"],
                                 number2_roi["left"]:number2_roi["left"] + number2_roi["width"]]

    # Perform OCR
    number1_results = reader.readtext(number1_frame, detail=0)
    number2_results = reader.readtext(number2_frame, detail=0)

    # Extract numbers
    try:
        number1 = int(number1_results[0].strip()) if number1_results else 0
        number2 = int(number2_results[0].strip()) if number2_results else 0
    except ValueError:
        number1, number2 = 0, 0

    round_number = number1 + number2 + 1
    return number1, number2, round_number

# Detect time from the time ROI
def detect_time(sct):
    global last_detected_time, time_countdown

    screen = sct.grab(sct.monitors[1])
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

    # Crop time area
    time_frame = screen_image[time_roi["top"]:time_roi["top"] + time_roi["height"],
                              time_roi["left"]:time_roi["left"] + time_roi["width"]]

    # Perform OCR
    time_results = reader.readtext(time_frame, detail=0)

    # Extract detected time or use countdown logic
    if time_results:
        detected_time = time_results[0].strip()
        last_detected_time = detected_time  # Update last detected time
        return detected_time
    elif last_detected_time:
        # If no time is detected, use the countdown
        minutes, seconds = divmod(time_countdown, 60)
        countdown_time = f"{minutes}:{seconds:02d}"
        time_countdown = max(0, time_countdown - int(capture_interval))  # Decrease countdown
        return countdown_time
    return "00:00"  # Default if no time and no last detected time

def process_csv_with_team_split(csv_path, team1, team2):
    # Load the roster database with full "Team Player" combinations
    valid_rosters = load_team_rosters()
    team1_combinations = valid_rosters.get(team1.lower(), [])
    team2_combinations = valid_rosters.get(team2.lower(), [])

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        # Add new column to header
        header = rows[0][:3] + ["Player 2", "Template"]
        updated_rows = [header]

        for row in rows[1:]:
            round_number, detected_time, detected_text = row

            # Format detected_time
            detected_time = re.sub(r'[^\d]', '', detected_time)
            if len(detected_time) == 4:
                detected_time = f"{detected_time[:2]}:{detected_time[2:]}"
            elif len(detected_time) == 3:
                detected_time = f"{detected_time[0]}:{detected_time[1:]}"
            elif len(detected_time) == 2:
                detected_time = f"0:{detected_time}"

            words = detected_text.split()
            player1, player2 = "", ""

            # Look for team names and following words
            for i in range(len(words) - 1):
                if words[i] == team1:
                    if not player1:
                        player1 = f"{team1} {words[i+1]}"
                    elif not player2:
                        player2 = f"{team1} {words[i+1]}"
                elif words[i] == team2:
                    if not player1:
                        player1 = f"{team2} {words[i+1]}"
                    elif not player2:
                        player2 = f"{team2} {words[i+1]}"

            # Get template name from detections dictionary (assuming it's available in scope)
            template_name = ""
            if round_number in detections_dict:
                template_name = detections_dict[round_number]

            if player1 or player2:  # Only add rows where players were found
                updated_rows.append([round_number, detected_time, player1, player2, template_name])

    # Write back to the same CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

# Remove duplicates based on Round, Player, and Player 2
def remove_duplicates(csv_path):
    seen = {}
    unique_rows = []

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        unique_rows.append(header)  # Add the header to the unique rows

        for row in reader:
            round_number, detected_time, player1, player2 = row
            
            # Create a case-insensitive key combining round and both players
            # Sort players to ensure "A kills B" and "B kills A" are treated as duplicates
            players = sorted([player1.lower(), player2.lower()])
            key = (round_number, tuple(players))

            # If we haven't seen this combination or if this entry has an earlier time
            if key not in seen or detected_time < seen[key]["time"]:
                seen[key] = {
                    "time": detected_time,
                    "row": row
                }

    # Collect all unique rows, keeping only the earliest occurrence
    unique_rows.extend(seen[key]["row"] for key in seen)

    # Sort by round number and time
    data_rows = unique_rows[1:]  # Exclude header
    data_rows.sort(key=lambda x: (int(x[0]), x[1]))  # Sort by round number, then time
    unique_rows = [header] + data_rows

    # Write the unique rows back to the CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(unique_rows)

    print(f"Removed duplicates from {csv_path} by keeping earliest entries per round and player combination.")

# Main processing loop
all_text_data = []
frame_count = 0
end_time = datetime.now() + timedelta(minutes=duration_minutes)

if os.path.exists('detected_screenshots'):
    for file in os.listdir('detected_screenshots'):
        os.remove(os.path.join('detected_screenshots', file))
else:
    os.makedirs('detected_screenshots')

with mss() as sct:
    team1_name, team2_name = detect_team_names(sct)

    # Ensure both team names are detected before starting OCR processing
    if team1_name == "Unknown" or team2_name == "Unknown":
        print("Team names not detected. Aborting OCR processing.")
        sys.exit(1)

    print(f"Starting OCR processing with Team1: {team1_name} and Team2: {team2_name}")

    while datetime.now() < end_time:
        # Detect numbers from the number ROIs
        number1, number2, round_number = detect_numbers(sct)

        # Detect time from the time ROI
        detected_time = detect_time(sct)

        # Detect images in the static ROI
        detections = detect_image_in_roi(sct, icon_templates, round_number)
        if detections:
            for template_name, location, confidence in detections:
                print(f"Template {template_name} detected in static ROI at {location} with confidence {confidence:.2f}")
                detected_frame = sct.grab(static_roi)
                detected_image = np.array(detected_frame)
                screenshot_path = os.path.join('detected_screenshots', f"detected_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(screenshot_path, detected_image)

        # Define the static ROI for main text detection
        roi = {"left": static_roi["left"], "top": static_roi["top"],
               "width": static_roi["width"], "height": static_roi["height"]}
        screen = sct.grab(roi)
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results_static = reader.readtext(frame)

        # Output detected text within the static ROI to the console
        # if results_static:
        #     for _, text, _ in results_static:
        #         print(f"Detected text in static ROI: {text}")

        # Group text into horizontal lines
        line_map = defaultdict(list)
        for result in results_static:
            bbox = result[0]  # Bounding box
            text = result[1]  # Detected text
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

        # Sort lines top to bottom and text within each line left to right
        detected_text_lines = []
        for _, line in sorted(line_map.items()):
            line = sorted(line, key=lambda item: item[0][0])
            detected_text_lines.append(" ".join([text[1] for text in line]))

        for line in detected_text_lines:
            all_text_data.append([frame_count, number1, number2, round_number, detected_time, line])

        frame_count += 1
        time.sleep(capture_interval)

# Save all detected text to CSV
with open(output_csv_all, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Number 1", "Number 2", "Round", "Time", "Player 1"])
    writer.writerows(all_text_data)

# Post-process and save filtered text
filtered_text_data = []
seen_lines = set()

for row in all_text_data:
    _, _, _, round_number, detected_time, text_line = row
    words = text_line.split()
    if words and words[0] in (team1_name, team2_name):  # Check if the line starts with a team name
        if text_line not in seen_lines:
            filtered_text_data.append([round_number, detected_time, text_line])
            seen_lines.add(text_line)

# Save filtered text to CSV
with open(output_csv_filtered, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "Time", "Player 1"])  # Header
    writer.writerows(filtered_text_data)

# Call the function to split teams and update the CSV
process_csv_with_team_split(output_csv_filtered, team1_name, team2_name)

# Remove duplicates based on Round, Player, and Player 2
remove_duplicates(output_csv_filtered)

def convert_time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

df = pd.read_csv(output_csv_filtered)
df['TimeSeconds'] = df['Time'].apply(convert_time_to_seconds)
df = df.sort_values(['Round', 'TimeSeconds'], ascending=[True, False])
df = df.drop('TimeSeconds', axis=1)

# Save the sorted DataFrame back to CSV
df.to_csv(output_csv_filtered, index=False)

print(f"All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")

def validate_and_correct_against_roster(csv_path):
    # First, create a copy of the original file
    pre_validation_path = csv_path.replace('.csv', '_pre_validation.csv')
    df = pd.read_csv(csv_path)
    df.to_csv(pre_validation_path, index=False)
    
    # Load team rosters
    rosters = {}
    valid_combinations = []
    with open('team_rosters.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            team = row[0]
            # Store the original case combinations
            players = [f"{team} {p.strip()}" for p in row[1:6] if p.strip()]
            valid_combinations.extend(players)
            # Store lowercase for matching
            rosters[team] = [p.lower() for p in players]
    
    def find_closest_match(player_name):
        if pd.isna(player_name) or player_name == '':
            return player_name
            
        player_lower = player_name.lower()
        best_match = None
        min_distance = float('inf')
        
        # Check against all valid combinations
        for valid_name in valid_combinations:
            valid_lower = valid_name.lower()
            # Calculate Levenshtein distance
            distance = sum(1 for a, b in zip(player_lower, valid_lower) if a != b)
            distance += abs(len(player_lower) - len(valid_lower))
            
            # Update if this is the best match within margin of error
            if distance <= 3 and distance < min_distance:
                min_distance = distance
                best_match = valid_name
        
        return best_match if best_match else player_name

    # Correct player names
    df['Player 1'] = df['Player 1'].apply(find_closest_match)
    df['Player 2'] = df['Player 2'].apply(find_closest_match)
    
    # Save corrected data to a new file
    post_validation_path = csv_path.replace('.csv', '_validated.csv')
    df.to_csv(post_validation_path, index=False)
    
    # Also update the original file
    df.to_csv(csv_path, index=False)
    
    print(f"Pre-validation data saved to: {pre_validation_path}")
    print(f"Post-validation data saved to: {post_validation_path}")
    print(f"Original file updated: {csv_path}")

# Add this line after other post-processing steps
validate_and_correct_against_roster(output_csv_filtered)

def load_team_rosters():
    """Load team rosters from CSV file and return a dictionary of team-player combinations."""
    rosters = {}
    with open('team_rosters.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Check if row is not empty
                team = row[0].lower()
                # Create full "Team Player" combinations for each valid player
                players = [f"{row[0]} {p.strip()}" for p in row[1:6] if p.strip()]
                rosters[team] = players
    return rosters

