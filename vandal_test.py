import cv2
import easyocr
import csv
import sys
import time
import re
from mss import mss
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz

# Set UTF-8 encoding for console
sys.stdout.reconfigure(encoding='utf-8')

# Static ROI for main text detection
static_roi = {"x": 1275, "y": 89, "width": 645, "height": 248}

# ROIs for Team 1, Team 2, and the two number detection areas
team1_roi = {"x": 631, "y": 26, "width": 79, "height": 40}
team2_roi = {"x": 1216, "y": 26, "width": 79, "height": 40}
number1_roi = {"x": 791, "y": 1, "width": 886 - 791, "height": 66 - 1}
number2_roi = {"x": 1030, "y": 3, "width": 1121 - 1030, "height": 69 - 3}
time_roi = {"x": 909, "y": 19, "width": 1009 - 909, "height": 74 - 19}

# Paths
output_csv_all = 'detected_text_all.csv'  # Output CSV for all detected text
output_csv_filtered = 'filtered_text.csv'  # Output CSV for filtered lines with team names

# Parameters
capture_interval = 0.5  # Time interval (in seconds) between captures
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line
duration_minutes = 1.5  # Run for 1.5 minutes
min_confidence = 0.9  # Minimum confidence for OCR

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Initialize time tracking
last_detected_time = None
time_countdown = 40

# Function to detect an image within the static ROI
def detect_image_in_roi(sct, template_path, threshold=0.8):
    """
    Detect if a template image exists within the static ROI.
    :param sct: MSS screenshot object
    :param template_path: Path to the template image
    :param threshold: Confidence threshold for detection (default: 0.8)
    :return: Tuple of (found, location, confidence) where:
             - found: True if the template is detected, False otherwise
             - location: (x, y) coordinates of the detected template
             - confidence: Confidence score of the detection
    """
    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Template image not found at {template_path}")
        return False, None, 0

    # Capture the screen within the static ROI
    screen = sct.grab(static_roi)
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check if the confidence exceeds the threshold
    if max_val >= threshold:
        return True, max_loc, max_val
    else:
        return False, None, max_val

# Detect Team 1 and Team 2 names from the first screenshot
def detect_team_names(sct):
    screen = sct.grab(sct.monitors[1])  # Use the first monitor
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

    # Crop Team 1 and Team 2 areas
    team1_frame = screen_image[team1_roi["y"]:team1_roi["y"] + team1_roi["height"],
                               team1_roi["x"]:team1_roi["x"] + team1_roi["width"]]
    team2_frame = screen_image[team2_roi["y"]:team2_roi["y"] + team2_roi["height"],
                               team2_roi["x"]:team2_roi["x"] + team2_roi["width"]]

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
    number1_frame = screen_image[number1_roi["y"]:number1_roi["y"] + number1_roi["height"],
                                 number1_roi["x"]:number1_roi["x"] + number1_roi["width"]]
    number2_frame = screen_image[number2_roi["y"]:number2_roi["y"] + number2_roi["height"],
                                 number2_roi["x"]:number2_roi["x"] + number2_roi["width"]]

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
    time_frame = screen_image[time_roi["y"]:time_roi["y"] + time_roi["height"],
                              time_roi["x"]:time_roi["x"] + time_roi["width"]]

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

# Function to split team names into Player and Player 2 columns
def process_csv_with_team_split(csv_path, team1, team2):
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Update header to include the "Player 2" column
    if rows:
        header = rows[0][:3] + ["Player 2"]  # Include "Player 2" in the header
        updated_rows = [header]

        # Process each row
        for row in rows[1:]:  # Skip the header row
            round_number, detected_time, detected_text = row

            # Ensure the detected_time is in the format X:XX
            detected_time = re.sub(r'[^\d]', '', detected_time)  # Remove non-numeric characters
            if len(detected_time) == 4:  # Format as XX:XX
                detected_time = f"{detected_time[:2]}:{detected_time[2:]}"
            elif len(detected_time) == 3:  # Format as X:XX
                detected_time = f"{detected_time[0]}:{detected_time[1:]}"
            elif len(detected_time) == 2:  # Invalid case, set as default
                detected_time = f"0:{detected_time}"

            words = detected_text.split()
            player1, player2 = "", ""

            i = 0
            while i < len(words):
                if words[i] in (team1, team2):  # Detect team name
                    if not player1:
                        player1 = words[i]
                        if i + 1 < len(words):
                            player1 += f" {words[i + 1]},"  # Add a comma after the next word
                            i += 1
                    elif not player2:
                        player2 = words[i]
                        if i + 1 < len(words):
                            player2 += f" {words[i + 1]},"  # Add a comma after the next word
                            i += 1
                i += 1

            # Add processed rows with no extra commas
            updated_rows.append([round_number, detected_time, player1.strip(","), player2.strip(",")])

    # Write back to the same CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

# Remove duplicates based on Round, Player, and Player 2
def remove_duplicates(csv_path):
    seen = set()
    unique_rows = []

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        unique_rows.append(header)  # Add the header to the unique rows

        for row in reader:
            round_number, detected_time, player, player2 = row

            # Create a unique key based on Round, Player, and Player 2
            unique_key = (round_number, player, player2)

            # Check if a similar key already exists in the seen set
            is_duplicate = False
            for existing_key in seen:
                existing_round, existing_player, existing_player2 = existing_key

                # Check if the round matches
                if round_number != existing_round:
                    continue

                # Check if Player (Team 1) is similar (within 2 characters difference)
                player_similarity = fuzz.ratio(player, existing_player)

                # Check if Player 2 (Team 2) is similar (within 3 characters difference)
                player2_similarity = fuzz.ratio(player2, existing_player2)

                # Apply different thresholds for Player and Player 2
                if player_similarity >= 98 and player2_similarity >= 97:  # 98% for Player, 97% for Player 2
                    is_duplicate = True
                    break

            # If the key is not a duplicate, add it to unique_rows and seen
            if not is_duplicate:
                seen.add(unique_key)
                unique_rows.append(row)

    # Write the unique rows back to the CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(unique_rows)

    print(f"Removed duplicates from {csv_path} with leniency for 2 characters difference in Player and 3 in Player 2.")

# Main processing loop
all_text_data = []
frame_count = 0
end_time = datetime.now() + timedelta(minutes=duration_minutes)

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

        # Detect image in the static ROI
        template_path = "Vandal_icon.jpeg"  # Path to the template image
        found, location, confidence = detect_image_in_roi(sct, template_path)
        if found:
            print(f"Template detected at {location} with confidence {confidence:.2f}")

        # Define the static ROI for main text detection
        roi = {"left": static_roi["x"], "top": static_roi["y"],
               "width": static_roi["width"], "height": static_roi["height"]}
        screen = sct.grab(roi)
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results_static = reader.readtext(frame)

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
            line = sorted(line, key=lambda item: item[0][0][0])
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
    writer.writerow(["Round", "Time", "Player"])  # Header
    writer.writerows(filtered_text_data)

# Call the function to split teams and update the CSV
process_csv_with_team_split(output_csv_filtered, team1_name, team2_name)

# Remove duplicates based on Round, Player, and Player 2
remove_duplicates(output_csv_filtered)

print(f"All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")