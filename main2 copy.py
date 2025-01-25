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
capture_interval = .5  # Time interval (in seconds) between captures
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line
duration_minutes = 1.5  # Run for 1 minute
min_confidence = 0.9  # Minimum confidence for OCR

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

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

    # print(f"Detected Team 1 Name: {team1_name}")
    # print(f"Detected Team 2 Name: {team2_name}")
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
    screen = sct.grab(sct.monitors[1])
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2BGR)

    # Crop time area
    time_frame = screen_image[time_roi["y"]:time_roi["y"] + time_roi["height"],
                              time_roi["x"]:time_roi["x"] + time_roi["width"]]

    # Perform OCR
    time_results = reader.readtext(time_frame, detail=0)

    # Extract detected time or default to "00:00"
    detected_time = time_results[0].strip() if time_results else "00:00"
    return detected_time

# Process desktop captures and detect text
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

        # Check if team names are still detectable before each capture
        current_team1_name, current_team2_name = detect_team_names(sct)
        if current_team1_name == "Unknown" or current_team2_name == "Unknown":
            print("One or both team names are not detected. Skipping this frame.")
            time.sleep(capture_interval)
            continue

        # print(f"Detected Numbers: Number 1 = {number1}, Number 2 = {number2}, Round = {round_number}")

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

# Postprocess: Filter lines starting with team names
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

# Call the function to split teams and update the CSV
process_csv_with_team_split(output_csv_filtered, team1_name, team2_name)
print(f"All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")
