import cv2
import easyocr
import csv
import os
import sys
from collections import defaultdict

# Set UTF-8 encoding for console
sys.stdout.reconfigure(encoding='utf-8')

# Static ROI for main text detection
static_roi = {"x": 1275, "y": 89, "width": 645, "height": 248}

# Fixed ROIs for Team 1 and Team 2 detection (first frame only)
team1_roi = {"x": 631, "y": 26, "width": 79, "height": 40}
team2_roi = {"x": 1219, "y": 24, "width": 74, "height": 38}

# Paths
video_path = 'ogvslll.mp4'  # Input video
output_csv_all = 'detected_text_all.csv'  # Output CSV for all detected text
output_csv_filtered = 'filtered_text.csv'  # Output CSV for filtered lines with team names
frame_output_dir = 'frames_output'  # Temporary directory to save frames
os.makedirs(frame_output_dir, exist_ok=True)

# Parameters
frame_skip = 60  # Process every 60th frame
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Detect Team 1 and Team 2 names from the first frame
def detect_team_names(video):
    success, first_frame = video.read()
    if not success:
        print("Failed to read the first frame.")
        exit()

    # Crop Team 1 and Team 2 areas
    team1_frame = first_frame[team1_roi["y"]:team1_roi["y"] + team1_roi["height"],
                              team1_roi["x"]:team1_roi["x"] + team1_roi["width"]]
    team2_frame = first_frame[team2_roi["y"]:team2_roi["y"] + team2_roi["height"],
                              team2_roi["x"]:team2_roi["x"] + team2_roi["width"]]

    # Perform OCR to detect team names
    team1_results = reader.readtext(team1_frame)
    team2_results = reader.readtext(team2_frame)

    team1_name = team1_results[0][1].strip() if team1_results else "Unknown"
    team2_name = team2_results[0][1].strip() if team2_results else "Unknown"

    print(f"Detected Team 1 Name: {team1_name}")
    print(f"Detected Team 2 Name: {team2_name}")
    return team1_name, team2_name

# Process video frames and detect text
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

# Detect team names
team1_name, team2_name = detect_team_names(video)

# Reinitialize video to start processing frames from the beginning
video.release()
video = cv2.VideoCapture(video_path)

all_text_data = []  # Store all detected text
frame_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Crop the static ROI for main text detection
    static_frame = frame[static_roi["y"]:static_roi["y"] + static_roi["height"],
                         static_roi["x"]:static_roi["x"] + static_roi["width"]]

    # Perform OCR on the static ROI
    results_static = reader.readtext(static_frame)

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
        line = sorted(line, key=lambda item: item[0][0][0])  # Sort by X-coordinate
        detected_text_lines.append(" ".join([text[1] for text in line]))

    # Save detected text
    for line in detected_text_lines:
        all_text_data.append([frame_count, line])

    frame_count += 1

# Save all detected text to CSV
with open(output_csv_all, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Detected Text"])  # Header
    writer.writerows(all_text_data)

# Post-process to filter lines containing team names and add a comma after the next word
filtered_text_data = []
seen_lines = set()  # Track unique lines to remove duplicates

for row in all_text_data:
    frame_number, text_line = row
    if team1_name in text_line or team2_name in text_line:
        words = text_line.split()
        new_line = []
        i = 0
        while i < len(words):
            new_line.append(words[i])
            if words[i] == team1_name or words[i] == team2_name:
                if i + 1 < len(words):
                    new_line.append(words[i + 1] + ',')
                    i += 1
            i += 1
        final_text_line = " ".join(new_line).replace('"', '')  # Remove quotation marks
        if final_text_line not in seen_lines:  # Check if the line is unique
            filtered_text_data.append([frame_number, final_text_line])
            seen_lines.add(final_text_line)

# Save filtered text to CSV
with open(output_csv_filtered, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Detected Text"])  # Header
    writer.writerows(filtered_text_data)

print(f"All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")
