import cv2
import os
import csv
import shutil
from google.cloud import vision
from google.oauth2 import service_account
from collections import defaultdict
import sys
import re  # For removing non-Unicode characters
from tkinter import Tk, Label, Button, BooleanVar, Checkbutton, Entry, StringVar

# Set default encoding for the console to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Set up Google Vision API credentials
credentials = service_account.Credentials.from_service_account_file(r"G:\valorant-vision\valorant-vod-scraper-8267e96f46ba.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

print("Google Vision API client initialized successfully!")

# Paths
video_path = 'ogvslll.mp4'  # Input video
output_csv_all = 'detected_text_all.csv'  # Output CSV for all detected text
output_csv_filtered = 'filtered_text.csv'  # Output CSV for filtered text
frame_output_dir = 'frames_output'  # Temporary directory to save cropped frames
os.makedirs(frame_output_dir, exist_ok=True)

# Parameters
frame_skip = 60  # Process every 60th frame
line_tolerance = 15  # Tolerance for grouping words into horizontal lines

# Static ROI coordinates
static_roi = {"x": 1275, "y": 89, "width": 645, "height": 248}
round_roi = {"x": 921, "y": 1, "width": 79, "height": 15}

# Initialize global variables for ROI and team names
x_player, y_player, w_player, h_player = static_roi.values()
team1_name = "LLL"
team2_name = "OPTC"

# UI for dynamic ROI and team name entry
def roi_and_team_ui():
    def submit():
        global team1_name, team2_name
        team1_name = team1_var.get().strip()
        team2_name = team2_var.get().strip()
        root.destroy()

    global dynamic_roi
    root = Tk()
    root.title("ROI and Team Entry")

    # ROI selection checkbox
    dynamic_roi = BooleanVar(root)
    dynamic_roi.set(False)
    Checkbutton(root, text="Enable dynamic ROI selection", variable=dynamic_roi).grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    # Team 1 entry
    Label(root, text="Team 1 Name:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    team1_var = StringVar(value=team1_name)
    Entry(root, textvariable=team1_var).grid(row=1, column=1, padx=10, pady=5)

    # Team 2 entry
    Label(root, text="Team 2 Name:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    team2_var = StringVar(value=team2_name)
    Entry(root, textvariable=team2_var).grid(row=2, column=1, padx=10, pady=5)

    # Submit button
    Button(root, text="Submit", command=submit).grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()

# Function to select ROI dynamically if needed
def select_roi_if_needed():
    global x_player, y_player, w_player, h_player, video
    if dynamic_roi.get():
        success, first_frame = video.read()
        if not success:
            print("Failed to read the first frame.")
            exit()
        print("Select ROI for player data...")
        roi_player = cv2.selectROI('Select Region of Interest for Player Data', first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x_player, y_player, w_player, h_player = map(int, roi_player)
        print(f"Dynamic ROI selected: x={x_player}, y={y_player}, width={w_player}, height={h_player}")
    else:
        print(f"Using static ROI: x={x_player}, y={y_player}, width={w_player}, height={h_player}")

# Initialize the video object and show the UI for team and ROI configuration
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

roi_and_team_ui()
select_roi_if_needed()

# Reinitialize video object after ROI selection
video.release()
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Failed to reopen video: {video_path}")
    exit()

# Initialize variables
all_text_data = []  # To store all detected text
frame_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Crop the region of interest for player data
    cropped_player_frame = frame[y_player:y_player+h_player, x_player:x_player+w_player]
    player_frame_filename = os.path.join(frame_output_dir, f"player_frame_{frame_count:04d}.jpg")
    cv2.imwrite(player_frame_filename, cropped_player_frame)

    # Perform OCR on the player data
    with open(player_frame_filename, 'rb') as image_file:
        content = image_file.read()
        vision_image = vision.Image(content=content)

    player_response = client.text_detection(image=vision_image)
    player_texts = player_response.text_annotations

    # Crop the region of interest for round number using static location
    x_round = round_roi["x"]
    y_round = round_roi["y"]
    w_round = round_roi["width"]
    h_round = round_roi["height"]
    cropped_round_frame = frame[y_round:y_round+h_round, x_round:x_round+w_round]
    round_frame_filename = os.path.join(frame_output_dir, f"round_frame_{frame_count:04d}.jpg")
    cv2.imwrite(round_frame_filename, cropped_round_frame)

    # Perform OCR on the round number
    with open(round_frame_filename, 'rb') as image_file:
        content = image_file.read()
        vision_image = vision.Image(content=content)

    round_response = client.text_detection(image=vision_image)
    round_texts = round_response.text_annotations

    # Skip if no text detected for player data or round number
    if not player_texts or not round_texts:
        frame_count += 1
        continue

    # Extract detected round number
    round_number = round_texts[0].description.strip() if round_texts else "Unknown"

    # Group detected words for player data
    line_map = defaultdict(list)
    for annotation in player_texts[1:]:  # Skip the first element (full text)
        word = annotation.description
        vertices = annotation.bounding_poly.vertices
        ymin = min(vertex.y for vertex in vertices)
        ymax = max(vertex.y for vertex in vertices)
        xmin = min(vertex.x for vertex in vertices)
        xmax = max(vertex.x for vertex in vertices)

        # Determine the line based on Y-center of the bounding box
        line_center = (ymin + ymax) // 2
        added_to_line = False

        # Group words by horizontal lines
        for existing_line in line_map.keys():
            if abs(existing_line - line_center) <= line_tolerance:
                line_map[existing_line].append((word, xmin, xmax))
                added_to_line = True
                break

        # If not added to any existing line, create a new one
        if not added_to_line:
            line_map[line_center].append((word, xmin, xmax))

    # Sort lines top to bottom and words left to right
    for _, words in sorted(line_map.items()):  # Sort lines by Y-center
        words = sorted(words, key=lambda x: x[1])  # Sort words by X-min
        line_text = " ".join([word[0] for word in words])
        all_text_data.append([round_number, line_text])

    frame_count += 1

# Step 2: Save all detected text to CSV
with open(output_csv_all, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Round Number", "Detected Text"])  # Header
    writer.writerows(all_text_data)

# Step 3: Post-process to filter lines, split every 2 words, and remove duplicates
filtered_text_data = []
seen_lines = set()

with open(output_csv_all, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        round_number, detected_line = row
        detected_line = re.sub(r'[^\x00-\x7F]+', '', detected_line)  # Remove non-Unicode chars
        words = detected_line.split()

        # Filter lines that contain at least one team name
        if any(team in words for team in [team1_name, team2_name]):
            # Split into pairs of 2 words
            split_pairs = [" ".join(words[i:i+2]) for i in range(0, len(words), 2)]
            unique_entry = [round_number] + split_pairs

            # Avoid duplicates
            entry_str = " | ".join(unique_entry)
            if entry_str not in seen_lines:
                filtered_text_data.append(unique_entry)
                seen_lines.add(entry_str)

# Save the filtered text to CSV
with open(output_csv_filtered, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Round Number", "Kill", "Death"])  # Header
    writer.writerows(filtered_text_data)

# Cleanup temporary frames
shutil.rmtree(frame_output_dir)
print(f"Temporary frames deleted. All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")
