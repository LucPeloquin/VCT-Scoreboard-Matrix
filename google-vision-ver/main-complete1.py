import cv2
import os
import csv
import shutil
from google.cloud import vision
from google.oauth2 import service_account
from collections import defaultdict
import sys
import re  # For removing non-Unicode characters

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

# Initialize video
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Failed to open video: {video_path}")
    exit()

# Step 1: Set ROI (fixed for all frames)
success, first_frame = video.read()
if not success:
    print("Failed to read the first frame.")
    exit()

roi = cv2.selectROI('Select Region of Interest', first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
x, y, w, h = map(int, roi)

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

    # Crop the region of interest
    cropped_frame = frame[y:y+h, x:x+w]

    # Save the cropped frame temporarily
    frame_filename = os.path.join(frame_output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, cropped_frame)

    # Perform OCR on the cropped frame
    with open(frame_filename, 'rb') as image_file:
        content = image_file.read()
        vision_image = vision.Image(content=content)

    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    # Skip if no text detected
    if not texts:
        print(f"Frame {frame_count}: No text detected.")
        frame_count += 1
        continue

    # Group detected words into horizontal lines
    line_map = defaultdict(list)
    for annotation in texts[1:]:  # Skip the first element (full text)
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
        all_text_data.append([line_text])
        print(f"Frame {frame_count}: {line_text}")  # Properly handles Unicode

    frame_count += 1

# Step 2: Save all detected text to CSV
with open(output_csv_all, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Detected Text"])  # Header
    writer.writerows(all_text_data)

# Step 3: Post-process to filter lines containing "LLL" or "OPTC"
filtered_text_data = []

# Read all detected text and filter based on prefixes
with open(output_csv_all, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        detected_line = row[0]
        # Remove non-Unicode characters
        detected_line = re.sub(r'[^\x00-\x7F]+', '', detected_line)
        # Check if line contains "LLL" or "OPTC"
        if "LLL" in detected_line or "OPTC" in detected_line:
            filtered_text_data.append([detected_line])

# Step 4: Remove duplicates while maintaining order
unique_filtered_data = []
seen_lines = set()
for line in filtered_text_data:
    if line[0] not in seen_lines:
        unique_filtered_data.append(line)
        seen_lines.add(line[0])

# Save filtered text to CSV
with open(output_csv_filtered, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Filtered Text"])  # Header
    writer.writerows(unique_filtered_data)

# Cleanup temporary frames
shutil.rmtree(frame_output_dir)
print(f"Temporary frames deleted. All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")