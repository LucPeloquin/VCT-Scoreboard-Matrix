import cv2
import os
import csv
from google.cloud import vision
from google.oauth2 import service_account
from collections import defaultdict

credentials = service_account.Credentials.from_service_account_file(r"G:\valorant-vision\valorant-vod-scraper-8267e96f46ba.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

print("Google Vision API client initialized successfully!")

# Paths
image_path = 'ss.png'  # Default image
output_cropped_image = 'cropped_image.png'  # Output cropped image
csv_file = 'split_text_by_large_gaps.csv'  # Output CSV file

# Step 1: Load image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image from {image_path}")
    exit()

# Step 2: Select ROI
roi = cv2.selectROI('Select Region of Interest', image, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = map(int, roi)
cropped_section = image[y:y+h, x:x+w]

# Step 3: Save the cropped section
cv2.imwrite(output_cropped_image, cropped_section)
print(f"Cropped image saved to {output_cropped_image}")

# Step 4: Use Google Vision to extract text
with open(output_cropped_image, 'rb') as image_file:
    content = image_file.read()
    vision_image = vision.Image(content=content)

response = client.text_detection(image=vision_image)
texts = response.text_annotations

# Step 5: Group texts by horizontal lines and split by large gaps
lines = []
if texts:
    line_map = defaultdict(list)
    tolerance = 10  # Tolerance for considering text to be on the same line

    # Group words by their vertical position
    for text in texts[1:]:  # Skip the first element (full text)
        vertices = text.bounding_poly.vertices
        ymin = min(vertex.y for vertex in vertices)
        ymax = max(vertex.y for vertex in vertices)
        line_center = (ymin + ymax) // 2

        # Group words by line using a vertical tolerance
        for key in line_map.keys():
            if abs(key - line_center) <= tolerance:
                line_map[key].append((text.description, vertices))
                break
        else:
            line_map[line_center].append((text.description, vertices))

    # Split words into columns based on large gaps
    for _, words in sorted(line_map.items()):
        words = sorted(words, key=lambda x: min(v.x for v in x[1]))  # Sort words by x-coordinate
        line_columns = []
        current_column = []
        prev_x_max = None

        for word, vertices in words:
            x_min = min(vertex.x for vertex in vertices)
            x_max = max(vertex.x for vertex in vertices)

            if prev_x_max is not None and x_min - prev_x_max > 50:  # Large gap threshold
                # Start a new column
                line_columns.append(" ".join(current_column))
                current_column = [word]
            else:
                current_column.append(word)

            prev_x_max = x_max

        # Add the last column
        if current_column:
            line_columns.append(" ".join(current_column))

        lines.append(line_columns)

# Step 6: Save lines to CSV
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for line in lines:
        writer.writerow(line)

print(f"Text lines split by large gaps saved to {csv_file}")
