import cv2
import os
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
import logging

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
capture_interval = 0.2  # Time interval (in seconds) between captures
line_tolerance = 15  # Tolerance for grouping words into the same horizontal line
duration_minutes = .5  # Run for 1.5 minutes
min_confidence = 0.9  # Minimum confidence for OCR

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# Initialize time tracking
last_detected_time = None
time_countdown = 40

# Add logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='template_matching.log'
)

# Function to detect an image within the static ROI
def detect_image_in_roi(sct, template_path, threshold=0.7):
    # Load the template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Template image not found at {template_path}")
        return False, None, 0, None

    # Capture the screen within the static ROI
    screen = sct.grab(static_roi)
    screen_image = np.array(screen)
    screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGRA2GRAY)

    # Use a fixed scale for template matching
    found = False
    best_confidence = 0
    best_location = None

    # Set the scale factor to resize the template width to 81 pixels
    scale = 78 / template.shape[1]
    resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    if resized_template.shape[0] <= screen_image.shape[0] and resized_template.shape[1] <= screen_image.shape[1]:
        result = cv2.matchTemplate(screen_image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            found = True
            best_confidence = max_val
            best_location = max_loc

    if found:
        return True, best_location, best_confidence, resized_template.shape[:2]
    else:
        return False, None, best_confidence, None

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

# Function to split team names into Player and Player 2 columns
def process_csv_with_team_split(csv_path, team1, team2):
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        header = rows[0][:3] + ["Player 2", "Weapon"]  # Include "Weapon" in the header
        updated_rows = [header]

        for row in rows[1:]:
            round_number, detected_time, detected_text, weapon = row  # Unpack all 4 values

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

            if player1 or player2:  # Only add rows where players were found
                updated_rows.append([round_number, detected_time, player1, player2, weapon])

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
            round_number, detected_time, player, player2, weapon = row  # Unpack all 5 values

            # Create a similarity key for players
            player_key = ''.join(sorted(player))  # Sort characters for fuzzy matching

            if player_key not in seen or fuzz.ratio(seen[player_key]["original"], player) < 98:
                seen[player_key] = {"original": player, "time": detected_time}
                unique_rows.append(row)

            else:
                # If the current player has an earlier time, replace the entry
                existing_time = seen[player_key]["time"]
                if detected_time < existing_time:
                    seen[player_key] = {"original": player, "time": detected_time}
                    unique_rows = [r for r in unique_rows if r != seen[player_key]["original"]]
                    unique_rows.append(row)

    # Write the unique rows back to the CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(unique_rows)

    print(f"Removed duplicates from {csv_path} by prioritizing earliest entries for similar players.")
    print(f"Removed duplicates from {csv_path} with leniency for 2 characters difference in Player and 3 in Player 2.")

# Function to overlay the template on the detected location
def overlay_template_on_screenshot(screenshot, template, location, detected_size):
    # Convert the screenshot to a format suitable for overlay
    overlay = screenshot.copy()
    
    # Resize the template to match the detected size
    resized_template = cv2.resize(template, detected_size, interpolation=cv2.INTER_LINEAR)
    
    # Get the dimensions of the resized template
    template_height, template_width = resized_template.shape[:2]
    
    # Define the region of interest (ROI) on the overlay
    x, y = location
    roi = overlay[y:y+template_height, x:x+template_width]
    
    # Ensure the template is in BGR format
    if resized_template.shape[2] == 4:  # If the template has an alpha channel
        resized_template = cv2.cvtColor(resized_template, cv2.COLOR_BGRA2BGR)
    
    # Create a mask of the template and its inverse mask
    template_gray = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(template_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Ensure the ROI and mask are compatible
    if roi.shape[2] == 4:  # If the ROI has an alpha channel
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    
    # Ensure the mask is the same size as the ROI
    mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
    mask_inv = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))
    
    # Black-out the area of the template in the ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only the region of the template from the template image
    img_fg = cv2.bitwise_and(resized_template, resized_template, mask=mask)
    
    # Put the template in the ROI and modify the overlay
    dst = cv2.add(img_bg, img_fg)
    
    # Convert overlay to BGR if it has an alpha channel
    if overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
    
    overlay[y:y+template_height, x:x+template_width] = dst
    
    return overlay

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

    # Load only Ghost.png
    template_path = "icons/Ghost.png"
    print(f"Loaded template: {template_path}")

    while datetime.now() < end_time:
        # Detect numbers from the number ROIs
        number1, number2, round_number = detect_numbers(sct)

        # Detect time from the time ROI
        detected_time = detect_time(sct)

        # Define the static ROI for main text detection
        roi = {"left": static_roi["left"], "top": static_roi["top"],
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

        # Print the detected lines of text
        for line in detected_text_lines:
            print(f"Detected line: {line}")

            # Initialize weapon as empty string
            detected_weapon = ""
            
            # Only check for templates if we have detected text
            found, location, confidence, detected_size = detect_image_in_roi(sct, template_path, threshold=0.85)
                
            if found:
                template_name = "Ghost.png"
                print(f"Text line detected: {line}")
                print(f"Ghost icon detected at {location} with confidence {confidence:.2f}")
                
                # Capture a screenshot of the ROI when the template is detected
                detected_frame = sct.grab(static_roi)
                detected_image = np.array(detected_frame)
                
                # Load the template image
                template_img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                
                # Overlay the template on the screenshot
                overlay_image = overlay_template_on_screenshot(detected_image, template_img, location, detected_size)
                
                # Save the overlay image
                overlay_screenshot_path = os.path.join('detected_screenshots', 
                                                       f"overlay_Ghost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(overlay_screenshot_path, overlay_image)
                print(f"Overlay screenshot saved: {overlay_screenshot_path}")
                
                # Store the weapon name without extension
                detected_weapon = "Ghost"
            
            all_text_data.append([frame_count, number1, number2, round_number, detected_time, line, detected_weapon])

        frame_count += 1
        time.sleep(capture_interval)


# Save all detected text to CSV
with open(output_csv_all, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Number 1", "Number 2", "Round", "Time", "Player 1", "Weapon"])
    writer.writerows(all_text_data)

# Post-process and save filtered text
filtered_text_data = []
seen_lines = set()

for row in all_text_data:
    _, _, _, round_number, detected_time, text_line, weapon = row  # Added weapon to unpacking
    words = text_line.split()
    if words and words[0] in (team1_name, team2_name):  # Check if the line starts with a team name
        if text_line not in seen_lines:
            filtered_text_data.append([round_number, detected_time, text_line, weapon])  # Added weapon to data
            seen_lines.add(text_line)

# Save filtered text to CSV
with open(output_csv_filtered, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "Time", "Player 1", "Weapon"])  # Added "Weapon" to header
    writer.writerows(filtered_text_data)

# Call the function to split teams and update the CSV
process_csv_with_team_split(output_csv_filtered, team1_name, team2_name)

# Remove duplicates based on Round, Player, and Player 2
remove_duplicates(output_csv_filtered)

print(f"All detected text saved to {output_csv_all}")
print(f"Filtered text saved to {output_csv_filtered}")