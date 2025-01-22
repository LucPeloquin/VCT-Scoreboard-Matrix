import cv2
import easyocr
import numpy as np
import pyautogui
import keyboard
from PIL import ImageGrab

# Global variables for mouse callback
start_point = None
end_point = None
is_drawing = False
screenshot = None  # To store the captured screenshot

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for selecting a region on the screenshot.
    """
    global start_point, end_point, is_drawing
    
    # Left mouse button pressed down - start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        start_point = (x, y)
        end_point = (x, y)
    
    # Mouse movement - update end_point if drawing
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        end_point = (x, y)
    
    # Left mouse button released - finish drawing
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        end_point = (x, y)

def take_screenshot():
    """
    Takes a screenshot of the entire screen and returns it as a NumPy array.
    """
    print("Taking a screenshot of the screen...")
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)  # Convert to NumPy array
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (for OpenCV)
    return screenshot

def perform_ocr_on_selection(image, x1, y1, x2, y2):
    """
    Perform OCR using EasyOCR on the selected region of the image.
    """
    # Crop the selected region
    cropped_image = image[y1:y2, x1:x2]
    
    # Convert cropped image to RGB for EasyOCR
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])  # Add more languages as needed
    
    # Perform OCR on the cropped image
    results = reader.readtext(cropped_image)
    
    # Print the OCR results
    for (bbox, text, confidence) in results:
        print(f"Detected text: {text}")
        print(f"Bounding box: {bbox}")
        print(f"Confidence: {confidence}")
        print("---")

    return results

def main():
    global start_point, end_point, is_drawing, screenshot

    print("Press 'Alt+S' to take a screenshot. Then select a region for OCR.")

    while True:
        # Check if Alt+S is pressed
        if keyboard.is_pressed("alt+s"):
            # Step 1: Take a screenshot
            screenshot = take_screenshot()
            clone = screenshot.copy()  # Clone for resetting the image after drawing
            
            # Step 2: Display the screenshot and allow user to select a region
            cv2.namedWindow("Screenshot")
            cv2.setMouseCallback("Screenshot", mouse_callback)
            
            print("Drag your mouse to select a region. Press 'c' to confirm the selection or 'q' to quit.")

            while True:
                temp_image = clone.copy()
                
                # If the user is drawing, show the rectangle on the screen
                if start_point and end_point and is_drawing:
                    cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
                
                # Display the screenshot
                cv2.imshow("Screenshot", temp_image)
                
                # Wait for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("c"):  # Confirm selection
                    if start_point and end_point:
                        x1, y1 = start_point
                        x2, y2 = end_point
                        
                        # Ensure coordinates are in the correct order (top-left to bottom-right)
                        x1, x2 = sorted([x1, x2])
                        y1, y2 = sorted([y1, y2])
                        
                        print(f"Selected region: ({x1}, {y1}), ({x2}, {y2})")
                        
                        # Perform OCR on the selected region
                        perform_ocr_on_selection(screenshot, x1, y1, x2, y2)
                    else:
                        print("No region selected!")
                
                elif key == ord("q"):  # Quit
                    print("Exiting selection mode...")
                    break

            # Cleanup the OpenCV window
            cv2.destroyAllWindows()

        # Check if the user presses 'q' to exit the script
        if keyboard.is_pressed("q"):
            print("Exiting the program...")
            break

if __name__ == "__main__":
    main()
