import os
import cv2
import numpy as np
from PIL import Image

def outline_in_red(input_dir, output_dir):
    """
    Outline the content of PNG images in red, maintaining transparency in the background.

    :param input_dir: Directory containing input images.
    :param output_dir: Directory to save processed images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each PNG file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            file_path = os.path.join(input_dir, filename)
            with Image.open(file_path) as img:
                # Convert image to RGBA if not already
                img = img.convert('RGBA')

                # Convert to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

                # Convert to grayscale
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)

                # Edge detection
                edges = cv2.Canny(gray, 100, 200)

                # Dilate edges to make them more visible
                kernel = np.ones((3, 3), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)

                # Create a red outline
                outline = np.zeros_like(cv_img)
                outline[:, :, 0] = dilated_edges  # Red channel
                outline[:, :, 3] = dilated_edges  # Alpha channel

                # Combine the original image with the red outline
                final_img = cv2.addWeighted(cv_img, 1, outline, 1, 0)

                # Save the new image
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, final_img)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_directory = "icons_og"  # Replace with your input directory path
    output_directory = "outlined_icons"  # Replace with your output directory path
    outline_in_red(input_directory, output_directory) 