import os
from PIL import Image
import cv2
import numpy as np

def convert_to_white_silhouette(input_dir, output_dir, white_intensity=255, inpaint_radius=50):
    """
    Convert grayscale PNG images to white silhouettes with adjustable intensity and maintain transparency.

    :param input_dir: Directory containing input images.
    :param output_dir: Directory to save processed images.
    :param white_intensity: Intensity of the white fill (0-255).
    :param inpaint_radius: Radius of the circular neighborhood of each point inpainted.
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
                data = img.getdata()

                # Create a new image data list
                new_data = []
                for item in data:
                    # Change all non-transparent pixels to white with specified intensity
                    if item[3] > 0:  # Check if the pixel is not fully transparent
                        new_data.append((white_intensity, white_intensity, white_intensity, item[3]))  # White with original alpha
                    else:
                        new_data.append(item)  # Keep transparent pixels as they are

                # Update image data
                img.putdata(new_data)

                # Convert to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

                # Convert to BGR for inpainting
                bgr_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

                # Convert to grayscale
                gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

                # Create a mask for inpainting
                _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

                # Save or display the mask for verification
                mask_output_path = os.path.join(output_dir, f"mask_{filename}")
                cv2.imwrite(mask_output_path, mask)
                print(f"Mask saved: {mask_output_path}")

                # Inpaint the image with a higher radius
                inpainted = cv2.inpaint(bgr_img, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

                # Combine the inpainted image with the original alpha channel
                b, g, r, a = cv2.split(cv_img)
                final_img = cv2.merge((inpainted[:, :, 0], inpainted[:, :, 1], inpainted[:, :, 2], a))

                # Save the new image
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, final_img)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_directory = "icons_og"  # Replace with your input directory path
    output_directory = "icons"  # Replace with your output directory path
    white_intensity = 255  # Adjust this value to fine-tune the white intensity
    inpaint_radius = 50  # Increase this value to enhance the inpainting effect
    convert_to_white_silhouette(input_directory, output_directory, white_intensity, inpaint_radius) 