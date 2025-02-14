from PIL import Image
import os
from pathlib import Path

def flip_folder_images(input_folder, output_folder):
    """
    Flips all images in a folder horizontally and saves them to an output folder.
    Preserves transparency, original image format, and original filenames.
    Supports common image formats like PNG, JPEG, and BMP.

    :param input_folder: Path to the folder containing input images
    :param output_folder: Path to save the flipped images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        file_extension = Path(filename).suffix.lower()
        if file_extension not in supported_extensions:
            continue
            
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # Using original filename
        
        try:
            # Open and process the image
            with Image.open(input_path) as image:
                # Flip the image horizontally
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Save the flipped image with original format and transparency
                flipped_image.save(output_path, format=image.format)
                print(f"Successfully flipped: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "icons"
    output_folder = "icons_flipped"
    
    flip_folder_images(input_folder, output_folder)