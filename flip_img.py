from PIL import Image

def flip_image(image_path, output_path, flip_direction='horizontal'):
    """
    Flips an image horizontally or vertically and saves it as a JPEG.

    :param image_path: Path to the input image.
    :param output_path: Path to save the flipped JPEG image.
    :param flip_direction: 'horizontal' for horizontal flip, 'vertical' for vertical flip.
    """
    # Open the image
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening the image: {e}")
        return

    # Convert the image to RGB mode if it has an alpha channel (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Flip the image
    if flip_direction == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 'vertical':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Invalid flip direction. Use 'horizontal' or 'vertical'.")

    # Save the flipped image as a JPEG
    try:
        flipped_image.save(output_path, "JPEG")
        print(f"Image flipped {flip_direction}ly and saved to {output_path}")
    except Exception as e:
        print(f"Error saving the flipped image: {e}")

# Example usage
if __name__ == "__main__":
    input_image = "Vandal_icon.png"
    output_image = "Vandal_icon_flipped.png"
    flip_direction = "horizontal"  # or 'vertical'

    flip_image(input_image, output_image, flip_direction)