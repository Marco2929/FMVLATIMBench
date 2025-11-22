import base64
import os
from pathlib import Path
from PIL import Image

def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pad_image(image_path: Path, grid_size) -> Path:
    """Pad the image to make its dimensions multiples of grid_size.
        Args:
            image_path (str): Path to the input image.
            grid_size (int): The grid size to pad to.
        Returns:
            str: Path to the padded image (image_path with '_padded' suffix).
    """

    image = Image.open(image_path)
    width, height = image.size
    new_width = ((width + grid_size - 1) // grid_size) * grid_size
    new_height = ((height + grid_size - 1) // grid_size) * grid_size

    print(f"Padding image from ({width}, {height}) to ({new_width}, {new_height})")

    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, (0, 0))
    padded_image_path = image_path.with_name(image_path.stem + "_padded.png")
    padded_image.save(padded_image_path)
    return padded_image_path

def get_api_key() -> str:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if API_KEY:
        return API_KEY
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            API_KEY = os.getenv("OPENROUTER_API_KEY")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
    return API_KEY