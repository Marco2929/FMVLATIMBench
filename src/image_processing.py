
import base64
from pathlib import Path
from PIL import Image
import cv2

def b64encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_to_webp(image_path: Path) -> Path:
    """Convert image to WEBP format for better compression.
        Args:
            image_path (Path): Path to the input image.
        Returns:
            Path: Path to the converted WEBP image.
    """
    image = Image.open(image_path)
    webp_path = image_path.with_suffix('.g.webp')
    image.save(webp_path, 'WEBP', quality=80)
    return webp_path

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
    padded_image_path = image_path.with_name(image_path.stem + ".g.png")
    padded_image.save(padded_image_path)
    return padded_image_path

def draw_bounding_box(image_path: Path, bbox: list[int]) -> Path:
    '''Draw bounding box on the image and save it.
    bbox is absolute pixel coordinates [x_min, y_min, x_max, y_max]
    '''
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    output_path = image_path.with_name(image_path.stem + "_bbox.g").with_suffix(image_path.suffix)
    cv2.imwrite(str(output_path), image)
    return output_path

def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image dimensions (width, height)."""
    with Image.open(image_path) as image:
        return image.size