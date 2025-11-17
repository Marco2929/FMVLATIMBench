import base64
import os

def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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