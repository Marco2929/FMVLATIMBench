import base64
import json
import os
from pathlib import Path
from PIL import Image
from openai import OpenAI

import imagehash
import math
import cv2
import numpy as np

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
    padded_image_path = image_path.with_name(image_path.stem + ".g.png")
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


def parse_ground_truth(json_path: Path) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["solution"]

def generate_model_response(image_path: Path, api_key: str, SYSTEM_PROMPT: str, instruct_prompt: str,
                            model_name="qwen/qwen3-vl-8b-instruct",
                            base_url="https://openrouter.ai/api/v1"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + instruct_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    response = client.chat.completions.create(model=model_name, messages=messages)
    part_name = response.choices[0].message.content
    print(f"Model Response: {part_name}")
    return part_name

def parse_response(response: str):
    normalized_response = response.upper().replace(" ", "_")

    return normalized_response.strip()


def evaluate_response(ground_truth: str, response: str):
    return ground_truth == response


class VGBenchScorer:
    def __init__(self, checkpoint_folder, threshold=12):
        """
        initializes the scorer with reference images.

        :param checkpoint_folder: Path to folder containing '1.png', '2.png', etc.
        :param threshold: Hamming distance tolerance.
                          0 = exact match.
                          < 5 = extremely similar.
                          < 15 = similar structure (tolerant to minor shifts/artifacts).
        """
        self.threshold = threshold
        self.checkpoints = {}  # Dict to store loaded hashes: {1: hash_obj, 2: hash_obj}
        self.current_checkpoint_index = 0 # We start before the first checkpoint

        self._load_checkpoints(checkpoint_folder)

    def _load_checkpoints(self, folder):
        """Loads and pre-hashes all checkpoint images for speed."""
        print(f"Loading checkpoints from {folder}...")

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} not found.")

        # Assume files are named 1.png, 2.png, etc.
        files = sorted([f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))],
                       key=lambda x: int(os.path.splitext(x)[0]))

        for f in files:
            # Extract the number "1" from "1.png"
            cp_num = int(os.path.splitext(f)[0])
            img_path = os.path.join(folder, f)

            with Image.open(img_path) as img:
                # We use phash (Perceptual Hash) as per the VGBench paper.
                # It is robust against resizing and minor color shifts.
                img_hash = imagehash.phash(img)
                self.checkpoints[cp_num] = img_hash
                print(f"  Loaded Checkpoint {cp_num}: {img_hash}")

    def update(self, current_screen_image):
        """
        Takes the current game screen, hashes it, and checks against upcoming checkpoints.

        :param current_screen_image: A PIL Image object of the current game frame.
        :return: (found_new_checkpoint, checkpoint_number, distance)
        """
        # 1. Generate Perceptual Hash of current screen
        #    (The paper uses a 64-bit hash usually represented as a hex string)
        current_hash = imagehash.phash(current_screen_image)

        # 2. Optimization: Only check the *next* few checkpoints.
        #    We don't want to accidentally match Checkpoint 10 if we haven't passed Checkpoint 2.
        #    (Allowing a window of 3 lets the agent skip a minor intermediate checkpoint if it was missed)
        search_window = 3
        start_search = self.current_checkpoint_index + 1
        end_search = start_search + search_window

        # Check against relevant upcoming checkpoints
        for cp_num in range(start_search, end_search + 1):
            if cp_num not in self.checkpoints:
                continue

            target_hash = self.checkpoints[cp_num]

            # 3. Calculate Hamming Distance
            #    This counts how many bits differ between the two hashes.
            distance = current_hash - target_hash

            if distance <= self.threshold:
                # MATCH FOUND!
                self.current_checkpoint_index = cp_num
                return True, cp_num, distance

        # No new checkpoint found
        return False, None, 0

class DistanceTracker:
    def __init__(self, targets):
        """
        :param targets: Dict of { 'object_name': (target_x, target_y) } or { 'object_name': target_value }
                        Example: {'key_key': (200, 300), 'player': (50, 50)}
                        For single coordinate: {'object_x': 200} or {'object_y': 300}
                        Use None for coordinates that should be ignored: {'object': (200, None)} for x-only
        """
        self.targets = targets
        self.max_possible_distance = 0 # Optional: for normalization

    def calculate_progress(self, current_positions):
        """
        Calculates how far all objects are from their goals.
        LOWER score is better (0 = Solved).

        :param current_positions: Dict { 'object_name': (x, y) } or { 'object_name': value }
        :return: (total_distance, details_dict)
        """
        total_distance = 0
        self.details = {}

        for obj_name, target_pos in self.targets.items():
            if obj_name in current_positions:
                curr_pos = current_positions[obj_name]

                # Handle single coordinate (scalar value)
                if isinstance(target_pos, (int, float)) and isinstance(curr_pos, (int, float)):
                    # Both are scalars - calculate 1D distance
                    dist = abs(target_pos - curr_pos)

                # Handle tuple coordinates
                elif isinstance(target_pos, tuple) and isinstance(curr_pos, tuple):
                    # Check if we should only consider x or y coordinate
                    if target_pos[0] is None and target_pos[1] is not None:
                        # Only y-coordinate matters
                        dist = abs(target_pos[1] - curr_pos[1])
                    elif target_pos[1] is None and target_pos[0] is not None:
                        # Only x-coordinate matters
                        dist = abs(target_pos[0] - curr_pos[0])
                    elif target_pos[0] is not None and target_pos[1] is not None:
                        # Both coordinates matter - Euclidean Distance
                        dist = math.hypot(target_pos[0] - curr_pos[0], target_pos[1] - curr_pos[1])
                    else:
                        # Both are None - invalid
                        dist = float('inf')

                # Handle mixed types (tuple vs scalar)
                elif isinstance(target_pos, tuple) and isinstance(curr_pos, (int, float)):
                    # Assume curr_pos is x-coordinate if target has x, otherwise y
                    if target_pos[0] is not None and target_pos[1] is None:
                        dist = abs(target_pos[0] - curr_pos)
                    elif target_pos[1] is not None and target_pos[0] is None:
                        dist = abs(target_pos[1] - curr_pos)
                    else:
                        dist = float('inf')

                elif isinstance(curr_pos, tuple) and isinstance(target_pos, (int, float)):
                    # Target is scalar, current is tuple - use first non-None value
                    if curr_pos[0] is not None:
                        dist = abs(target_pos - curr_pos[0])
                    elif curr_pos[1] is not None:
                        dist = abs(target_pos - curr_pos[1])
                    else:
                        dist = float('inf')

                else:
                    # Unexpected type combination
                    dist = float('inf')

                total_distance += dist
                self.details[obj_name] = dist
            else:
                # Penalty if object is missing from screen
                self.details[obj_name] = float('inf')

        return total_distance, self.details

    def visualize(self, image, current_positions):
        """Draws lines from current position to target position for debugging."""
        debug_img = np.array(image) # Convert PIL to numpy if needed
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

        for name, pos in current_positions.items():
            if name in self.targets:
                target = self.targets[name]
                # Draw Line (Red)
                cv2.line(debug_img, pos, target, (0, 0, 255), 2)
                # Draw Target (Green Circle)
                cv2.circle(debug_img, target, 5, (0, 255, 0), -1)
                # Draw Current (Blue Circle)
                cv2.circle(debug_img, pos, 5, (255, 0, 0), -1)
                # Draw Text
                cv2.putText(debug_img, f"{int(self.details[name])}px",
                           (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return debug_img

class VisualLocator:
    def __init__(self, templates):
        """
        :param templates: Dict { 'object_name': 'path/to/icon.png' }
        """
        self.templates = {}
        for name, path in templates.items():
            if os.path.exists(path):
                # Load in grayscale for robustness
                self.templates[name] = cv2.imread(path, 0)
            else:
                print(f"Warning: Template {path} not found")

    def locate_objects(self, screenshot):
        """
        Finds objects in the full screenshot.
        :param screenshot: PIL Image or path
        :return: Dict { 'object_name': (center_x, center_y) }
        """
        # Convert PIL to OpenCv Grayscale
        img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
        found_positions = {}

        for name, template in self.templates.items():
            w, h = template.shape[::-1]

            # Match Template
            res = cv2.matchTemplate(img_cv, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8 # 80% confidence match

            # Get location of best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= threshold:
                # max_loc is top-left corner. We want the center.
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                found_positions[name] = (center_x, center_y)

        return found_positions