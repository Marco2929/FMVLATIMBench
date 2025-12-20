import base64
import json
from pathlib import Path
import re
from openai import OpenAI
import argparse
from typing import List
import os
import io
import importlib
from datetime import datetime
import traceback
from PIL import Image

from benchmark4_manipulation.system_prompts.ui_tars_1_5_7B_object_manipul_actions import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_MANIPUL_ACTIONS
from benchmark4_manipulation.system_prompts.ui_tars_1_5_7B_object_placement import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_PLACEMENT
from benchmark4_manipulation.system_prompts.ui_tars_1_5_7B_object_manipul_cot import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_MANIPUL_COT

allowed_categories = ["manipul_actions", "placement", "manipul_cot"]

try:
    import pyautogui
    _have_pyautogui = True
except Exception:
    _have_pyautogui = False

try:
    from PIL import ImageGrab, ImageDraw
    _have_imagegrab = True
except Exception:
    _have_imagegrab = False

try:
    from pynput import keyboard as pynput_keyboard
    _have_pynput = True
except Exception:
    pynput_keyboard = None
    _have_pynput = False

from src.bechmark_base import get_base_url
from src.llm_wrapper import Qwen3VLLLMWrapper
from ui_tars_1_5_7B.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, parsing_response_to_pydirectinput_code
from utils import get_api_key, encode_image, DistanceTracker, VisualLocator, calculate_iou


# Configuration constants
MAX_MESSAGE_SIZE = 10  # maximum number of messages to keep in history (including user and assistant messages)
HOTKEY = "<ctrl>+<shift>+s"  # Cross-platform hotkey (Ctrl+Shift+S)
MY_RESOLUTION = (1728, 1117)
DEVICE_PIXEL_RATIO = 2.0  # Device pixel ratio (1.0 for standard displays, 2.0 for Retina/HiDPI)
MAX_ACTIONS = 5

# Global flag for hotkey detection
_hotkey_pressed = False


def get_system_prompt(input_category: str):
    """Get the appropriate system prompt based on input category."""
    if input_category == allowed_categories[0]:
        return SYSTEM_PROMPT_MANIPUL_ACTIONS
    elif input_category == allowed_categories[1]:
        return SYSTEM_PROMPT_PLACEMENT
    elif input_category == allowed_categories[2]:
        return SYSTEM_PROMPT_MANIPUL_COT
    else:
        # Placeholder for future system prompt implementation
        pass
    return ""


def parse_ground_truth(json_path: Path) -> dict:
    """Parse ground truth from JSON file.
    
    Args:
        json_path: Path to the JSON file containing ground truth data
        
    Returns:
        Dictionary containing:
            - 'targets': Dict of target positions {'object_name': (x, y)} or single values
            - 'templates': Dict of template paths {'object_name': 'path/to/template.png'}
            - 'threshold': Distance threshold for success (default: 10)
            - 'use_iou': Boolean flag to enable IoU evaluation (default: False)
            - 'target_boxes': Optional dict of target bounding boxes (only used if use_iou=True)
            - 'iou_threshold': Optional IoU threshold for success (default: 0.5)
            - 'negative': Dict of negative flags {'object_name': bool} (default: False for all)
    
    Expected JSON format:
    {
        "targets": {
            "ball": [200, 300],
            "key": [100, 150]
        },
        "templates": {
            "ball": "templates/ball.png",
            "key": "templates/key.png"
        },
        "threshold": 15,
        "use_iou": true,
        "target_boxes": {
            "ball": [180, 280, 220, 320],
            "key": [80, 130, 120, 170]
        },
        "iou_threshold": 0.5,
        "negative": {
            "key": true
        }
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Parse targets - convert lists to tuples
    targets = {}
    for obj_name, pos in data.get("targets", {}).items():
        if isinstance(pos, list):
            # Convert list to tuple, handling None values
            targets[obj_name] = tuple(pos) if len(pos) == 2 else pos[0]
        else:
            # Keep scalar values as-is
            targets[obj_name] = pos
    
    # Get templates - ensure paths are relative to the JSON file location
    templates = {}
    json_dir = json_path.parent
    for obj_name, template_path in data.get("templates", {}).items():
        # Make template path absolute if it's relative
        template_path = Path(template_path)
        if not template_path.is_absolute():
            template_path = json_dir / template_path
        templates[obj_name] = str(template_path)
    
    # Check if IoU evaluation is enabled
    use_iou = data.get("use_iou", False)
    
    # Parse target boxes if IoU is enabled and boxes are provided
    target_boxes = {}
    if use_iou:
        for obj_name, box in data.get("target_boxes", {}).items():
            if isinstance(box, list) and len(box) == 4:
                target_boxes[obj_name] = tuple(box)
    
    # Parse negative flags (objects that should NOT be found)
    negative = data.get("negative", {})
    
    return {
        "targets": targets,
        "templates": templates,
        "threshold": data.get("threshold", 10),
        "use_iou": use_iou,
        "target_boxes": target_boxes,
        "iou_threshold": data.get("iou_threshold", 0.5),
        "negative": negative
    }


def get_initial_message(SYSTEM_PROMPT: str, instruct_prompt: str):
    """Get initial message to model before loop.
    
    Args:
        SYSTEM_PROMPT: System prompt for the model
        instruct_prompt: Additional instruction prompt
        
    Returns:
        Initial message list with system prompt only
    """
    message = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + instruct_prompt
        }
    ]
    
    return message


def parse_model_response(response: str):
    """Parse and normalize model response.
    
    Args:
        response: Raw model response string
        
    Returns:
        Parsed and normalized response
    """
    # Get actual screen size
    if _have_pyautogui:
        screen_size = pyautogui.size()
        original_image_width, original_image_height = screen_size.width, screen_size.height
        print(f"Detected screen size via pyautogui: {original_image_width}x{original_image_height}")
    else:
        original_image_width, original_image_height = MY_RESOLUTION
    
    print(f"Screen resolution: {original_image_width}x{original_image_height}")
    
    parsed_dict = parse_action_to_structure_output(
        response,
        factor=1000,
        origin_resized_height=original_image_height,
        origin_resized_width=original_image_width,
        model_type="qwen25vl"
    )
    print(f"Parsed action: {parsed_dict}")
    
    return parsed_dict


def evaluate_response(ground_truth_data: dict, screenshot_path: str = None, screenshot_image: Image.Image = None):
    """Evaluate model response against ground truth using visual matching.
    
    Args:
        ground_truth_data: Dictionary containing:
            - 'targets': Dict of target positions {'object_name': (x, y)} or {'object_name': value}
            - 'templates': Dict of template paths {'object_name': 'path/to/template.png'}
            - 'threshold': Optional distance threshold for success (default: 10 pixels)
            - 'use_iou': Boolean flag to enable IoU evaluation
            - 'target_boxes': Optional dict of target bounding boxes (only used if use_iou=True)
            - 'iou_threshold': Optional IoU threshold for success (default: 0.5)
            - 'negative': Dict of negative flags {'object_name': bool} - objects that should NOT be found
        screenshot_path: Path to the screenshot to evaluate (optional if screenshot_image provided)
        screenshot_image: PIL Image object of the screenshot (optional if screenshot_path provided)
        
    Returns:
        Tuple of (success: bool, total_distance: float, distance_details: dict, iou_scores: dict, mean_iou: float)
    """
    if screenshot_image is None and screenshot_path is None:
        raise ValueError("Either screenshot_path or screenshot_image must be provided")
    
    if screenshot_image is None:
        screenshot_image = Image.open(screenshot_path)
    
    # Extract ground truth parameters
    targets = ground_truth_data.get('targets', {})
    templates = ground_truth_data.get('templates', {})
    threshold = ground_truth_data.get('threshold', 10)
    use_iou = ground_truth_data.get('use_iou', False)
    target_boxes = ground_truth_data.get('target_boxes', {})
    iou_threshold = ground_truth_data.get('iou_threshold', 0.5)
    negative = ground_truth_data.get('negative', {})
    
    # Initialize VisualLocator with templates
    locator = VisualLocator(templates)
    
    # Locate objects - get boxes if IoU is enabled
    if use_iou:
        detected_objects = locator.locate_objects(screenshot_image, return_boxes=True)
        current_positions = {name: obj['center'] for name, obj in detected_objects.items()}
        detected_boxes = {name: obj['box'] for name, obj in detected_objects.items()}
        print(f"Located objects with boxes: {current_positions}")
    else:
        current_positions = locator.locate_objects(screenshot_image, return_boxes=False)
        detected_boxes = {}
        print(f"Located objects: {current_positions}")
    
    # Check negative targets (objects that should NOT be found)
    negative_success = True
    for obj_name, is_negative in negative.items():
        if is_negative:
            object_found = obj_name in current_positions
            if object_found:
                print(f"NEGATIVE CHECK FAILED: Object '{obj_name}' should NOT be found but was detected at {current_positions[obj_name]}")
                negative_success = False
            else:
                print(f"NEGATIVE CHECK PASSED: Object '{obj_name}' correctly NOT found")
    
    # Calculate distance-based metrics (only for non-negative targets)
    positive_targets = {name: target for name, target in targets.items() if not negative.get(name, False)}
    tracker = DistanceTracker(positive_targets)
    total_distance, distance_details = tracker.calculate_progress(current_positions)
    print(f"Total distance: {total_distance:.2f} pixels")
    print(f"Per-object distances: {distance_details}")
    
    # Calculate IoU scores if enabled
    iou_scores = {}
    mean_iou = 0.0
    
    if use_iou and target_boxes:
        total_iou = 0.0
        num_objects = 0
        
        for obj_name, target_box in target_boxes.items():
            if obj_name in detected_boxes:
                detected_box = detected_boxes[obj_name]
                iou = calculate_iou(target_box, detected_box)
                iou_scores[obj_name] = iou
                total_iou += iou
                num_objects += 1
            else:
                # Object not found
                iou_scores[obj_name] = 0.0
        
        if num_objects > 0:
            mean_iou = total_iou / num_objects
        
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Per-object IoU scores: {iou_scores}")
    
    # Determine success based on thresholds
    distance_success = total_distance <= threshold and all(d != float('inf') for d in distance_details.values())
    
    if use_iou and target_boxes:
        iou_success = mean_iou >= iou_threshold and all(iou >= iou_threshold for iou in iou_scores.values() if iou > 1e-6)
        success = distance_success and iou_success and negative_success
    else:
        success = distance_success and negative_success
    
    return success, total_distance, distance_details, iou_scores, mean_iou


def _wait_for_start():
    """Wait for the configured hotkey to be pressed. Fall back to Enter if `pynput` isn't available."""
    global _hotkey_pressed
    
    if _have_pynput:
        print(f"Waiting for hotkey 'Ctrl+Shift+S' to start. Press it to continue...")
        print("(Or press Ctrl+C to cancel and use Enter instead)")
        
        def on_activate():
            global _hotkey_pressed
            _hotkey_pressed = True
        
        try:
            from pynput.keyboard import HotKey, Listener, Key, KeyCode
            
            # Define the hotkey combination: Ctrl+Shift+S
            hotkey = HotKey(
                [KeyCode.from_char('s')],
                on_activate
            )
            
            # Create a listener
            with Listener(
                on_press=lambda k: hotkey.press(k),
                on_release=lambda k: hotkey.release(k)
            ) as listener:
                # Wait until hotkey is pressed
                while not _hotkey_pressed:
                    import time
                    time.sleep(0.1)
        except Exception as e:
            print("Warning: pynput hotkey failed:", e)
            input("Press Enter to start the loop...")
    else:
        input("Press Enter to start the loop (install 'pynput' for hotkey support)...")


def _screenshot_to_base64():
    """Take a screenshot and return as base64-encoded PNG.
    
    Handles device pixel ratio for high-DPI displays by resizing the screenshot
    to match the logical screen resolution.
    
    Returns:
        Base64-encoded PNG bytes as utf-8 string
    """
    try:
        if _have_pyautogui:
            img = pyautogui.screenshot()
        elif _have_imagegrab:
            img = ImageGrab.grab()
        else:
            raise RuntimeError("No screenshot backend available (install pyautogui or pillow).")
        
        # Resize screenshot if device pixel ratio is not 1.0
        if DEVICE_PIXEL_RATIO > 1.0:
            logical_width = int(img.width / DEVICE_PIXEL_RATIO)
            logical_height = int(img.height / DEVICE_PIXEL_RATIO)
            img = img.resize((logical_width, logical_height), Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64
    except Exception as e:
        # If screenshot fails, return empty string so we don't break the loop
        print("Warning: failed to capture screenshot:", e)
        return ""


def _create_action_path_overlay(screenshot_path: str, action_history: List[str], output_path: str, y_offset_factor: float = 0.0):
    """Create an overlay on the screenshot showing the path of actions taken.
    
    Args:
        screenshot_path: Path to the base screenshot
        action_history: List of generated pyautogui/pydirectinput code strings
        output_path: Path to save the overlaid image
        y_offset_factor: Y-axis offset factor to reverse (subtract the offset that was added during code generation)
    """
    import re
    
    try:
        # Open the screenshot
        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # Extract coordinates from generated code
        points = []
        for code in action_history:
            if not code or code == "DONE":
                continue
            
            # Parse click actions: pyautogui.click(x, y, ...) with floats or ints
            click_matches = re.finditer(r'(?:pydirectinput|pyautogui)\.click\s*\(\s*([\d.]+)\s*,\s*([\d.]+)', code)
            for match in click_matches:
                x, y = float(match.group(1)), float(match.group(2))
                # Reverse the y_offset_factor: original_y = adjusted_y / (1 + y_offset_factor)
                original_y = y / (1 + y_offset_factor) if y_offset_factor != 0 else y
                points.append((int(x), int(original_y), 'click'))
            
            # Parse moveTo/hover actions: pyautogui.moveTo(x, y) with floats or ints
            move_matches = re.finditer(r'(?:pydirectinput|pyautogui)\.moveTo\s*\(\s*([\d.]+)\s*,\s*([\d.]+)', code)
            for match in move_matches:
                x, y = float(match.group(1)), float(match.group(2))
                # Reverse the y_offset_factor
                original_y = y / (1 + y_offset_factor) if y_offset_factor != 0 else y
                points.append((int(x), int(original_y), 'move'))
            
            # Parse dragTo actions: pyautogui.dragTo(x, y, ...) with floats or ints
            drag_matches = re.finditer(r'(?:pydirectinput|pyautogui)\.dragTo\s*\(\s*([\d.]+)\s*,\s*([\d.]+)', code)
            for match in drag_matches:
                x, y = float(match.group(1)), float(match.group(2))
                # Reverse the y_offset_factor
                original_y = y / (1 + y_offset_factor) if y_offset_factor != 0 else y
                points.append((int(x), int(original_y), 'drag_end'))
        
        # Draw lines connecting consecutive points
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1, _ = points[i]
                x2, y2, _ = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill='lime', width=3)
        
        # Draw markers for each action point
        for i, (x, y, action_type) in enumerate(points):
            # Color code by action type
            if action_type == 'click':
                color = 'red'
                radius = 8
            elif action_type == 'move':
                color = 'yellow'
                radius = 6
            elif action_type == 'drag_end':
                color = 'cyan'
                radius = 8
            else:
                color = 'white'
                radius = 5
            
            # Draw circle marker
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline='black',
                width=2
            )
            
            # Draw step number
            text = str(i + 1)
            # Simple text positioning (center of circle)
            draw.text((x - 3, y - 6), text, fill='black')
        
        # Save the overlaid image
        img.save(output_path)
        print(f"Action path overlay saved to: {output_path}")
        
    except Exception as e:
        print(f"Warning: failed to create action path overlay: {e}")


if __name__ == "__main__":
    # Check for macOS and warn about permissions
    import platform
    if platform.system() == "Darwin":
        print("=" * 60)
        print("MACOS PERMISSIONS NOTICE")
        print("=" * 60)
        print("This script requires Accessibility and Screen Recording permissions.")
        print("If you encounter permission errors:")
        print("1. Go to System Preferences > Security & Privacy > Privacy")
        print("2. Add Terminal (or your terminal app) to:")
        print("   - Accessibility")
        print("   - Screen Recording")
        print("3. Restart your terminal and run the script again")
        print("=" * 60)
        print()
    
    parser = argparse.ArgumentParser(description="Benchmark Manipulation Model Evaluation")
    parser.add_argument("--input", required=True, type=str, metavar="FILE",
                        help="Path to the input test messages JSON file.")
    parser.add_argument("--category", type=str, default="manipul_cot",
                        help="Category of manipulation task")
    
    args = parser.parse_args()

    # Prepare per-level results directories with timestamp for each run
    # (Create early so we can write errors to it)
    results_root = Path("benchmark4_manipulation") / "results" / args.category.lower()
    level_key = Path(args.input).name or Path(args.input).stem
    experiment_dir = results_root / level_key
    
    # Create a new run folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = experiment_dir / f"run_{timestamp}"
    
    screenshots_dir = run_dir / "screenshots"
    actions_dir = run_dir / "actions"
    messages_dir = run_dir / "messages"
    raw_dir = run_dir / "raw"
    eval_dir = run_dir / "evaluation"
    for d in (results_root, experiment_dir, run_dir, screenshots_dir, actions_dir, messages_dir, raw_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    error_log_path = run_dir / "error.log"
    
    try:
        input_json = Path(args.input).with_suffix(".json")
        if not input_json.exists():
            raise FileNotFoundError(f"Input Json file not found: {input_json}")
        input_py = Path(args.input).with_suffix(".py")
        if not input_py.exists():
            raise FileNotFoundError(f"Input Python file not found: {input_py}")
        else:
            with open(input_py, 'r') as f:
                instruct_prompt = f.read()
        input_category = args.category.lower()
        if input_category not in allowed_categories:
            raise ValueError(f"Category {input_category} is not supported.")
        
        SYSTEM_PROMPT = get_system_prompt(input_category)
        
        # Parse ground truth data
        ground_truth_data = parse_ground_truth(input_json)
        print(f"Ground truth loaded: {len(ground_truth_data['targets'])} targets, {len(ground_truth_data['templates'])} templates")

        message = get_initial_message(
            SYSTEM_PROMPT=SYSTEM_PROMPT,
            instruct_prompt=instruct_prompt
        )

        model = Qwen3VLLLMWrapper(get_api_key(), get_base_url(), "qwen/qwen3-vl-235b-a22b-instruct")

        iter_idx = 0
        action_history = []  # Track all actions for path overlay
        
        # Pause here until the user triggers the start
        _wait_for_start()
        print("Starting the action loop...")
        
        # Take initial screenshot and add to message
        b64 = _screenshot_to_base64()
        if b64:
            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    }
                ]
            }
            message.append(image_message)
        else:
            raise RuntimeError("Failed to capture initial screenshot")
        
        action_type = "init"
        
        while action_type != "finished":
            # Check if maximum number of actions reached
            if iter_idx >= MAX_ACTIONS:
                print(f"\nMaximum number of actions ({MAX_ACTIONS}) reached. Stopping loop and starting evaluation...")
                b64 = _screenshot_to_base64()
                if b64:
                    try:
                        screenshot_bytes = base64.b64decode(b64)
                        shot_path = screenshots_dir / f"screenshot_{iter_idx:04d}.png"
                        eval_path = eval_dir / "final_screenshot.png"
                        overlay_path = eval_dir / "final_screenshot_with_path.png"
                        with open(eval_path, "wb") as ef:
                            ef.write(screenshot_bytes)
                        with open(shot_path, "wb") as sf:
                            sf.write(screenshot_bytes)
                        
                        # Create action path overlay (reverse the y_offset_factor)
                        _create_action_path_overlay(str(eval_path), action_history, str(overlay_path), y_offset_factor=-441/4800)
                    except Exception as e:
                        print("Warning: failed to write per-iteration screenshot:", e)
                break
            
            # Note: Ctrl+C handling removed as pynput doesn't support is_pressed()
            # User can use Ctrl+C in terminal to interrupt the script

            response = model.generate_model_response(full_messages=message, logging=True) or ""

            parsed_dict = parse_model_response(response)
            action_type = parsed_dict[0].get("action_type", "N/A")
            
            # Get screen size for code generation
            if _have_pyautogui:
                screen_size = pyautogui.size()
                original_image_width, original_image_height = screen_size.width, screen_size.height
            else:
                original_image_width, original_image_height = MY_RESOLUTION
            
            parsed_pyautogui_code = parsing_response_to_pyautogui_code(
                responses=parsed_dict,
                image_height=original_image_height,
                image_width=original_image_width,
                y_offset_factor=-441/4800
            )
            print(f"Generated code:\n{parsed_pyautogui_code}")
            
            # Track generated code for path overlay
            if parsed_pyautogui_code and parsed_pyautogui_code != "DONE":
                action_history.append(parsed_pyautogui_code)
            
            if parsed_pyautogui_code == "DONE":
                b64 = _screenshot_to_base64()
                if b64:
                    try:
                        screenshot_bytes = base64.b64decode(b64)
                        shot_path = screenshots_dir / f"screenshot_{iter_idx:04d}.png"
                        eval_path = eval_dir / "final_screenshot.png"
                        overlay_path = eval_dir / "final_screenshot_with_path.png"
                        with open(eval_path, "wb") as ef:
                            ef.write(screenshot_bytes)
                        with open(shot_path, "wb") as sf:
                            sf.write(screenshot_bytes)
                        
                        # Create action path overlay (reverse the y_offset_factor)
                        _create_action_path_overlay(str(eval_path), action_history, str(overlay_path), y_offset_factor=-441/4800)
                    except Exception as e:
                        print("Warning: failed to write per-iteration screenshot:", e)
                break
            exec(parsed_pyautogui_code)

            # wait a bit for UI to update
            import time
            time.sleep(1.0)
            
            # Append the assistant response to chat history
            assistant_message = {"role": "assistant", "content": parsed_dict[0].get("text")}
            message.append(assistant_message)
            
            # Take a screenshot and append as a user message
            b64 = _screenshot_to_base64()
            if _have_pyautogui:
                print(pyautogui.size())
            
            if b64:
                image_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        }
                    ]
                }
                message.append(image_message)
                if len(message) > MAX_MESSAGE_SIZE:
                    # keep user prompt (first message) and new message + last MAX_MESSAGE_SIZE-1 messages
                    message = [message[0]] + message[-MAX_MESSAGE_SIZE:]
            
            # print message to file for debugging (global debug)
            with open(f"{eval_dir}/final_message.json", "w") as f:
                json.dump(message, f, indent=4)

            # Save per-iteration artifacts into the per-level results folder
            # 1) Save latest screenshot (if present)
            if b64:
                try:
                    screenshot_bytes = base64.b64decode(b64)
                    shot_path = screenshots_dir / f"screenshot_{iter_idx:04d}.png"
                    with open(shot_path, "wb") as sf:
                        sf.write(screenshot_bytes)
                except Exception as e:
                    print("Warning: failed to write per-iteration screenshot:", e)

            # 2) Save parsed actions / assistant response (parsed_dict may be list)
            try:
                actions_path = actions_dir / f"actions_{iter_idx:04d}.json"
                with open(actions_path, "w", encoding="utf-8") as af:
                    json.dump(parsed_dict, af, indent=4, ensure_ascii=False)
            except Exception as e:
                print("Warning: failed to write parsed actions:", e)

            # 3) Save full message history snapshot
            try:
                msg_path = messages_dir / f"messages_{iter_idx:04d}.json"
                with open(msg_path, "w", encoding="utf-8") as mf:
                    json.dump(message, mf, indent=4, ensure_ascii=False)
            except Exception as e:
                print("Warning: failed to write messages snapshot:", e)

            # 4) Save raw model response text
            try:
                raw_path = raw_dir / f"response_{iter_idx:04d}.txt"
                with open(raw_path, "w", encoding="utf-8") as rf:
                    rf.write(response)
            except Exception as e:
                print("Warning: failed to write raw response:", e)

            iter_idx += 1
        
        # After the loop ends, save action history and evaluate the final result
        # Save complete action history
        try:
            action_history_path = eval_dir / "action_history.json"
            with open(action_history_path, "w", encoding="utf-8") as ahf:
                json.dump(action_history, ahf, indent=4, ensure_ascii=False)
            print(f"Action history saved to: {action_history_path}")
        except Exception as e:
            print(f"Warning: failed to write action history: {e}")
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Get the final screenshot for evaluation
        final_screenshot_path = f"{eval_dir}/final_screenshot.png"
        
        if os.path.exists(final_screenshot_path):
            success, total_distance, details, iou_scores, mean_iou = evaluate_response(
                ground_truth_data=ground_truth_data,
                screenshot_path=final_screenshot_path
            )
            
            print(f"\nSuccess: {success}")
            print(f"Total Distance: {total_distance:.2f} pixels")
            print(f"Threshold: {ground_truth_data['threshold']} pixels")
            
            # Display IoU metrics if enabled
            if ground_truth_data.get('use_iou', False) and iou_scores:
                print(f"\nIoU Evaluation: ENABLED")
                print(f"Mean IoU: {mean_iou:.4f}")
                print(f"IoU Threshold: {ground_truth_data.get('iou_threshold', 0.5):.2f}")
            else:
                print(f"\nIoU Evaluation: DISABLED")
            
            print("\nPer-object distances:")
            for obj_name, distance in details.items():
                status = "✓" if distance != float('inf') and distance <= ground_truth_data['threshold'] else "✗"
                dist_str = f"{distance:.2f}" if distance != float('inf') else "NOT FOUND"
                iou_str = f" | IoU: {iou_scores[obj_name]:.4f}" if obj_name in iou_scores else ""
                print(f"  {status} {obj_name}: {dist_str} pixels{iou_str}")
            
            # Save evaluation results
            eval_results = {
                "success": success,
                "total_distance": total_distance,
                "threshold": ground_truth_data['threshold'],
                "details": {k: (v if v != float('inf') else "inf") for k, v in details.items()},
                "use_iou": ground_truth_data.get('use_iou', False),
                "iou_scores": iou_scores,
                "mean_iou": mean_iou,
                "iou_threshold": ground_truth_data.get('iou_threshold', 0.5),
                "ground_truth": ground_truth_data
            }
            
            with open(f"{eval_dir}/results.json", "w") as f:
                json.dump(eval_results, f, indent=4)
            
            print(f"\nEvaluation results saved to: {eval_dir}/results.json")
        else:
            print(f"Warning: Final screenshot not found at {final_screenshot_path}")
            print("Skipping evaluation.")
    
    except Exception as e:
        # Log the error to file
        error_message = f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        error_message += f"Error type: {type(e).__name__}\n"
        error_message += f"Error message: {str(e)}\n\n"
        error_message += "Full traceback:\n"
        error_message += traceback.format_exc()
        
        with open(error_log_path, "w", encoding="utf-8") as error_file:
            error_file.write(error_message)

        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"Error log saved to: {error_log_path}")
        print(f"{'='*60}")
        
        # Re-raise the exception
        raise