import base64
import json
from pathlib import Path
from openai import OpenAI
import argparse
from typing import List
import os
import io
import importlib
from datetime import datetime
import time
import traceback

from benchmark4_manipulation.system_prompts.system_prompt_uitars import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_UITARS
from benchmark4_manipulation.system_prompts.system_prompt_gemini import \
    SYSTEM_PROMPT as SYSTEM_PROMPT_GEMINI

from src.benchmark_base import get_api_keys, get_base_url

try:
    import pyautogui
    _have_pyautogui = True
except Exception:
    _have_pyautogui = False

try:
    from PIL import ImageGrab
    _have_imagegrab = True
except Exception:
    _have_imagegrab = False

try:
    keyboard = importlib.import_module("keyboard")
    _have_keyboard = True
except Exception:
    keyboard = None
    _have_keyboard = False

from ui_tars_1_5_7B.action_parser import parse_action_to_structure_output, parsing_response_to_pyautogui_code, parsing_response_to_pydirectinput_code
from utils import get_api_key, encode_image, DistanceTracker, VisualLocator, calculate_intersection
from PIL import Image, ImageDraw

# Configuration constants
MAX_MESSAGE_SIZE = 5  # maximum number of messages to keep in history (including user and assistant messages)
HOTKEY = "f9"
MY_RESOLUTION = (1920, 1080)
pyautogui.FAILSAFE = False  # Disable failsafe to allow clicks at corners
max_actions = 5


def get_system_prompt(model_name: str):
    """Get the appropriate system prompt based on model input."""
    if model_name in openrouter_model_list:
        if 'ui-tars' in model_name:
            return SYSTEM_PROMPT_UITARS
        elif 'qwen' in model_name:
            return SYSTEM_PROMPT_GEMINI
    elif model_name in openai_model_list:
        return False
    elif model_name in gemini_model_list:
        return SYSTEM_PROMPT_GEMINI
    elif model_name in hyperbolic_model_list:
        return SYSTEM_PROMPT_UITARS
    else:
        pass
    return ""

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
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": instruct_prompt
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
        model_type="gpt"
    )
    print(f"Parsed action: {parsed_dict}")
    
    return parsed_dict

def select_model(model_name, message):
    if model_name:
        print(f"Using model: {model_name}")

        if model_name in openrouter_model_list:
            API_KEY = get_api_keys('OPENROUTER_API_KEY')
            BASE_URL = get_base_url('BASE_URL')
            if 'ui-tars' in model_name:
                client = OpenAI(
                    base_url=BASE_URL,
                    api_key=API_KEY
                )
                chat_completion = client.chat.completions.create(
                    model="bytedance/ui-tars-1.5-7b",
                    messages=message,
                    temperature=0.0,
                    max_tokens=4000,
                    stream=True,
                )
            elif 'qwen3' in model_name:
                client = OpenAI(
                    base_url=BASE_URL,
                    api_key=API_KEY
                )
                chat_completion = client.chat.completions.create(
                    model="qwen/qwen3-vl-235b-a22b-instruct",
                    messages=message,
                    temperature=0.0,
                    max_tokens=4000,
                    stream=True,
                )
            else:
                raise ValueError(f"Model not implemented: {model_name}")
        elif model_name in openai_model_list:
            API_KEY = get_api_keys('OPENAI_API_KEY')
            client = OpenAI(
                api_key=API_KEY
            )
            chat_completion = client.chat.completions.create(
                model="gpt-5-mini",
                messages=message,
            )
        elif model_name in gemini_model_list:
            API_KEY = get_api_keys('GEMINI_API_KEY')
            BASE_URL = get_base_url('GEMINI_BASE_URL')
            client = OpenAI(
                base_url=BASE_URL,
                api_key=API_KEY
            )
            chat_completion = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=message,
                temperature=0.0,
                max_tokens=4000,
                stream=True,
            )

        elif model_name in hyperbolic_model_list:
            API_KEY = get_api_keys('HYPERBOLIC_API_KEY')
            BASE_URL = get_base_url('HYPERBOLIC_BASE_URL')
            if 'Qwen2.5' in model_name:
                client = OpenAI(
                    base_url=BASE_URL,
                    api_key=API_KEY
                )
                chat_completion = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-7B-Instruct",
                    messages=message,
                    temperature=0.0,
                    max_tokens=4000,
                    stream=True,
                )
            else:
                raise ValueError(f"Model not implemented: {model_name}")
        else:
            raise ValueError(f"Model not implemented: {model_name}")

    return chat_completion


def _wait_for_start():
    """Wait for the configured hotkey to be pressed. Fall back to Enter if `keyboard` isn't available."""
    if _have_keyboard:
        print(f"Waiting for hotkey '{HOTKEY}' to start. Press it to continue...")
        try:
            # keyboard.wait(HOTKEY)
            pass
        except Exception as e:
            print("Warning: keyboard.wait failed:", e)
            input("Press Enter to start the loop...")
    else:
        input("Press Enter to start the loop (install 'keyboard' to use a hotkey)...")


def _apply_grey_overlay(img: Image.Image, regions: List[List[int]]) -> Image.Image:
    """Apply grey overlay to specified regions of an image.
    
    Args:
        img: PIL Image to modify
        regions: List of regions [x1, y1, x2, y2] to grey out
        
    Returns:
        Modified PIL Image with grey overlays
    """
    if not regions:
        return img
    
    # Create a copy to avoid modifying the original
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    for region in regions:
        if len(region) == 4:
            x1, y1, x2, y2 = region
            # Draw opaque grey rectangle
            draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 255))
    
    return img_copy


def _screenshot_to_base64(grey_out_regions: List[List[int]] = None):
    """Take a screenshot and return as base64-encoded PNG.
    
    Args:
        grey_out_regions: Optional list of regions [x1, y1, x2, y2] to grey out
    
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
        
        # Apply grey overlay to specified regions
        if grey_out_regions:
            img = _apply_grey_overlay(img, grey_out_regions)
        
        buf = io.BytesIO()
        # resize image to reduce size
        # img = img.resize((640, int(640 * img.height / img.width)))
        img.save(buf, format="PNG")
        # save image also to disk for debugging
        # with open(f"{eval_dir}/final_screenshot.png", "wb") as f:
        #     f.write(buf.getvalue())
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
    openrouter_model_list = ['qwen/qwen3-vl-235b-a22b-instruct',
                                  'qwen/qwen3-vl-8b-instruct',
                                  'bytedance/ui-tars-1.5-7b']
    openai_model_list = ['gpt-5-mini']
    gemini_model_list = ['gemini-2.5-flash']
    hyperbolic_model_list = ['Qwen/Qwen2.5-VL-7B-Instruct']

    parser = argparse.ArgumentParser(description="Benchmark Manipulation Model Evaluation")
    parser.add_argument("--input", required=True, type=str, metavar="FILE",
                        help="Path to the input test messages JSON file.")
    parser.add_argument(
        "--model",
        type=str,
        choices=openrouter_model_list + gemini_model_list + openai_model_list + hyperbolic_model_list,
        required=False,
        help="Optional model name override.",
    )

    args = parser.parse_args()

    # Prepare per-level results directories with timestamp for each run
    # (Create early so we can write errors to it)
    results_root = Path("benchmark4_manipulation") / "results" / args.model
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
    action_history = []  # Track all actions for path overlay
    
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

        model_name = args.model

        SYSTEM_PROMPT = get_system_prompt(model_name)

        message = get_initial_message(
            SYSTEM_PROMPT=SYSTEM_PROMPT,
            instruct_prompt=instruct_prompt
        )

        iter_idx = 0
        latest_response = ""  # Track latest model response for error reporting
        
        # Pause here until the user triggers the start
        _wait_for_start()
        print("Starting the action loop...")
        time.sleep(4)
        
        # Take initial screenshot and add to message
        b64 = _screenshot_to_base64(grey_out_regions=None)
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
            if iter_idx >= max_actions:
                pyautogui.click(0, 0)
                print(f"\nMaximum number of actions ({max_actions}) reached. Stopping loop and starting evaluation...")
                b64 = _screenshot_to_base64(grey_out_regions=None)
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
            
            # break if ctrl+c is pressed
            if _have_keyboard and keyboard.is_pressed("ctrl+c"):
                print("Ctrl+C detected, exiting...")
                pyautogui.click(0, 0)
                b64 = _screenshot_to_base64(grey_out_regions=None)
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

            chat_completion = select_model(model_name=args.model, message=message)

            response = ""
            for msg in chat_completion:
                response += msg.choices[0].delta.content if msg.choices[0].delta.content else ""
            
            # Track latest response for error reporting
            latest_response = response
            
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
                pyautogui.click(0, 0)
                b64 = _screenshot_to_base64(grey_out_regions=None)
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
            b64 = _screenshot_to_base64(grey_out_regions=None)
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
            # Add this at the end of the script

        except Exception as e:
            print(f"Warning: failed to write action history: {e}")
            # Add this at the end of the script
    
    except Exception as e:
        # Log the error to file
        error_message = f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        error_message += f"Error type: {type(e).__name__}\n"
        error_message += f"Error message: {str(e)}\n\n"
        
        # Include latest model response if available
        if 'latest_response' in locals() and latest_response:
            error_message += f"Latest model response:\n{latest_response}\n\n"
        
        error_message += "Full traceback:\n"
        error_message += traceback.format_exc()
        
        with open(error_log_path, "w", encoding="utf-8") as error_file:
            error_file.write(error_message)
        
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        
        # Print latest model response
        if 'latest_response' in locals() and latest_response:
            print(f"\nLatest model response:\n{latest_response}")
        
        print(f"Error log saved to: {error_log_path}")
        print(f"{'='*60}")

        pyautogui.click(0, 0)
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
        
        # Re-raise the exception
        raise