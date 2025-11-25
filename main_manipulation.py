import base64
import json
from pathlib import Path
from openai import OpenAI
import argparse
from typing import List
import os
import io
import importlib

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
from utils import get_api_key, encode_image

# Configuration constants
MAX_MESSAGE_SIZE = 10  # maximum number of messages to keep in history (including user and assistant messages)
HOTKEY = "ctrl+shift+s"
MY_RESOLUTION = (1920, 1200)


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


def parse_ground_truth(json_path: Path) -> str:
    """Parse ground truth from JSON file.
    
    Args:
        json_path: Path to the JSON file containing ground truth data
        
    Returns:
        The ground truth value
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("solution", "")


def get_initial_message(image_path: Path, SYSTEM_PROMPT: str, instruct_prompt: str):
    """Get initial message to model before loop.
    
    Args:
        image_path: Path to input image
        SYSTEM_PROMPT: System prompt for the model
        instruct_prompt: Additional instruction prompt
        
    Returns:
        Model response string
    """
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"
    
    message = [
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


def evaluate_response(ground_truth: str, response: str):
    """Evaluate model response against ground truth.
    
    Args:
        ground_truth: Expected correct answer
        response: Model's response
        
    Returns:
        Boolean indicating if response matches ground truth
    """
    return ground_truth == response


def calculate_benchmark_results():
    """Calculate and display benchmark results."""
    pass


def _wait_for_start():
    """Wait for the configured hotkey to be pressed. Fall back to Enter if `keyboard` isn't available."""
    if _have_keyboard:
        print(f"Waiting for hotkey '{HOTKEY}' to start. Press it to continue...")
        try:
            keyboard.wait(HOTKEY)
        except Exception as e:
            print("Warning: keyboard.wait failed:", e)
            input("Press Enter to start the loop...")
    else:
        input("Press Enter to start the loop (install 'keyboard' to use a hotkey)...")


def _screenshot_to_base64():
    """Take a screenshot and return as base64-encoded PNG.
    
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
        buf = io.BytesIO()
        # resize image to reduce size
        # img = img.resize((640, int(640 * img.height / img.width)))
        img.save(buf, format="PNG")
        # save image also to disk for debugging
        with open("benchmark4_manipulation/debug/debug_screenshot.png", "wb") as f:
            f.write(buf.getvalue())
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64
    except Exception as e:
        # If screenshot fails, return empty string so we don't break the loop
        print("Warning: failed to capture screenshot:", e)
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Manipulation Model Evaluation")
    parser.add_argument("--input", required=False, type=str, metavar="FILE",
                        help="Path to the input test messages JSON file.",
                        default="./benchmark4_manipulation/test_messages_uitars_tim.json")
    parser.add_argument("--category", type=str, default="manipulation",
                        help="Category of manipulation task")
    
    args = parser.parse_args()

    input_png = Path(args.input).with_suffix(".png")
    if not input_png.exists():
        raise FileNotFoundError(f"Input image file not found: {input_png}")
    # input_json = Path(args.input).with_suffix(".json")
    # if not input_json.exists():
    #     raise FileNotFoundError(f"Input Json file not found: {input_json}")
    input_py = Path(args.input).with_suffix(".py")
    if not input_py.exists():
        raise FileNotFoundError(f"Input Python file not found: {input_py}")
    else:
        with open(input_py, 'r') as f:
            instruct_prompt = f.read()
    input_category = args.category.lower()
    if input_category not in allowed_categories:
        raise ValueError(f"Category {input_category} is not supported.")
    
    input_file = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    SYSTEM_PROMPT = get_system_prompt(input_category)
    
    API_KEY = get_api_key()
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY
    )

    message = get_initial_message(
        image_path=input_png,
        SYSTEM_PROMPT=SYSTEM_PROMPT,
        instruct_prompt=instruct_prompt
    )

    # Pause here until the user triggers the start
    _wait_for_start()
    
    action_type = "init"
    
    while action_type != "finished":
        # break if ctrl+c is pressed
        if _have_keyboard and keyboard.is_pressed("ctrl+c"):
            print("Ctrl+C detected, exiting...")
            break
        
        chat_completion = client.chat.completions.create(
            model="bytedance/ui-tars-1.5-7b",
            messages=message,
            top_p=None,
            temperature=0.0,
            max_tokens=400,
            stream=True,
            seed=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None
        )
        
        response = ""
        for msg in chat_completion:
            response += msg.choices[0].delta.content if msg.choices[0].delta.content else ""
        
        parsed_dict = parse_model_response(response)
        action_type = parsed_dict[0].get("action_type", "N/A")
        
        # Get screen size for code generation
        if _have_pyautogui:
            screen_size = pyautogui.size()
            original_image_width, original_image_height = screen_size.width, screen_size.height
        else:
            original_image_width, original_image_height = MY_RESOLUTION
        
        parsed_pyautogui_code = parsing_response_to_pydirectinput_code(
            responses=parsed_dict,
            image_height=original_image_height,
            image_width=original_image_width
        )
        print(f"Generated code:\n{parsed_pyautogui_code}")
        
        exec(parsed_pyautogui_code)
        
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
        
        # print message to file for debugging
        with open("benchmark4_manipulation/debug/debug_message.json", "w") as f:
            json.dump(message, f, indent=4)