from abc import abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from pprint import pprint
from typing import override

from openai import OpenAI

from .image_processing import b64encode_image, convert_to_webp, get_image_dimensions, pad_image

@dataclass
class Point:
    x: int
    y: int
    
    def euclidian_distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

@dataclass
class BoundingBox:
    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def center(self) -> Point:
        center_x = (self.x_min + self.x_max) // 2
        center_y = (self.y_min + self.y_max) // 2
        return Point(x=center_x, y=center_y)
    def width(self): return abs(self.x_max - self.x_min)
    def height(self): return abs(self.y_max - self.y_min)
    def area(self): return self.width() * self.height()
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        xA = max(self.x_min, other.x_min)
        yA = max(self.y_min, other.y_min)
        xB = min(self.x_max, other.x_max)
        yB = min(self.y_max, other.y_max)
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight
        if interArea == 0:
            return 0.0
        unionArea = self.area() + other.area() - interArea
        if unionArea == 0:
            return 0.0
        iou = interArea / unionArea
        return iou
    def bbox_list(self) -> list[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def __str__(self):
        return f"BoundingBox(label={self.label}, x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"


class LLMWrapperBase:
    INVALID_BBOX = ('', [])

    def __init__(self, api_key: str, base_url: str | None, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
    
    def encode_image(self, image_path: Path) -> str:
        if False: # webp disabled for now
            width, height = get_image_dimensions(image_path)
        #webp has effects only for large images
            if width*height > 1024*1024:
                webp_image_path = convert_to_webp(image_path)
                return b64encode_image(webp_image_path)
        return b64encode_image(image_path)

    def parse_response_text(self, response: str) -> str:
        return response.strip()
    
    @abstractmethod
    def parse_response_bbox(self, response: str, image_width: int, image_height: int) -> BoundingBox|None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def parse_response_bboxes(self, response: str, image_width: int, image_height: int) -> list[BoundingBox]:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def parse_response_point(self, response: str) -> tuple[int, int]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_model_response(self, image_path:Path, system_prompt:str, additional_user_prompt="", logging=False):
        encoded_image = self.encode_image(image_path)
        data_url = f"data:image/webp;base64,{encoded_image}"
        user_prompt = []
        if additional_user_prompt:
            user_prompt.append({"type": "text", "text": additional_user_prompt})
        user_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        if logging:
            print("Sending request to model...")
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.1, timeout=60, max_tokens=4096)
        part_name = response.choices[0].message.content
        if logging:
            pprint(response.model_dump())
            print(f"Model Response: {part_name}")
        return part_name

class Qwen3VLLLMWrapper(LLMWrapperBase):
    def __init__(self, api_key: str, base_url: str|None, model_name: str):
        super().__init__(api_key, base_url, model_name)
    
    def encode_image(self, image_path: Path) -> str:
        return super().encode_image(pad_image(image_path, 28))
    
    @override
    def parse_response_bbox(self, response: str, image_width: int, image_height: int) -> BoundingBox|None:
        response_text = response.strip().replace('```json', '').replace('```', '')
        try:
            bbox_data = json.loads(response_text)
            if not isinstance(bbox_data, dict):
                print("Response JSON is not an object.")
                print("Raw response:", response_text)
                return None
            bbox = bbox_data.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return None
            
            label = bbox_data.get("label")
            if not isinstance(label, str) or not label:
                print("No label given.")
                return None
            
            # Convert normalized coordinates (0-1000) to absolute pixels
            x_min, y_min, x_max, y_max = bbox
            x_min_px = int((x_min / 1000.0) * image_width)
            y_min_px = int((y_min / 1000.0) * image_height)
            x_max_px = int((x_max / 1000.0) * image_width)
            y_max_px = int((y_max / 1000.0) * image_height)

            bounding_box = BoundingBox(
                label=label.upper(),
                x_min=x_min_px,
                y_min=y_min_px,
                x_max=x_max_px,
                y_max=y_max_px
            )
            
            return bounding_box
        except json.JSONDecodeError:
            print("Failed to parse JSON from model response.")
            print("Raw response:", response_text)
            return None
        
    @override
    def parse_response_bboxes(self, response: str, image_width: int, image_height: int) -> list[BoundingBox]:
        response_text = response.strip().replace('```json', '').replace('```', '')
        results = []
        try:
            bbox_list = json.loads(response_text)
            if not isinstance(bbox_list, list):
                raise ValueError("Invalid bounding boxes format.")
            
            for bbox_data in bbox_list:
                if not isinstance(bbox_data, dict):
                    print("Invalid bounding box entry, skipping.", bbox_data)
                    continue
                bbox = bbox_data.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print("Invalid bounding box format, skipping.", bbox)
                    continue
                
                label = bbox_data.get("label")
                if not isinstance(label, str) or not label:
                    print("No label given, skipping.")
                    continue
                
                # Convert normalized coordinates (0-1000) to absolute pixels
                x_min, y_min, x_max, y_max = bbox
                x_min_px = int((x_min / 1000.0) * image_width)
                y_min_px = int((y_min / 1000.0) * image_height)
                x_max_px = int((x_max / 1000.0) * image_width)
                y_max_px = int((y_max / 1000.0) * image_height)
                
                results.append(BoundingBox(
                    label=label.upper(),
                    x_min=x_min_px,
                    y_min=y_min_px,
                    x_max=x_max_px,
                    y_max=y_max_px
                ))
            return results
        
        except json.JSONDecodeError:
            print("Failed to parse JSON from model response.")
            print("Raw response:", response_text)
            return []

class UiTarsLLMWrapper(LLMWrapperBase):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)

    def encode_image(self, image_path: Path) -> str:
        return super().encode_image(pad_image(image_path, 28))

    @override
    def parse_response_bbox(self, response: str, image_width: int, image_height: int) -> BoundingBox|None:
        response_text = response.strip()
        for line in response_text.splitlines():
            if line.startswith("Action:"):
                # regex Action: drag(start_point='(130,150)', end_point='(170,200)')
                import re
                match = re.search(r"drag.*\((\d+),(\d+)\).*\((\d+),(\d+)\)", line)
                if match:
                    x1 = int(match.group(1))
                    y1 = int(match.group(2))
                    x2 = int(match.group(3))
                    y2 = int(match.group(4))
                    return BoundingBox(
                        label="DRAG",
                        x_min=min(x1, x2),
                        y_min=min(y1, y2),
                        x_max=max(x1, x2),
                        y_max=max(y1, y2)
                    )
        return None
    
    @override
    def parse_response_point(self, response: str) -> tuple[int, int]:
        response_text = response.strip()
        for line in response_text.splitlines():
            if line.startswith("Action:"):
                # regex Action: click(start_box='(230,131)')
                import re
                match = re.search(r"click\(.*='\(\s*(\d+)\s*,\s*(\d+)\s*\)'", line)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))
                    return (x, y)
        return (-1, -1)


class OpenAILLMWrapper(Qwen3VLLLMWrapper):
    def __init__(self, api_key: str, base_url: None, model_name: str):
        super().__init__(api_key, base_url, model_name)

    @override
    def parse_response_point(self, response: str) -> tuple[int, int]:
        pass

    @override
    def generate_model_response(self, image_path:Path, system_prompt:str, additional_user_prompt="", logging=False):
        encoded_image = self.encode_image(image_path)
        data_url = f"data:image/webp;base64,{encoded_image}"
        user_prompt = []
        if additional_user_prompt:
            user_prompt.append({"type": "text", "text": additional_user_prompt})
        user_prompt.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        if logging:
            print("Sending request to model...")
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, timeout=60)
        part_name = response.choices[0].message.content
        if logging:
            pprint(response.model_dump())
            print(f"Model Response: {part_name}")
        return part_name

class GeminiLLMWrapper(Qwen3VLLLMWrapper):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        super().__init__(api_key, base_url, model_name)

    def encode_image(self, image_path: Path) -> str:
        return super().encode_image(pad_image(image_path, 28))

    @override
    def parse_response_point(self, response: str) -> tuple[int, int]:
        pass


class Qwen25VLLLMWrapper(LLMWrapperBase):
    def __init__(self, api_key: str, base_url: str|None, model_name: str):
        super().__init__(api_key, base_url, model_name)
    
    def encode_image(self, image_path: Path) -> str:
        return super().encode_image(pad_image(image_path, 28))

    @override
    def generate_model_response(self, image_path:Path, system_prompt:str, additional_user_prompt="", logging=False):
        # time.sleep(3) # rate limit: 60 per minute
        return super().generate_model_response(image_path, system_prompt, additional_user_prompt, logging)
    
    @override
    def parse_response_bbox(self, response: str, image_width: int, image_height: int) -> BoundingBox|None:
        '''Uses absolute coordinates in pixels
        '''
        response_text = response.strip().replace('```json', '').replace('```', '')
        try:
            bbox_data = json.loads(response_text)
            if not isinstance(bbox_data, dict):
                print("Response JSON is not an object.")
                print("Raw response:", response_text)
                return None
            bbox = bbox_data.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return None
            
            label = bbox_data.get("label")
            if not isinstance(label, str) or not label:
                print("No label given.")
                return None
            
            x_min, y_min, x_max, y_max = bbox
            x_min_px = x_min
            y_min_px = y_min
            x_max_px = x_max
            y_max_px = y_max

            bounding_box = BoundingBox(
                label=label.upper(),
                x_min=x_min_px,
                y_min=y_min_px,
                x_max=x_max_px,
                y_max=y_max_px
            )
            
            return bounding_box
        except json.JSONDecodeError:
            print("Failed to parse JSON from model response.")
            print("Raw response:", response_text)
            return None
        
    @override
    def parse_response_bboxes(self, response: str, image_width: int, image_height: int) -> list[BoundingBox]:
        response_text = response.strip().replace('```json', '').replace('```', '')
        results = []
        try:
            bbox_list = json.loads(response_text)
            if not isinstance(bbox_list, list):
                raise ValueError("Invalid bounding boxes format.")
            
            for bbox_data in bbox_list:
                if not isinstance(bbox_data, dict):
                    print("Invalid bounding box entry, skipping.", bbox_data)
                    continue
                bbox = bbox_data.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print("Invalid bounding box format, skipping.", bbox)
                    continue
                
                label = bbox_data.get("label")
                if not isinstance(label, str) or not label:
                    print("No label given, skipping.")
                    continue
                
                x_min, y_min, x_max, y_max = bbox
                x_min_px = x_min
                y_min_px = y_min
                x_max_px = x_max
                y_max_px = y_max
                
                results.append(BoundingBox(
                    label=label.upper(),
                    x_min=x_min_px,
                    y_min=y_min_px,
                    x_max=x_max_px,
                    y_max=y_max_px
                ))
            return results
        
        except json.JSONDecodeError:
            print("Failed to parse JSON from model response.")
            print("Raw response:", response_text)
            return []