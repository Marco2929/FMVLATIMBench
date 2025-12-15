#!/usr/bin/env python3
"""
Simple GUI tool for drawing bounding boxes on images and saving them to JSON files.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import json
import os
import argparse
from pathlib import Path


class BBoxAnnotator:
    def __init__(self, root, initial_dir=None):
        self.root = root
        self.root.title("Bounding Box Annotator")
        
        # Variables
        self.input_dir = initial_dir
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_image_path = None
        self.photo = None
        self.canvas_image = None
        self.display_scale = 1.0
        
        # Bounding box state
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.bbox = None
        self.crosshair_h = None
        self.crosshair_v = None
        
        # Setup UI
        self.setup_ui()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        self.resize_timer = None
        
        # Load initial directory if provided
        if self.input_dir:
            self.dir_label.config(text=f"...{self.input_dir[-40:]}" if len(self.input_dir) > 40 else self.input_dir)
            self.load_image_files()
    
    def on_window_resize(self, event):
        # Debounce resize events - only reload after 200ms of no resize
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(200, self.reload_current_image)
    
    def reload_current_image(self):
        if self.current_image:
            self.load_current_image()
        
    def setup_ui(self):
        # Top frame for controls
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Directory selection
        tk.Button(top_frame, text="Select Directory", command=self.select_directory).pack(side=tk.LEFT, padx=5)
        self.dir_label = tk.Label(top_frame, text="No directory selected")
        self.dir_label.pack(side=tk.LEFT, padx=5)
        
        # Task description frame
        task_frame = tk.Frame(self.root)
        task_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(task_frame, text="Task:", bg='lightyellow').pack(side=tk.LEFT, padx=(10, 5), pady=5)
        
        self.task_text = tk.Text(task_frame, height=2, wrap=tk.WORD, bg='lightyellow', relief=tk.SUNKEN, padx=5, pady=5)
        self.task_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)
        
        # Middle frame for canvas
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Leave>', self.on_mouse_leave)
        
        # Bottom frame for buttons
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Navigation and action buttons
        tk.Button(bottom_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.image_label = tk.Label(bottom_frame, text="No image loaded")
        self.image_label.pack(side=tk.LEFT, padx=20)
        
        tk.Button(bottom_frame, text="Clear", command=self.clear_bbox, bg='orange').pack(side=tk.RIGHT, padx=5)
        tk.Button(bottom_frame, text="Save", command=self.save_bbox, bg='lightgreen').pack(side=tk.RIGHT, padx=5)
        tk.Button(bottom_frame, text="Skip (No BBox)", command=self.skip_bbox, bg='lightblue').pack(side=tk.RIGHT, padx=5)
        
    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir = directory
            self.dir_label.config(text=f"...{directory[-40:]}" if len(directory) > 40 else directory)
            self.load_image_files()
            
    def load_image_files(self):
        if not self.input_dir:
            return
            
        # Find all PNG files in directory, excluding _solution.png and *.g.* files
        path = Path(self.input_dir)
        self.image_files = sorted([
            f for f in path.glob('*.png') 
            if not f.name.endswith('_solution.png') and '.g.' not in f.name
        ])
        
        if self.image_files:
            self.current_index = 0
            self.load_current_image()
        else:
            messagebox.showwarning("No Images", "No PNG files found in the selected directory.")
            
    def load_current_image(self):
        if not self.image_files:
            return
            
        self.current_image_path = self.image_files[self.current_index]
        self.current_image = Image.open(self.current_image_path)
        
        # Get available canvas space (subtract some padding)
        self.canvas.update_idletasks()
        self.max_canvas_width = max(self.canvas.winfo_width() - 20, 400)
        self.max_canvas_height = max(self.canvas.winfo_height() - 20, 300)
        
        # Calculate scale to fit within available space while preserving aspect ratio
        width_scale = self.max_canvas_width / self.current_image.width
        height_scale = self.max_canvas_height / self.current_image.height
        self.display_scale = min(width_scale, height_scale)  # Scale to fill view
        
        display_width = int(self.current_image.width * self.display_scale)
        display_height = int(self.current_image.height * self.display_scale)
        
        # Create scaled image for display
        display_image = self.current_image.resize((display_width, display_height), Image.LANCZOS)
        
        # Display image (canvas size is managed by pack)
        self.photo = ImageTk.PhotoImage(display_image)
        self.canvas.delete('all')
        self.canvas_image = self.canvas.create_image(
            self.max_canvas_width // 2, self.max_canvas_height // 2, anchor=tk.CENTER, image=self.photo
        )
        
        # Reset bbox
        self.bbox = None
        self.rect_id = None
        
        # Load and display task description and existing solution
        self.load_task_description()
        self.load_existing_solution()
        
        # Update label
        self.image_label.config(text=f"Image {self.current_index + 1}/{len(self.image_files)}: {self.current_image_path.name}")
    
    def load_task_description(self):
        """Load task description from JSON file"""
        if not self.current_image_path:
            return
        
        base_name = self.current_image_path.stem
        json_path = self.current_image_path.parent / f"{base_name}.json"
        
        # Clear existing text
        self.task_text.delete('1.0', tk.END)
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                task_desc = data.get('TASK_DESCRIPTION', 'No task description available')
                self.task_text.insert('1.0', task_desc)
            except Exception as e:
                self.task_text.insert('1.0', f"Error loading task description: {str(e)}")
        else:
            self.task_text.insert('1.0', "JSON file not found")
    
    def load_existing_solution(self):
        """Load and display existing solution bbox if it exists"""
        if not self.current_image_path:
            return
        
        base_name = self.current_image_path.stem
        json_path = self.current_image_path.parent / f"{base_name}.json"
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if solution exists and has valid bbox
                if 'solution' in data and isinstance(data['solution'], dict):
                    solution = data['solution']
                    if 'bbox' in solution and isinstance(solution['bbox'], list) and len(solution['bbox']) == 4:
                        bbox = solution['bbox']
                        # Validate bbox values are numbers
                        if all(isinstance(x, (int, float)) for x in bbox):
                            self.bbox = [int(x) for x in bbox]
                            self.draw_bbox_on_canvas()
            except Exception as e:
                # Silently ignore errors when loading existing solution
                pass
    
    def draw_bbox_on_canvas(self):
        """Draw the current bbox on the canvas"""
        if not self.bbox or not self.current_image:
            return
        
        # Clear any existing rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        # Convert bbox from original image coordinates to display coordinates
        x1, y1, x2, y2 = self.bbox
        
        # Scale to display coordinates
        x1_display = x1 * self.display_scale
        y1_display = y1 * self.display_scale
        x2_display = x2 * self.display_scale
        y2_display = y2 * self.display_scale
        
        # Account for image centering offset
        display_width = int(self.current_image.width * self.display_scale)
        display_height = int(self.current_image.height * self.display_scale)
        offset_x = (self.max_canvas_width - display_width) / 2.0
        offset_y = (self.max_canvas_height - display_height) / 2.0
        
        x1_display += offset_x
        y1_display += offset_y
        x2_display += offset_x
        y2_display += offset_y
        
        # Draw rectangle
        self.rect_id = self.canvas.create_rectangle(
            x1_display, y1_display, x2_display, y2_display,
            outline='red', width=2
        )
    
    def on_mouse_move(self, event):
        """Draw crosshair lines at mouse position"""
        if not self.current_image:
            return
        
        # Remove old crosshair lines
        if self.crosshair_h:
            self.canvas.delete(self.crosshair_h)
        if self.crosshair_v:
            self.canvas.delete(self.crosshair_v)
        
        # Draw new crosshair lines
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.crosshair_h = self.canvas.create_line(
            0, event.y, canvas_width, event.y,
            fill='yellow', width=1, dash=(2, 2)
        )
        self.crosshair_v = self.canvas.create_line(
            event.x, 0, event.x, canvas_height,
            fill='yellow', width=1, dash=(2, 2)
        )
    
    def on_mouse_leave(self, event):
        """Remove crosshair lines when mouse leaves canvas"""
        if self.crosshair_h:
            self.canvas.delete(self.crosshair_h)
            self.crosshair_h = None
        if self.crosshair_v:
            self.canvas.delete(self.crosshair_v)
            self.crosshair_v = None
    
    def on_mouse_down(self, event):
        if not self.current_image:
            return
        
        # Store the canvas position
        self.start_x = event.x
        self.start_y = event.y
            
        # Clear previous rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
    def on_mouse_drag(self, event):
        if not self.current_image or self.start_x is None:
            return
            
        # Remove previous rectangle
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            
        # Draw new rectangle
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )
        
    def on_mouse_up(self, event):
        if not self.current_image or self.start_x is None:
            return
        
        # Use the same dimensions as where the image was centered
        display_width = int(self.current_image.width * self.display_scale)
        display_height = int(self.current_image.height * self.display_scale)
        
        # Calculate image offset (centered) - use the same max_canvas values used during centering
        offset_x = (self.max_canvas_width - display_width) / 2.0
        offset_y = (self.max_canvas_height - display_height) / 2.0
            
        # Calculate bbox coordinates in display space, adjusted for image offset
        x1 = min(self.start_x, event.x) - offset_x
        y1 = min(self.start_y, event.y) - offset_y
        x2 = max(self.start_x, event.x) - offset_x
        y2 = max(self.start_y, event.y) - offset_y
        
        # Convert from display coordinates to original image coordinates
        x1 = x1 / self.display_scale
        y1 = y1 / self.display_scale
        x2 = x2 / self.display_scale
        y2 = y2 / self.display_scale
        
        # Round to nearest integer for pixel coordinates
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, self.current_image.width))
        y1 = max(0, min(y1, self.current_image.height))
        x2 = max(0, min(x2, self.current_image.width))
        y2 = max(0, min(y2, self.current_image.height))
        
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]
        
    def clear_bbox(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.bbox = None
        self.start_x = None
        self.start_y = None
        
    def save_bbox(self):
        if not self.current_image_path:
            return
            
        # Get base name without extension
        base_name = self.current_image_path.stem
        json_path = self.current_image_path.parent / f"{base_name}.json"
        
        if not json_path.exists():
            messagebox.showerror("Error", f"JSON file not found: {json_path.name}")
            return
            
        try:
            # Load existing JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Update task description from text widget
            new_task = self.task_text.get('1.0', tk.END).strip()
            data['TASK_DESCRIPTION'] = new_task
            
            # Add solution - use None if no bbox drawn
            if self.bbox:
                data['solution'] = {
                    'label': '',
                    'bbox': self.bbox
                }
            else:
                data['solution'] = None
            
            # Save JSON
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save image with bbox only if bbox exists
            if self.bbox:
                img_with_bbox = self.current_image.copy()
                draw = ImageDraw.Draw(img_with_bbox)
                draw.rectangle(self.bbox, outline='red', width=2)
                
                solution_path = self.current_image_path.parent / f"{base_name}_solution.png"
                img_with_bbox.save(solution_path)
            
            # Clear bbox and move to next image
            self.clear_bbox()
            self.next_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def skip_bbox(self):
        """Save with no bbox (solution = None) and move to next image"""
        self.bbox = None
        self.save_bbox()
            
    def previous_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_current_image()
        
    def next_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_current_image()


def main():
    parser = argparse.ArgumentParser(description='Bounding Box Annotation Tool')
    parser.add_argument('--path', type=str, default=None,
                        help='Initial directory path containing images and JSON files')
    args = parser.parse_args()
    
    # Default to script location if no path provided
    initial_dir = args.path if args.path else str(Path(__file__).parent)
    
    root = tk.Tk()
    root.geometry('1400x1000')
    app = BBoxAnnotator(root, initial_dir=initial_dir)
    root.mainloop()


if __name__ == '__main__':
    main()
