import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance

class ImageOverlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Overlay with Scaling")

        # Load background and overlay images
        self.bg_image = None
        self.overlay_image = None
        self.overlay_tk = None

        # Create canvas
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Add buttons to load images
        self.load_bg_button = tk.Button(root, text="Load Background Image", command=self.load_background_image)
        self.load_bg_button.pack(side='left')

        self.load_overlay_button = tk.Button(root, text="Load Overlay Image", command=self.load_overlay_image)
        self.load_overlay_button.pack(side='right')

        # Initialize overlay position and scale
        self.overlay_pos = (0, 0)
        self.overlay_scale = 1.0

        # Bind mouse events for dragging and scaling
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_overlay)
        self.canvas.bind("<ButtonPress-3>", self.start_scale)
        self.canvas.bind("<B3-Motion>", self.scale_overlay)

        # Bind Ctrl+S to print the scaling factor
        self.root.bind("<Control-s>", self.print_scale_factor)

    def load_background_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.bg_image = Image.open(file_path)
            self.bg_tk = ImageTk.PhotoImage(self.bg_image)
            self.canvas.create_image(0, 0, anchor='nw', image=self.bg_tk)

    def load_overlay_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.overlay_image = Image.open(file_path)
            self.update_overlay()

    def update_overlay(self):
        if self.bg_image and self.overlay_image:
            overlay_resized = self.overlay_image.resize(
                (int(self.overlay_image.width * self.overlay_scale), int(self.overlay_image.height * self.overlay_scale)), Image.LANCZOS)
            overlay_resized = ImageEnhance.Brightness(overlay_resized).enhance(0.5)  # 50% transparency
            self.overlay_tk = ImageTk.PhotoImage(overlay_resized)

            # Clear previous overlay
            self.canvas.delete("overlay")
            self.canvas.create_image(0, 0, anchor='nw', image=self.bg_tk)
            self.canvas.create_image(self.overlay_pos[0], self.overlay_pos[1], anchor='nw', image=self.overlay_tk, tags="overlay")

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag_overlay(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.overlay_pos = (self.overlay_pos[0] + dx, self.overlay_pos[1] + dy)
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_overlay()

    def start_scale(self, event):
        self.scale_start_x = event.x
        self.scale_start_y = event.y

    def scale_overlay(self, event):
        dx = event.x - self.scale_start_x
        dy = event.y - self.scale_start_y
        self.overlay_scale += (dx + dy) / 200.0  # Adjust scaling sensitivity
        self.overlay_scale = max(0.1, self.overlay_scale)  # Prevent negative or zero scale
        self.scale_start_x = event.x
        self.scale_start_y = event.y
        self.update_overlay()

    def print_scale_factor(self, event):
        print(f"Current overlay scale factor: {self.overlay_scale:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageOverlayApp(root)
    root.mainloop() 