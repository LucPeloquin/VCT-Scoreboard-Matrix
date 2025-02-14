import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2
import time

class SnippingTool:
    def __init__(self):
        self.root = tk.Tk()
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        
        # Make the root window full screen and transparent
        self.root.attributes('-fullscreen', True, '-alpha', 0.3)
        self.root.configure(background='grey')
        
        # Bind mouse events
        self.root.bind('<Button-1>', self.on_click)
        self.root.bind('<B1-Motion>', self.on_drag)
        self.root.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Make canvas transparent
        self.canvas.configure(background='grey')
        
    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )

    def on_drag(self, event):
        if self.current_rect:
            self.canvas.coords(
                self.current_rect,
                self.start_x, self.start_y,
                event.x, event.y
            )
            
    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Capture the region
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        
        # Convert to CV2 format
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Save coordinates
        self.roi = {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1
        }
        
        # Save both the image and coordinates
        cv2.imwrite('roi_screenshot.png', frame)
        print(f"ROI coordinates: {self.roi}")
        
        self.root.quit()

def start_snipping():
    snipping_tool = SnippingTool()
    snipping_tool.root.mainloop()
    snipping_tool.root.destroy()
    return snipping_tool.roi

def main():
    print("Snipping tool will start in 3 seconds...")
    time.sleep(3)
    roi = start_snipping()
    print("Snipping complete. You can close the terminal to exit.")

if __name__ == "__main__":
    main() 