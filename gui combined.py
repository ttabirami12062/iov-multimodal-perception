import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from ttkthemes import ThemedStyle  # Import ThemedStyle from ttkthemes

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A YOLO-based Brain Tumor detection")
        self.root.geometry("800x800")  # Set window size

        style = ThemedStyle(root)
        style.set_theme("equilux")  # Set the theme for the whole app

        title_label = tk.Label(root, text="A YOLO-based Brain Tumor detection ", font=("Helvetica", 20))
        title_label.pack(pady=20)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.canvas_1 = tk.Canvas(root, bg="white", width=400, height=600)
        self.canvas_1.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_2 = tk.Canvas(root, bg="white", width=400, height=600)
        self.canvas_2.pack(side=tk.LEFT, padx=10, pady=10)

        self.loaded_image = None
        self.loaded_cv_image = None

        # Load YOLO models
        self.model_1 = YOLO("runs/segment/train/weights/best.pt")
        self.model_2 = YOLO("yolov8m.pt")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.loaded_image = Image.open(file_path)
            self.loaded_image = self.loaded_image.resize((800, 600))  # Resize image if needed
            self.loaded_cv_image = cv2.cvtColor(np.array(self.loaded_image), cv2.COLOR_RGB2BGR)
            self.display_cv_image()

    def detect_objects(self):
        if self.loaded_cv_image is not None:
            if self.model_1 is not None and self.model_2 is not None:
                # Run inference with the first YOLO model
                results_1 = self.model_1(source=self.loaded_cv_image)
                res_plotted_1 = results_1[0].plot()

                # Run inference with the second YOLO model
                results_2 = self.model_2(source=self.loaded_cv_image)
                res_plotted_2 = results_2[0].plot()

                # Display the predictions from both models on the canvas
                self.display_cv_image(res_plotted_1, self.canvas_1)
                self.display_cv_image(res_plotted_2, self.canvas_1)
            else:
                self.display_message("Models not loaded!")

    def display_cv_image(self, cv_image, canvas):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(cv_image))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.photo = photo

    def display_message(self, message):
        # Display message if any error occurs
        error_label = tk.Label(self.root, text=message, font=("Helvetica", 16))
        error_label.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()