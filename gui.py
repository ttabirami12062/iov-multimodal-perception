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
        self.root.title("A multi model detection")
        self.root.geometry("1600x800")  # Set window size

        style = ThemedStyle(root)
        style.set_theme("equilux")  # Set the theme for the whole app

        title_label = tk.Label(root, text="A multi model detection ", font=("Helvetica", 20))
        title_label.pack(pady=20)

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect Objects", command=self.detect_objects)
        self.detect_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="LightYellow", padx=20, pady=20)  # Set panel color and padding
        self.image_label.pack(fill=tk.BOTH, expand=True)  # Expand label to fill window

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
            self.display_cv_image(self.loaded_cv_image)

    def detect_objects(self):
        if self.loaded_cv_image is not None:
            if self.model_1 is not None and self.model_2 is not None:
                # Run inference with the first YOLO model
                results_1 = self.model_1(source=self.loaded_cv_image)
                res_plotted_1 = results_1[0].plot()

                # Run inference with the second YOLO model
                results_2 = self.model_2(source=self.loaded_cv_image)
                res_plotted_2 = results_2[0].plot()

                # Display the results from both models
                # For demonstration, let's display the results from both models side by side
                res_plotted_rgb_1 = cv2.cvtColor(res_plotted_1, cv2.COLOR_BGR2RGB)
                res_plotted_rgb_2 = cv2.cvtColor(res_plotted_2, cv2.COLOR_BGR2RGB)

                # Combine the results horizontally
                combined_image = np.hstack((res_plotted_rgb_1, res_plotted_rgb_2))

                self.display_cv_image(combined_image)
            else:
                self.display_message("Models not loaded!")

    def display_cv_image(self, cv_image):
        cv_image = Image.fromarray(cv_image)
        cv_image = cv_image.resize((1600, 600))
        photo = ImageTk.PhotoImage(cv_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def display_message(self, message):
        self.image_label.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()