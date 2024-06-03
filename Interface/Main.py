import os
import glob
from tkinter import Tk, Label, Button, LEFT, RIGHT, TOP, BOTTOM, Menu
from tkinter import Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ImageViewer:
    def __init__(self, root, directories):
        self.root = root
        self.root.title("Image Viewer")
        
        self.image_label = Label(root)
        self.image_label.pack()
        
        self.prev_button = Button(root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=LEFT)
        
        self.next_button = Button(root, text="Next", command=self.next_image)
        self.next_button.pack(side=RIGHT)
        
        self.gray_button = Button(root, text="Gray Scale", command=self.show_gray_image)
        self.gray_button.pack(side=TOP)
        
        self.original_button = Button(root, text="Original", command=self.show_image)
        self.original_button.pack(side=BOTTOM)
        
        self.directories = directories
        self.image_paths = self.get_image_paths(directories)
        self.current_image_index = 0
        self.show_image()
        
        self.create_menu()
        
    def create_menu(self):
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)
        
        histogram_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Histogram", menu=histogram_menu)
        histogram_menu.add_command(label="Gray Histogram", command=self.show_gray_histogram)
        histogram_menu.add_command(label="HSV Histogram", command=self.show_hsv_histogram)
        
    def get_image_paths(self, directories):
        image_paths = []
        for directory in directories:
            image_paths.extend(glob.glob(os.path.join(directory, '*.*')))
        return sorted(image_paths)

    def show_image(self, index=None):
        if index is not None:
            self.current_image_index = index
        image_path = self.image_paths[self.current_image_index]
        self.image = Image.open(image_path)
        image = self.image.resize((800, 600), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def show_gray_image(self):
        image = self.image.convert("L")
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)
    
    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)
    
    def show_gray_histogram(self):
        gray_image = self.image.convert("L")
        gray_array = np.array(gray_image)
        
        plt.figure()
        plt.hist(gray_array.ravel(), bins=256, range=(0, 256), color='gray')
        plt.title("Gray Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

    def show_hsv_histogram(self):
        hsv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)
        
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        plt.figure()
        plt.title("HSV Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        plt.plot(h_hist, color='r', label='H channel')
        plt.plot(s_hist, color='g', label='S channel')
        plt.plot(v_hist, color='b', label='V channel')
        
        plt.legend()
        plt.show()

if __name__ == "__main__":
    directories = [
        "28-05-2024/ASC-H", 
        "28-05-2024/ASC-US", 
        "28-05-2024/HSIL", 
        "28-05-2024/LSIL", 
        "28-05-2024/Negative for intraepithelial lesion", 
        "28-05-2024/SCC"
    ]
    
    root = Tk()
    viewer = ImageViewer(root, directories)
    root.mainloop()
