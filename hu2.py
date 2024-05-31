import cv2
import numpy as np
import os
import pandas as pd

# Function to calculate Hu Moments for each channel
def calculate_hu_moments(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate Hu Moments for each channel
    moments_gray = cv2.moments(gray)
    huMoments_gray = cv2.HuMoments(moments_gray).flatten()
    
    moments_hue = cv2.moments(hsv[:,:,0])
    huMoments_hue = cv2.HuMoments(moments_hue).flatten()
    
    moments_saturation = cv2.moments(hsv[:,:,1])
    huMoments_saturation = cv2.HuMoments(moments_saturation).flatten()
    
    moments_value = cv2.moments(hsv[:,:,2])
    huMoments_value = cv2.HuMoments(moments_value).flatten()
    
    return np.concatenate([huMoments_gray, huMoments_hue, huMoments_saturation, huMoments_value])

# Define paths and labels
paths = ["28-05-2024/ASC-H",
         "28-05-2024/ASC-US",
         "28-05-2024/HSIL",
         "28-05-2024/LSIL",
         "28-05-2024/Negative for intraepithelial lesion",
         "28-05-2024/SCC"]
labels = ["ASC-H", "ASC-US", "HSIL", "LSIL", "Negative for intraepithelial lesion", "SCC"]

# Initialize lists to store data
data = []
for path, label in zip(paths, labels):
    # Iterate through images in the directory
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Get the full path of the image
            image_path = os.path.join(path, filename)
            
            # Calculate Hu Moments for each channel
            hu_moments = calculate_hu_moments(image_path)
            
            # Append data to list
            data.append(list(hu_moments) + [image_path, label])

# Convert data to DataFrame
columns = [f"Hu_{i}" for i in range(28)] + ["cell_path", "cell_label"]
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_csv("hu_moments_all_channels.csv", index=False)
