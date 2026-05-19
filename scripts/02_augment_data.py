import cv2
import numpy as np
import os
import glob
import pandas as pd
import random

# Paths
input_folder = "data/raw/mango-leaf-disease-dataset/Healthy/*.*"
output_folder = "data/processed/"
os.makedirs(output_folder, exist_ok=True)

def apply_urban_dust(image, dust_density):
    """Apply extreme beige/brown dust occlusion based on pollution level (0-100)."""
    # Create the beige dust color layer (BGR format for OpenCV)
    dust_color = np.full_like(image, (170, 195, 210), dtype=np.uint8)
    
    # Add noise to the dust layer to simulate heavy sand/soil texture
    gauss = np.random.normal(0, 30, image.shape).astype(np.int16)
    dust_texture = np.clip(dust_color.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    
    # Calculate occlusion percentage (0.0 to 1.0)
    occlusion = min(dust_density / 100.0, 1.0)
    
    # Dull the underlying leaf slightly based on pollution (less sunlight)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:,:,1] *= (1.0 - (0.6 * occlusion)) # Desaturate heavily
    hsv[:,:,2] *= (1.0 - (0.4 * occlusion)) # Darken
    dulled_leaf = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Blend the dulled leaf with the heavy beige dust texture
    # If occlusion is 100%, the image becomes 95% beige dust and 5% leaf texture (to keep veins visible)
    blend_weight = occlusion * 0.95
    dusty_leaf = cv2.addWeighted(dulled_leaf, 1.0 - blend_weight, dust_texture, blend_weight, 0)
    
    return dusty_leaf

image_paths = glob.glob(input_folder)
data_log = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (224, 224))
    pollution = random.uniform(10.0, 100.0)
    dusty = apply_urban_dust(img, pollution)
    fname = f"dusty_{os.path.basename(img_path)}"
    cv2.imwrite(os.path.join(output_folder, fname), dusty)
    data_log.append({"filename": fname, "pollution_level": round(pollution, 2)})

df = pd.DataFrame(data_log)
df.to_csv(os.path.join(output_folder, "labels.csv"), index=False)
print(f"Generated {len(data_log)} images and labels.csv")