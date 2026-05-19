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
    """Apply dust effect based on pollution level (0-100)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:,:,1] *= 0.65
    hsv[:,:,2] *= 0.75
    dulled = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    gauss = np.random.normal(0, dust_density, dulled.shape)
    noisy = np.clip(dulled + gauss * 0.4, 0, 255).astype(np.uint8)
    dust_tint = np.full_like(noisy, (140, 160, 170), dtype=np.uint8)
    return cv2.addWeighted(noisy, 0.75, dust_tint, 0.25, 0)

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