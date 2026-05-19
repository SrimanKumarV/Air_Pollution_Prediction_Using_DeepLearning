import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("aryashah2k/mango-leaf-disease-dataset")
print(f"Dataset downloaded to: {path}")

# Move raw data to project folder
os.makedirs("data/raw", exist_ok=True)
shutil.move(path, "data/raw/mango-leaf-disease-dataset")