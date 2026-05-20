import tensorflow as tf
import cv2
import numpy as np
import sys

def predict():
    try:
        model = tf.keras.models.load_model("models/regressor.keras")
        import pandas as pd
        df = pd.read_csv("data/processed/labels.csv")
        sample = df.iloc[0]
        img_path = "data/processed/" + sample["filename"]
        actual_val = sample["pollution_level"]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found! {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        print(f"Prediction for {sample['filename']}: {prediction[0][0]:.2f} (Actual: {actual_val})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    predict()
