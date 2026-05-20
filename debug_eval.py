import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("models/regressor.keras")

img_path = "data/processed/dusty_beige_20211231_123105 (Custom).jpg"
img = Image.open(img_path).convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(img), axis=0)

pred = model.predict(x)[0][0]
print(f"Prediction for {img_path}: {pred}")
