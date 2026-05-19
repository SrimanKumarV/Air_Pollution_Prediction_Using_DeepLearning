import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("data/processed/labels.csv")
image_paths = ["data/processed/" + fname for fname in df["filename"]]
labels = df["pollution_level"].values

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return img, label

ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
ds = ds.map(load_image).batch(32)
train_size = int(0.8 * len(df)) // 32
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze base model for initial training

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x) # MobileNetV2 expects [-1, 1] inputs
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mean_absolute_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(train_ds, validation_data=val_ds, epochs=15)
model.save("models/regressor.keras")

import pandas as pd
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("models/regressor_history.csv", index=False)