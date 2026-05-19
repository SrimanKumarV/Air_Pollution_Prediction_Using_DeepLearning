import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import shutil, glob

# Create directories: data/classifier/clean and data/classifier/dusty
clean_src = "data/raw/mango-leaf-disease-dataset/Healthy"
dusty_src = "data/processed"
classifier_dir = "data/classifier"
clean_dst = os.path.join(classifier_dir, "clean")
dusty_dst = os.path.join(classifier_dir, "dusty")

os.makedirs(clean_dst, exist_ok=True)
os.makedirs(dusty_dst, exist_ok=True)

for f in glob.glob(f"{clean_src}/*.*"):
    shutil.copy(f, clean_dst)
for f in glob.glob(f"{dusty_src}/dusty_*.*"):
    shutil.copy(f, dusty_dst)

# Load dataset from the specific classifier directory
batch_size = 32
img_size = (224, 224)
train_ds = tf.keras.utils.image_dataset_from_directory(
    classifier_dir, validation_split=0.2, subset="training",
    seed=123, image_size=img_size, batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    classifier_dir, validation_split=0.2, subset="validation",
    seed=123, image_size=img_size, batch_size=batch_size
)

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

# Build classifier
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x) # MobileNetV2 expects [-1, 1] inputs
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train top layer
print("Training top layer...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Fine-tune
print("Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(train_ds, validation_data=val_ds, epochs=5)

import pandas as pd
hist_df1 = pd.DataFrame(history1.history)
hist_df2 = pd.DataFrame(history2.history)
hist_df = pd.concat([hist_df1, hist_df2], ignore_index=True)
hist_df.to_csv("models/classifier_history.csv", index=False)

model.save("models/classifier.keras")