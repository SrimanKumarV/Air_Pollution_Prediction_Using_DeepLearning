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

model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("models/classifier.keras")