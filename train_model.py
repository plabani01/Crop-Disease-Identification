import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

# Paths
train_dir = pathlib.Path("dataset/train")
val_dir = pathlib.Path("dataset/val")

# Image parameters
img_height, img_width = 224, 224
batch_size = 16

# Load training dataset
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# Get class names BEFORE mapping
class_names = raw_train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Load validation dataset
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalize pixel values (0–255 → 0–1)
normalization_layer = layers.Rescaling(1./255)
train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))

# Cache + prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5  # increase if dataset is large
)

# Save the model
model.save("crop_disease_model.h5")
print("✅ Model saved as crop_disease_model.h5")

import json

# Save class names to a JSON file
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("✅ Saved class names to class_names.json")