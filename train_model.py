import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from collections import Counter

# -----------------------------
# ✅ Paths & Settings
# -----------------------------
DATASET_PATH = "classification"
TEST_PATH = "classification_test"
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "fetus_growth_model.keras")
HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

FINE_TUNE = False
FINE_TUNE_AT = 100

GRADCAM_SAVE_DIR = os.path.join(OUTPUT_DIR, "gradcam_overlays")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRADCAM_SAVE_DIR, exist_ok=True)

# -----------------------------
# 🔁 Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training",
    seed=SEED
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation",
    seed=SEED
)

# -----------------------------
# ⚖️ Class Weights
# -----------------------------
train_labels = train_generator.classes
class_counts = Counter(train_labels)
total = float(sum(class_counts.values()))
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

print("Class counts:", class_counts)
print("Class weights:", class_weights)

# -----------------------------
# 🏗 Model: MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 🛠 Callbacks
# -----------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
csv_logger = CSVLogger(HISTORY_CSV)

callbacks = [early_stopping, checkpoint, reduce_lr, csv_logger]

# -----------------------------
# 🚀 Train
# -----------------------------
start = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)
end = time.time()
print(f"Training time: {(end-start)/60:.2f} minutes")

pd.DataFrame(history.history).to_csv(HISTORY_CSV, index=False)

# -----------------------------
# 📊 Plot Training Curves
# -----------------------------
def plot_history(history):
    plt.figure()
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.show()

plot_history(history.history)

# -----------------------------
# 🔥 GRAD-CAM (UPDATED: ALL IMAGES)
# -----------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def save_gradcam(image_path, model, last_conv_layer_name):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = np.array(img).astype(np.uint8)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    # Plot 3-panel output
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_color)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    save_path = os.path.join(GRADCAM_SAVE_DIR, os.path.basename(image_path))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("✅ Saved 3-panel Grad-CAM:", save_path)

# -----------------------------
# 🧠 Generate Grad-CAM for ALL Test Images
# -----------------------------
print("\nGenerating Grad-CAM explanations...")

try:
    last_conv_layer = find_last_conv_layer(model)
    print("Using last conv layer:", last_conv_layer)
except:
    last_conv_layer = "Conv_1"

test_images = []
for root, dirs, files in os.walk(TEST_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            test_images.append(os.path.join(root, file))

print(f"🔍 Found {len(test_images)} test images.")

for img_path in test_images:
    try:
        save_gradcam(img_path, model, last_conv_layer)
    except Exception as e:
        print("Skipping:", img_path, "Error:", e)

print("✅ All Grad-CAM results saved in:", GRADCAM_SAVE_DIR)
