
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import time

import tensorflow as tf
import numpy as np
import random
import os

# Fix random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================
# 1️⃣ Paths and Parameters
# ==============================
DATASET_PATH = "classification"        # Folder containing subfolders for each class
TEST_PATH = "classification_test"      # Folder for unseen test data
MODEL_SAVE_PATH = "fetus_growth_model.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50

# ==============================
# 2️⃣ Data Generators
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2   # Split 80% train, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation"
)

# ==============================
# 3️⃣ Load Pretrained Model (MobileNetV2)
# ==============================
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Show model summary (for report computational requirements)
model.summary()

# ==============================
# 4️⃣ Compile Model
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 5️⃣ Callbacks
# ==============================
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# ==============================
# 6️⃣ Train Model
# ==============================
start_time = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)
end_time = time.time()
training_time = end_time - start_time
print(f"\n🕒 Total training time: {training_time/60:.2f} minutes\n")

# ==============================
# 7️⃣ Plot Accuracy & Loss
# ==============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# 8️⃣ Evaluate on Validation Data
# ==============================
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc:.2%}")

# ==============================
# 9️⃣ Evaluate on Unseen (Test) Data
# ==============================
if os.path.exists(TEST_PATH):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy (Unseen Data): {test_acc:.2%}")

    # Predictions and true labels
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix (Unseen Data)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Classification Report
    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # ==============================
    # 🔍 ROC Curves and AUC
    # ==============================
    y_true_cat = to_categorical(y_true, num_classes=len(class_labels))
    plt.figure(figsize=(7,5))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==============================
    # 🔍 Precision-Recall Curves
    # ==============================
    plt.figure(figsize=(7,5))
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_true_cat[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f"{label}")
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# 🔟 Save Final Model
# ==============================
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at {MODEL_SAVE_PATH}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

# ---- Accuracy/Loss Curves ----
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ---- Predictions ----
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# ---- Confusion Matrix ----
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ---- Classification Report ----
print("Classification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))

# ---- ROC Curve ----
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
n_classes = len(test_generator.class_indices)
y_true_bin = label_binarize(test_generator.classes, classes=range(n_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ---- Computational Details ----
import time, psutil, tensorflow as tf
print(f"Model Parameters: {model.count_params()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"System CPU usage: {psutil.cpu_percent()}%")
print("Training performed on GPU" if tf.config.list_physical_devices('GPU') else "Training performed on CPU")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load test images
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "classification_test",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f"🧪 Test Accuracy: {test_acc:.2%}")

# Predict
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# ==============================
# 🔍 Precision, Recall, F1-Score, ROC, and PR Curves
# ==============================
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import label_binarize

# Binarize true labels for multi-class ROC/PR
n_classes = len(class_labels)
y_true_bin = label_binarize(y_true, classes=range(n_classes))

# ---- 📋 Precision, Recall, F1 per Class ----
report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
print("\n📊 Precision, Recall, and F1-Score per Class:")
for label in class_labels:
    print(f"{label:15s} | Precision: {report[label]['precision']:.2f} | Recall: {report[label]['recall']:.2f} | F1: {report[label]['f1-score']:.2f}")

# ---- 🧭 ROC Curves ----
plt.figure(figsize=(8,6))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curves for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# ---- 🎯 Precision-Recall Curves ----
plt.figure(figsize=(8,6))
for i, label in enumerate(class_labels):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
    plt.plot(recall, precision, label=f"{label}")
plt.title('Precision-Recall Curves for Each Class')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# ---- 🌟 Macro Averages ----
roc_auc_macro = np.mean([auc(*roc_curve(y_true_bin[:, i], y_pred[:, i])[:2]) for i in range(n_classes)])
print(f"\n⭐ Average AUC (macro): {roc_auc_macro:.3f}")
# --- 📋 Precision, Recall, F1 table ---
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True)
print("\n📊 Precision, Recall, and F1-Score per Class:")
for label in class_labels:
    print(f"{label:20s} | Precision: {report[label]['precision']:.2f} | Recall: {report[label]['recall']:.2f} | F1: {report[label]['f1-score']:.2f}")
