import os
import random
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 2. DATASET PATH
# =========================================================

# Dataset folder structure example:
# dataset/
# ├── coated/
# └── non_coated/

DATASET_PATH = "dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25

# =========================================================
# 3. LOAD DATASET
# =========================================================

images = []
labels = []

classes = ["coated", "non_coated"]

for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    label = 1 if class_name == "coated" else 0

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        try:
            img = cv2.imread(file_path)

            # Skip unreadable images
            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize image
            img = img / 255.0

            images.append(img)
            labels.append(label)

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

X = np.array(images)
y = np.array(labels)

print("Dataset loaded successfully")
print("Total samples:", len(X))
print("Image shape:", X[0].shape)

# =========================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

# 80% Train
# 10% Validation
# 10% Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp
)

print("\nDataset Split")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =========================================================
# 5. DATA AUGMENTATION
# ONLY FOR TRAINING SET
# =========================================================

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# =========================================================
# 6. BASELINE CNN MODEL
# =========================================================

baseline_model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

baseline_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nBaseline Model Summary")
baseline_model.summary()

# =========================================================
# 7. TRAIN BASELINE MODEL
# =========================================================

print("\nTraining Baseline CNN...")

history_baseline = baseline_model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1
)

# =========================================================
# 8. EVALUATE BASELINE MODEL
# =========================================================

baseline_predictions = baseline_model.predict(X_test)
baseline_predictions = (baseline_predictions > 0.5).astype(int)

baseline_acc = accuracy_score(y_test, baseline_predictions)
baseline_precision = precision_score(y_test, baseline_predictions)
baseline_f1 = f1_score(y_test, baseline_predictions)

print("\n===== BASELINE MODEL RESULTS =====")
print(f"Accuracy  : {baseline_acc:.4f}")
print(f"Precision : {baseline_precision:.4f}")
print(f"F1-Score  : {baseline_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, baseline_predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, baseline_predictions))

# =========================================================
# 9. IMPROVED CNN MODEL
# =========================================================

improved_model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

improved_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nImproved Model Summary")
improved_model.summary()

# =========================================================
# 10. EARLY STOPPING
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================================================
# 11. TRAIN IMPROVED MODEL
# =========================================================

print("\nTraining Improved CNN...")

history_improved = improved_model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# 12. EVALUATE IMPROVED MODEL
# =========================================================

improved_predictions = improved_model.predict(X_test)
improved_predictions = (improved_predictions > 0.5).astype(int)

improved_acc = accuracy_score(y_test, improved_predictions)
improved_precision = precision_score(y_test, improved_predictions)
improved_f1 = f1_score(y_test, improved_predictions)

print("\n===== IMPROVED MODEL RESULTS =====")
print(f"Accuracy  : {improved_acc:.4f}")
print(f"Precision : {improved_precision:.4f}")
print(f"F1-Score  : {improved_f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, improved_predictions))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, improved_predictions))

# =========================================================
# 13. FINAL COMPARISON
# =========================================================

print("\n================================================")
print("FINAL PERFORMANCE COMPARISON")
print("================================================")

print(f"Baseline Accuracy  : {baseline_acc:.4f}")
print(f"Improved Accuracy  : {improved_acc:.4f}")
print()
print(f"Baseline Precision : {baseline_precision:.4f}")
print(f"Improved Precision : {improved_precision:.4f}")
print()
print(f"Baseline F1-Score  : {baseline_f1:.4f}")
print(f"Improved F1-Score  : {improved_f1:.4f}")

# =========================================================
# 14. SAVE MODELS
# =========================================================

baseline_model.save("baseline_cnn_model.h5")
improved_model.save("improved_cnn_model.h5")

print("\nModels saved successfully.")

# =========================================================
# 15. OPTIONAL TRAINING CURVES
# =========================================================

import matplotlib.pyplot as plt

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history_improved.history['accuracy'], label='Train Accuracy')
plt.plot(history_improved.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history_improved.history['loss'], label='Train Loss')
plt.plot(history_improved.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

print("\nPipeline completed successfully.")
