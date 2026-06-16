import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==========================================
# KONFIGURASI
# ==========================================

DATASET_PATH = "Data/flowers"
MODEL_PATH = "models/flower_model.h5"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

# ==========================================
# BUAT FOLDER MODELS
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# DATA AUGMENTATION
# ==========================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# ==========================================
# DATA TRAINING
# ==========================================

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# ==========================================
# DATA VALIDASI
# ==========================================

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================================
# SIMPAN NAMA KELAS
# ==========================================

class_names = list(train_generator.class_indices.keys())

with open("models/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n===================================")
print("KELAS YANG DITEMUKAN")
print("===================================")
print(class_names)

# ==========================================
# MODEL CNN
# ==========================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        len(class_names),
        activation="softmax"
    )

])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n===================================")
print("RINGKASAN MODEL")
print("===================================")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================================
# TRAINING MODEL
# ==========================================

print("\n===================================")
print("MEMULAI TRAINING MODEL...")
print("===================================\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ==========================================
# EVALUASI MODEL
# ==========================================

loss, accuracy = model.evaluate(validation_generator)

print("\n===================================")
print(f"AKURASI VALIDASI : {accuracy * 100:.2f}%")
print(f"LOSS VALIDASI    : {loss:.4f}")
print("===================================")

# ==========================================
# SIMPAN TRAINING HISTORY KE JSON
# ==========================================

history_data = {
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    "loss": [float(x) for x in history.history["loss"]],
    "val_loss": [float(x) for x in history.history["val_loss"]],
    "final_accuracy": float(accuracy),
    "final_loss": float(loss)
}

with open("models/training_history.json", "w") as f:
    json.dump(history_data, f, indent=2)

print("\nTraining history disimpan di: models/training_history.json")

# ==========================================
# GRAFIK AKURASI
# ==========================================

plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Model Accuracy", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2,
    color="red"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2,
    color="orange"
)

plt.title("Model Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/accuracy.png", dpi=150)
plt.close()

print("\nGrafik Accuracy & Loss disimpan di: models/accuracy.png")

# ==========================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

# Prediksi pada data validasi
validation_generator.reset()
y_pred_probs = model.predict(validation_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = validation_generator.classes

# Classification Report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
print(report)

# Simpan classification report ke file
with open("models/classification_report.txt", "w") as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)

print("Classification report disimpan di: models/classification_report.txt")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

# Tampilkan angka di setiap cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12, fontweight="bold"
        )

plt.ylabel("Actual", fontsize=12)
plt.xlabel("Predicted", fontsize=12)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix disimpan di: models/confusion_matrix.png")

# ==========================================
# INFORMASI FILE HASIL
# ==========================================

print("\n===================================")
print("TRAINING SELESAI")
print("===================================")

print("\nFile yang dihasilkan:")
print("  1. models/flower_model.h5         — Model CNN")
print("  2. models/class_names.txt         — Nama kelas")
print("  3. models/accuracy.png            — Grafik Accuracy & Loss")
print("  4. models/confusion_matrix.png    — Confusion Matrix")
print("  5. models/classification_report.txt — Classification Report")
print("  6. models/training_history.json   — History Training (JSON)")