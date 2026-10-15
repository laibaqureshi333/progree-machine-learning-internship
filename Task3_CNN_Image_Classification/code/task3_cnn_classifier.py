"""
TASK 3: Multi-Class Image Classification CNN
=====================================================
Objective : Train a deep-learning vision pipeline that categorizes raw pixel
            image arrays into one of several classes.

Dataset   : CIFAR-10 (10 classes, 32x32 RGB images) via tf.keras.datasets.
            (Swap DATASET = "mnist" below to run the same pipeline on MNIST.)

Pipeline stages:
    1. Data ingestion (Keras built-in dataset loader)
    2. Preprocessing -> normalization, one-hot label encoding, train/val split
    3. Data augmentation -> random flips, rotations, zoom, shifts (Keras layers)
    4. Custom CNN architecture -> stacked Conv2D + MaxPooling2D blocks,
       BatchNorm, Dropout, then Dense fully-connected head
    5. Training with callbacks (early stopping, LR reduction)
    6. Evaluation -> test accuracy/loss, classification report, confusion matrix
    7. Plots -> training/validation accuracy & loss curves
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# 0. Paths & logging setup (portable — works in Colab, notebooks, anywhere)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training_log.txt"), mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("Task3-CNN")

DATASET = "cifar10"     # "cifar10" or "mnist"
EPOCHS = 25             # increase for higher accuracy in a full run
BATCH_SIZE = 32
RANDOM_STATE = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CLASS_NAMES_CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

# --------------------------------------------------------------------------
# 1. Data ingestion
#    Standard path: tf.keras.datasets.{cifar10,mnist}.load_data(), which
#    downloads the canonical dataset the first time it's called. Some
#    sandboxed/offline environments block that outbound download, so this
#    falls back to Scikit-Learn's bundled `load_digits` dataset (1,797 8x8
#    grayscale digit images, 10 classes, ships with sklearn -> zero network
#    calls) upsampled to a CNN-friendly resolution. On a normal machine with
#    internet access, the CIFAR-10/MNIST branch runs as-is and this fallback
#    is never triggered.
# --------------------------------------------------------------------------
log.info(f"Loading {DATASET} dataset...")
try:
    if DATASET == "cifar10":
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        class_names = CLASS_NAMES_CIFAR10
        input_shape = (32, 32, 3)
    else:
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train_full = x_train_full[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        class_names = [str(i) for i in range(10)]
        input_shape = (28, 28, 1)
    log.info(f"Successfully downloaded {DATASET} via tf.keras.datasets.")
except Exception as e:
    log.warning(f"Could not download {DATASET} ({e}). Falling back to "
                f"Scikit-Learn's bundled 'load_digits' dataset (no network "
                f"required) as a drop-in 10-class image dataset.")
    from sklearn.datasets import load_digits
    from scipy.ndimage import zoom

    digits = load_digits()
    images = digits.images  # (1797, 8, 8), values 0-16
    labels = digits.target

    # Upsample 8x8 -> 32x32 so the conv/pool stack and augmentation layers
    # have a realistic amount of spatial resolution to work with, then
    # replicate to 3 channels to keep the pipeline RGB-shaped like CIFAR-10.
    images_resized = np.stack([zoom(img, 4, order=1) for img in images])  # -> 32x32
    images_resized = images_resized * (255.0 / 16.0)  # digits pixels are 0-16; rescale to 0-255
    images_rgb = np.repeat(images_resized[..., np.newaxis], 3, axis=-1)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        images_rgb, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    class_names = [str(i) for i in range(10)]
    input_shape = (32, 32, 3)

num_classes = len(class_names)
log.info(f"Train+val shape: {x_train_full.shape} | Test shape: {x_test.shape}")

# --------------------------------------------------------------------------
# 2. Preprocessing: normalize to [0,1], carve out a validation split,
#    one-hot encode labels
# --------------------------------------------------------------------------
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

val_fraction = 0.1
n_val = int(len(x_train_full) * val_fraction)
rng = np.random.RandomState(RANDOM_STATE)
perm = rng.permutation(len(x_train_full))
val_idx, train_idx = perm[:n_val], perm[n_val:]

x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

y_train_oh = keras.utils.to_categorical(y_train, num_classes)
y_val_oh = keras.utils.to_categorical(y_val, num_classes)
y_test_oh = keras.utils.to_categorical(y_test, num_classes)

log.info(f"Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")

# --------------------------------------------------------------------------
# 3. Data augmentation (applied only at training time, inside the model)
# --------------------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")

# --------------------------------------------------------------------------
# 4. Custom CNN architecture
#    3 conv blocks (Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> MaxPool ->
#    Dropout) with increasing filter depth, then a dense classification head.
# --------------------------------------------------------------------------
def build_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # Fully connected head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="custom_cnn")

model = build_cnn(input_shape, num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary(print_fn=lambda line: log.info(line))

# --------------------------------------------------------------------------
# 5. Training with callbacks
# --------------------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=3, min_lr=1e-6),
]

log.info(f"Starting training for up to {EPOCHS} epochs...")
history = model.fit(
    x_train, y_train_oh,
    validation_data=(x_val, y_val_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

# --------------------------------------------------------------------------
# 6. Evaluation on held-out test set
# --------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
log.info(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
log.info("\nClassification Report:\n" + report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------------------------------
# 7. Plots: accuracy/loss curves + confusion matrix
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Training vs Validation Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Training vs Validation Loss")
axes[1].legend()

im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks(range(num_classes))
axes[2].set_yticks(range(num_classes))
axes[2].set_xticklabels(class_names, rotation=90, fontsize=7)
axes[2].set_yticklabels(class_names, fontsize=7)
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title(f"Confusion Matrix (Test Acc={test_acc:.3f})")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
log.info("Saved training curves + confusion matrix to training_curves.png")

model.save(os.path.join(OUTPUT_DIR, "cnn_model.keras"))
log.info("Saved trained model to cnn_model.keras")
log.info("Task 3 pipeline complete.")

# Display the training curves inline if running in a notebook (Colab/Jupyter);
# silently skip this if running as a plain .py script from the command line.
try:
    from IPython.display import Image, display
    display(Image(os.path.join(OUTPUT_DIR, "training_curves.png")))
except ImportError:
    pass
