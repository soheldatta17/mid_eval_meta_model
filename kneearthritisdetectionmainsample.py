# -*- coding: utf-8 -*-
"""KneeArthritisDetectionMainSample.py
Knee Arthritis Detection using CNN (Keras/TensorFlow).
- Emojis removed
- Step-by-step print statements added
- Final model saved at the end
"""

# ------------------------------------------------------------------------------
# STEP 1: Setup Kaggle and download dataset
# ------------------------------------------------------------------------------
print("[STEP 1/10] Setting up Kaggle credentials and downloading dataset ...")

import os
import subprocess

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
subprocess.run(["cp", "/content/kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)
print("[STEP 1/10] Kaggle credentials configured.")

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "hafiznouman786/annotated-dataset-for-knee-arthritis-detection"
], check=True)
print("[STEP 1/10] Dataset downloaded.")

subprocess.run(["unzip", "-qq", "annotated-dataset-for-knee-arthritis-detection.zip"], check=True)
print("[STEP 1/10] Dataset extracted.")

# ------------------------------------------------------------------------------
# STEP 2: Import libraries
# ------------------------------------------------------------------------------
print("\n[STEP 2/10] Importing libraries ...")

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import split_dataset
from tensorflow.keras import layers, backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print("[STEP 2/10] Libraries imported.")

# ------------------------------------------------------------------------------
# Helper: Squeeze-and-Excitation (SE) Block for Channel Attention
# ------------------------------------------------------------------------------
def se_block(input_tensor, ratio=16):
    """Channel attention mechanism: learns to weight feature channels.
    Args:
        input_tensor: input feature map
        ratio: compression ratio for internal dimension (default 16)
    """
    filters = K.int_shape(input_tensor)[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    return layers.Multiply()([input_tensor, se])

print("[STEP 2/10] SE block helper defined.")

# ------------------------------------------------------------------------------
# STEP 3: Load dataset
# ------------------------------------------------------------------------------
print("\n[STEP 3/10] Loading dataset from directory ...")

dataset = image_dataset_from_directory(
    "Training",
    color_mode="grayscale",
    image_size=(256, 256),
    batch_size=None,
)

print("[STEP 3/10] Dataset loaded.")

# Preview first 20 images
print("[STEP 3/10] Generating dataset preview plot ...")
dataset_preview = dataset.take(20)

plt.figure(figsize=(15, 12))
for i, (image, label) in enumerate(dataset_preview):
    plt.subplot(4, 5, i + 1)
    plt.imshow(image.numpy().astype("uint8").squeeze(), cmap="inferno")
    plt.title(f"Label: {label.numpy().astype('uint8')}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("dataset_preview.png", dpi=100)
plt.show()
print("[STEP 3/10] Dataset preview saved -> dataset_preview.png")

# ------------------------------------------------------------------------------
# STEP 4: Split dataset
# ------------------------------------------------------------------------------
print("\n[STEP 4/10] Splitting dataset into train / val / test ...")

train_dataset, test_dataset = split_dataset(dataset, right_size=0.1)
train_dataset, val_dataset  = split_dataset(train_dataset, right_size=0.1)

print(f"[STEP 4/10] Train size      : {train_dataset.cardinality().numpy()}")
print(f"[STEP 4/10] Test size       : {test_dataset.cardinality().numpy()}")
print(f"[STEP 4/10] Validation size : {val_dataset.cardinality().numpy()}")

# ------------------------------------------------------------------------------
# STEP 5: Build and train initial model (no augmentation)
# ------------------------------------------------------------------------------
print("\n[STEP 5/10] Building initial CNN model (no augmentation) ...")

inputs = keras.Input(shape=(256, 256, 1))
x = layers.Rescaling(1./255)(inputs)

x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = se_block(x, ratio=16)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

print("[STEP 5/10] Model built. Compiling ...")

batched_train_dataset = train_dataset.batch(32)
batched_val_dataset   = val_dataset.batch(32)

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("[STEP 5/10] Training initial model for 20 epochs ...")
history = model.fit(
    batched_train_dataset,
    epochs=20,
    validation_data=batched_val_dataset,
)

print("[STEP 5/10] Initial model training complete.")

# Plot helper
def show_plots(history, save_path=None):
    accuracy     = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    loss         = history["loss"]
    val_loss     = history["val_loss"]
    epochs       = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy,     "bo", label="Train Accuracy")
    plt.plot(epochs, val_accuracy, "b",  label="Val Accuracy")
    plt.title("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss,     "bo", label="Train Loss")
    plt.plot(epochs, val_loss, "b",  label="Val Loss")
    plt.title("Loss"); plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"  Plot saved -> {save_path}")
    plt.show()

show_plots(history.history, save_path="plots_initial_model.png")

# ------------------------------------------------------------------------------
# STEP 6: Model with data augmentation (no dense layers)
# ------------------------------------------------------------------------------
print("\n[STEP 6/10] Building CNN model with data augmentation ...")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

inputs = keras.Input(shape=(256, 256, 1))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(filters=8,   kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=16,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=32,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = se_block(x, ratio=16)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)

batched_train_dataset = train_dataset.batch(32)
batched_val_dataset   = val_dataset.batch(32)

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_aug = keras.callbacks.ModelCheckpoint(
    "best_cnn_with_data_augmentation.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

print("[STEP 6/10] Training augmented model for 100 epochs ...")
history = model.fit(
    batched_train_dataset,
    epochs=100,
    validation_data=batched_val_dataset,
    callbacks=callbacks_aug,
)

best_val_acc = max(history.history["val_accuracy"])
print(f"[STEP 6/10] Training complete. Best Val Accuracy: {best_val_acc * 100:.2f}%")
show_plots(history.history, save_path="plots_augmented_model.png")

print("[STEP 6/10] Loading best augmented model for evaluation ...")
loaded_model = keras.models.load_model("best_cnn_with_data_augmentation.h5")
loaded_model.summary()
eval_loss, eval_acc = loaded_model.evaluate(test_dataset.batch(32))
print(f"[STEP 6/10] Test Loss: {eval_loss:.4f}  Test Accuracy: {eval_acc * 100:.2f}%")

# ------------------------------------------------------------------------------
# STEP 7: Model with data augmentation + dense layers
# ------------------------------------------------------------------------------
print("\n[STEP 7/10] Building CNN model with augmentation + dense layers ...")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

inputs = keras.Input(shape=(256, 256, 1))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(filters=8,   kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=16,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=32,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64,  kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = se_block(x, ratio=16)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs, outputs)

batched_train_dataset = train_dataset.batch(16)
batched_val_dataset   = val_dataset.batch(16)

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_dense = keras.callbacks.ModelCheckpoint(
    "best_cnn_with_data_augmentation_and_dense.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

print("[STEP 7/10] Training augmented + dense model for 300 epochs ...")
history = model.fit(
    batched_train_dataset,
    epochs=300,
    validation_data=batched_val_dataset,
    callbacks=callbacks_dense,
)

best_val_acc = max(history.history["val_accuracy"])
print(f"[STEP 7/10] Training complete. Best Val Accuracy: {best_val_acc * 100:.2f}%")
show_plots(history.history, save_path="plots_augmented_dense_model.png")

# ------------------------------------------------------------------------------
# STEP 8: Evaluate final model
# ------------------------------------------------------------------------------
print("\n[STEP 8/10] Loading best augmented + dense model for evaluation ...")

loaded_model = keras.models.load_model("best_cnn_with_data_augmentation_and_dense.h5")
loaded_model.summary()

test_loss, test_acc = loaded_model.evaluate(test_dataset.batch(32), verbose=0)
train_acc_final     = history.history["accuracy"][-1]

print(f"[STEP 8/10] Final Training Accuracy  : {train_acc_final * 100:.2f}%")
print(f"[STEP 8/10] Best Validation Accuracy : {best_val_acc * 100:.2f}%")
print(f"[STEP 8/10] Test Loss                : {test_loss:.4f}")
print(f"[STEP 8/10] Test Accuracy            : {test_acc * 100:.2f}%")

# ------------------------------------------------------------------------------
# STEP 9: Random inference on 5 test samples
# ------------------------------------------------------------------------------
print("\n[STEP 9/10] Running inference on 5 random test samples ...")

correct_predictions = 0
total_predictions   = 5

for i, (image, label) in enumerate(test_dataset.shuffle(1000).take(total_predictions), 1):
    img              = np.expand_dims(image.numpy(), axis=0)
    prediction       = loaded_model.predict(img, verbose=0)
    predicted_label  = np.argmax(prediction)
    actual_label     = label.numpy().astype("uint8")

    result_str = "CORRECT" if predicted_label == actual_label else "WRONG"
    if predicted_label == actual_label:
        correct_predictions += 1

    print(f"[STEP 9/10] Sample {i}/{total_predictions}: "
          f"Actual: {actual_label}  Predicted: {predicted_label}  [{result_str}]")

    plt.imshow(image.numpy().astype("uint8").squeeze(), cmap="inferno")
    plt.title(f"Actual: {actual_label}  Predicted: {predicted_label}  [{result_str}]")
    plt.axis("off")
    plt.savefig(f"inference_sample_{i}.png", dpi=100, bbox_inches="tight")
    plt.show()

accuracy = (correct_predictions / total_predictions) * 100
print(f"\n[STEP 9/10] Inference complete: {correct_predictions}/{total_predictions} correct")
print(f"[STEP 9/10] Sample Accuracy: {accuracy:.2f}%")

# ------------------------------------------------------------------------------
# STEP 10: Save final model in both .h5 and SavedModel formats
# ------------------------------------------------------------------------------
print("\n[STEP 10/10] Saving final model ...")

# Save as .h5 (Keras native format)
loaded_model.save("final_knee_cnn_model.h5")
print("[STEP 10/10] Model saved -> final_knee_cnn_model.h5")

# Save as SavedModel format (TensorFlow standard, best for serving/meta-model)
loaded_model.save("final_knee_cnn_savedmodel.keras", save_format="keras")
print("[STEP 10/10] Model saved -> final_knee_cnn_savedmodel/ (TensorFlow SavedModel format)")

# List all saved files
print("\n[STEP 10/10] All model files saved:")
for fname in sorted(os.listdir(".")):
    if fname.endswith(".h5") or fname.endswith(".png") or os.path.isdir(fname) and "savedmodel" in fname.lower():
        size = os.path.getsize(fname) / 1e6 if os.path.isfile(fname) else 0
        if os.path.isfile(fname):
            print(f"  {fname}  ({size:.1f} MB)")
        else:
            print(f"  {fname}/  (directory)")

print("\n[DONE] Pipeline finished successfully.")