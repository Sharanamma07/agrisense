"""
AgriSense — training script.

Fine-tunes a MobileNetV2 backbone (ImageNet weights) on the PlantVillage
leaf-disease dataset using a two-phase transfer learning approach:
  Phase 1: freeze base, train only the new classification head
  Phase 2: unfreeze the top layers of the base and fine-tune at a low LR

Expects data laid out as:
  data/train/<class_name>/*.jpg
  data/val/<class_name>/*.jpg
"""

import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "data"
MODEL_DIR = "models"
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 6
FINE_TUNE_AT_LAYER = 100  # unfreeze from this layer index onward in phase 2


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    # Save class names now so app.py can map prediction indices -> labels
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # Light augmentation to reduce overfitting on the (fairly uniform) PlantVillage images
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def prep(ds, training):
        ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    return prep(train_ds, True), prep(val_ds, False), class_names


def build_model(num_classes):
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # phase 1: frozen backbone

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


def main():
    train_ds, val_ds, class_names = build_datasets()
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    model, base = build_model(num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "agrisense_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    ]

    # --- Phase 1: train the head only ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Phase 1: training classification head ===")
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=callbacks)

    # --- Phase 2: fine-tune top layers of the backbone ---
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Phase 2: fine-tuning top layers ===")
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)

    print(f"\nDone. Best model saved to {os.path.join(MODEL_DIR, 'agrisense_model.keras')}")


if __name__ == "__main__":
    main()
