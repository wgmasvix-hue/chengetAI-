"""Data preprocessing utilities for agricultural image and tabular data."""

import tensorflow as tf
import numpy as np
from pathlib import Path


def create_image_dataset(
    data_dir: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Create training and validation datasets from a directory of images.

    Expects directory structure:
        data_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
                ...

    Args:
        data_dir: Path to root image directory.
        image_size: Target (height, width) for resizing.
        batch_size: Number of samples per batch.
        validation_split: Fraction of data for validation.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_dataset, val_dataset, class_names).
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    # Prefetch for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def create_augmentation_layer():
    """Create a data augmentation pipeline for training images."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])


def load_and_preprocess_image(image_path: str, image_size: tuple = (224, 224)):
    """Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        image_size: Target (height, width).

    Returns:
        Preprocessed image tensor with batch dimension.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img


def load_csv_dataset(csv_path: str, target_column: str, batch_size: int = 32):
    """Load a CSV file as a tf.data.Dataset for tabular predictions.

    Args:
        csv_path: Path to the CSV file.
        target_column: Name of the target/label column.
        batch_size: Number of samples per batch.

    Returns:
        A tf.data.Dataset yielding (features_dict, label) pairs.
    """
    dataset = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=batch_size,
        label_name=target_column,
        num_epochs=1,
        shuffle=True,
    )
    return dataset
