"""Main training script for ChengetAI agricultural models."""

import argparse
from pathlib import Path

import tensorflow as tf

from utils.config import load_config
from data.preprocessing import create_image_dataset, create_augmentation_layer
from models.crop_disease import build_crop_disease_model, unfreeze_and_finetune
from models.yield_prediction import build_yield_prediction_model


def train_crop_disease(config: dict):
    """Train the crop disease classification model.

    Args:
        config: Project configuration dictionary.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    data_dir = data_cfg.get("raw_dir", "data/raw")

    print(f"Loading image data from: {data_dir}")
    train_ds, val_ds, class_names = create_image_dataset(
        data_dir=data_dir,
        image_size=tuple(data_cfg["image_size"]),
        batch_size=data_cfg["batch_size"],
        validation_split=data_cfg["validation_split"],
        seed=data_cfg["seed"],
    )

    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Apply data augmentation to training set
    augment = create_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Build model
    model = build_crop_disease_model(
        num_classes=num_classes,
        image_size=tuple(data_cfg["image_size"]),
        backbone=model_cfg["backbone"],
        dropout_rate=model_cfg["dropout_rate"],
        learning_rate=model_cfg["learning_rate"],
    )

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=train_cfg["reduce_lr_patience"],
            factor=0.5,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(train_cfg["checkpoint_dir"]) / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=train_cfg["log_dir"],
        ),
    ]

    # Phase 1: Train with frozen backbone
    print("\n--- Phase 1: Training with frozen backbone ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune top layers
    print("\n--- Phase 2: Fine-tuning top layers ---")
    model = unfreeze_and_finetune(model, learning_rate=1e-5)
    finetune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
    )

    print(f"\nModel saved to: {train_cfg['checkpoint_dir']}/best_model.keras")
    return model


def train_yield_prediction(config: dict):
    """Train the yield prediction regression model.

    Args:
        config: Project configuration dictionary.
    """
    train_cfg = config["training"]

    # Placeholder: Replace with actual feature count from your dataset
    input_features = 10

    model = build_yield_prediction_model(
        input_features=input_features,
        learning_rate=config["model"]["learning_rate"],
    )

    model.summary()

    print("Yield prediction model built successfully.")
    print("To train, provide a CSV dataset and update the data loading logic.")
    return model


def main():
    parser = argparse.ArgumentParser(description="ChengetAI Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["crop_disease", "yield_prediction"],
        default=None,
        help="Model task to train (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    task = args.task or config["model"]["task"]
    print(f"Starting training for task: {task}")

    if task == "crop_disease":
        train_crop_disease(config)
    elif task == "yield_prediction":
        train_yield_prediction(config)
    else:
        print(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
