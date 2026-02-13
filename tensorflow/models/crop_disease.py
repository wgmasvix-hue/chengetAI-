"""Crop disease classification model using transfer learning."""

import tensorflow as tf


def build_crop_disease_model(
    num_classes: int,
    image_size: tuple = (224, 224),
    backbone: str = "resnet50",
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
):
    """Build a crop disease classification model with a pretrained backbone.

    Args:
        num_classes: Number of disease/healthy classes.
        image_size: Input image dimensions (height, width).
        backbone: Pretrained model name ("resnet50", "mobilenetv2", "efficientnetb0").
        dropout_rate: Dropout rate before final dense layer.
        learning_rate: Optimizer learning rate.

    Returns:
        Compiled tf.keras.Model.
    """
    input_shape = (*image_size, 3)

    backbones = {
        "resnet50": tf.keras.applications.ResNet50,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }

    if backbone not in backbones:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(backbones.keys())}")

    base_model = backbones[backbone](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=input_shape),
        tf.keras.layers.Rescaling(1.0 / 255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_and_finetune(model, learning_rate: float = 1e-5, unfreeze_layers: int = 20):
    """Unfreeze top layers of the backbone for fine-tuning.

    Args:
        model: The compiled model from build_crop_disease_model.
        learning_rate: Lower learning rate for fine-tuning.
        unfreeze_layers: Number of top backbone layers to unfreeze.

    Returns:
        Recompiled model ready for fine-tuning.
    """
    # The base model is at index 1 (after Rescaling)
    base_model = model.layers[1]
    base_model.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
