"""Yield prediction model for tabular agricultural data."""

import tensorflow as tf


def build_yield_prediction_model(
    input_features: int,
    learning_rate: float = 0.001,
):
    """Build a regression model for crop yield prediction.

    Uses tabular input features such as soil moisture, temperature,
    rainfall, fertilizer usage, and crop type.

    Args:
        input_features: Number of input feature columns.
        learning_rate: Optimizer learning rate.

    Returns:
        Compiled tf.keras.Model for regression.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_features,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
