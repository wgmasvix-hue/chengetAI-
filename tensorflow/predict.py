"""Inference script for running predictions with trained models."""

import argparse
import json

import numpy as np
import tensorflow as tf

from utils.config import load_config
from data.preprocessing import load_and_preprocess_image


def predict_image(model_path: str, image_path: str, class_names: list = None, image_size: tuple = (224, 224)):
    """Run prediction on a single image.

    Args:
        model_path: Path to saved .keras model.
        image_path: Path to the input image.
        class_names: List of class label names.
        image_size: Expected input image dimensions.

    Returns:
        Dictionary with predicted class, confidence, and all probabilities.
    """
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(image_path, image_size)

    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])

    result = {
        "predicted_index": int(predicted_index),
        "confidence": confidence,
        "probabilities": predictions[0].tolist(),
    }

    if class_names:
        result["predicted_class"] = class_names[predicted_index]

    return result


def predict_batch(model_path: str, image_dir: str, image_size: tuple = (224, 224), batch_size: int = 32):
    """Run predictions on a directory of images.

    Args:
        model_path: Path to saved .keras model.
        image_dir: Directory containing images.
        image_size: Expected input image dimensions.
        batch_size: Batch size for inference.

    Returns:
        List of prediction results.
    """
    model = tf.keras.models.load_model(model_path)

    dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        labels=None,
    )

    predictions = model.predict(dataset)
    results = []

    for i, pred in enumerate(predictions):
        results.append({
            "index": i,
            "predicted_class": int(np.argmax(pred)),
            "confidence": float(np.max(pred)),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="ChengetAI Model Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.keras)")
    parser.add_argument("--image", type=str, help="Path to a single image for prediction")
    parser.add_argument("--image-dir", type=str, help="Directory of images for batch prediction")
    parser.add_argument("--classes", type=str, help="Comma-separated list of class names")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = tuple(config["data"]["image_size"])
    class_names = args.classes.split(",") if args.classes else None

    if args.image:
        result = predict_image(args.model, args.image, class_names, image_size)
        print(json.dumps(result, indent=2))
    elif args.image_dir:
        results = predict_batch(args.model, args.image_dir, image_size)
        print(json.dumps(results, indent=2))
    else:
        print("Provide either --image or --image-dir for inference.")


if __name__ == "__main__":
    main()
