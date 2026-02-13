"""Configuration loader for ChengetAI TensorFlow project."""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. Defaults to project root config.yaml.

    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
