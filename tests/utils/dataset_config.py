"""Shared test dataset configurations"""

from pathlib import Path

import yaml

from axolotl.utils.dict import DictDefault


def get_alpaca_dataset_config(num_epochs=1):
    """
    Returns a DictDefault config for the alpaca dataset with common settings.

    Args:
        num_epochs: Number of epochs for training

    Returns:
        DictDefault: Configuration dictionary with dataset and training settings
    """
    return DictDefault(
        {
            "datasets": [
                {
                    "path": "tatsu-lab/alpaca",
                    "type": "alpaca",
                    "split": "train[:10%]",
                },
            ],
            "num_epochs": num_epochs,
        }
    )


def write_config_to_yaml(cfg, temp_dir, config_filename="config.yaml"):
    """
    Write configuration to a YAML file.

    Args:
        cfg: Configuration dictionary (must have to_dict() method)
        temp_dir: Directory to write the config file to
        config_filename: Name of the config file

    Returns:
        Path: Path to the written config file
    """
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(temp_dir) / config_filename
    with open(config_path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))
    return config_path
