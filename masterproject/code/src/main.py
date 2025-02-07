import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# Import your custom modules
from dataset_loaders import get_dataset

# from models import get_model
from src.models import get_model
from src.validators import LOGO_architecture
from transforms import get_transform


class ExperimentManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.save_config()

    def setup_logging(self):
        # Create experiments directory if it doesn't exist
        exp_dir = Path("experiments")
        exp_dir.mkdir(exist_ok=True)

        # Create a unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = exp_dir / f"run_{timestamp}"
        self.run_dir.mkdir()

    def save_config(self):
        # Save the configuration for reproducibility
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)


def get_config():
    """Get configuration from command line arguments or default config file."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--override", type=json.loads, help="JSON string to override config"
    )
    args = parser.parse_args()

    # Load base config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "target_feature": "endianness",
            "data": {
                "name": "ISAdetectDataset",
                "dataset_path": "../dataset/ISAdetect/ISAdetect_full_dataset",
                "feature_csv_path": "../dataset/ISAdetect-features.csv",
                "per_architecture_limit": None,
                "file_byte_read_limit": 10000,
                "use_code_only": True,
            },
            "transforms": {
                "name": "GrayScaleImage",
                "dimx": 100,
                "dimy": 100,
                "normalize": True,
                "duplicate_to_n_channels": 1,
            },
            "model": {
                "name": "MINOS_cnn",
                "learning_rate": 0.001,
                "optimizer": "AdamW",
                "num_classes": 2,
                "weight_decay": 0.0001,
            },
            "validator": {"name": "LOGO_architecture"},
            "training": {"epochs": 2, "batch_size": 32},
        }

    # Override with command line arguments if provided
    if args.override:

        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        config = update_dict(config, args.override)

    return config


def main():
    # Get configuration
    config = get_config()

    # Initialize experiment manager
    # exp_manager = ExperimentManager(config)

    # Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Get dataset and transforms
    transforms = get_transform(**config["transforms"])
    dataset = get_dataset(transform=transforms, **config["data"])
    model = get_model(**config["model"])

    validator_name = config["validator"]["name"]

    if validator_name == "LOGO_architecture":
        print("LOGO_architecture")
        return LOGO_architecture(config, dataset, model, device)


if __name__ == "__main__":
    main()
