import argparse
import json
from pprint import pprint
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os
import torch
import wandb
import random


load_dotenv()
from src.dataset_loaders import get_dataset
from src.models import get_model
from src.validators import (
    ISAdetect_train_cpu_rec_test,
    LOGO_architecture,
    LOGO_architecture_wandb,
)
from transforms import get_transform


def get_config(configs_base_path: Path) -> Dict[str, Any]:
    """Get configuration from command line arguments or default config file."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--override",
        type=json.loads,
        help='JSON string to override config, ex: --override \'{"data": {"params": {"per_architecture_limit": 5}}}\'',
    )
    args = parser.parse_args()

    # Load base config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        with open(configs_base_path / Path("default.yml")) as f:
            config = yaml.safe_load(f)

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
        print("Overridden config:")
        pprint(config)

    return config


def main():
    DATASET_BASE_PATH = Path(os.environ["DATASET_BASE_PATH"])
    CONFIGS_BASE_PATH = Path(os.environ["CONFIGS_PATH"])
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

    # Get configuration
    config = get_config(configs_base_path=CONFIGS_BASE_PATH)

    # Generate random seed if not provided
    if "seed" not in config.get("validator", {}):
        config.setdefault("validator", {})
        config["validator"]["seed"] = random.randint(0, 2**32 - 1)

    print(f"Using random seed: {config["validator"]["seed"]}")

    # Login to wandb
    if not wandb.login(key=WANDB_API_KEY, timeout=60):
        raise ValueError("Failed to login to wandb")

    # Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Get dataset and transforms
    transforms = get_transform(**config["transforms"])
    dataset = get_dataset(
        transform=transforms,
        dataset_base_path=DATASET_BASE_PATH,
        target_feature=config["target_feature"],
        **config["data"],
    )
    model = get_model(**config["model"])

    validator_name = config["validator"]["name"]

    if validator_name == "LOGO_architecture":
        print("LOGO_architecture")
        LOGO_architecture(config, dataset, model, device)
    elif validator_name == "LOGO_architecture_wandb":
        print("LOGO_architecture_wandb")
        LOGO_architecture_wandb(config, dataset, model, device)
    elif validator_name == "ISAdetect_train_cpu_rec_test":
        validator_dataset = get_dataset(
            transform=transforms,
            dataset_base_path=DATASET_BASE_PATH,
            target_feature=config["target_feature"],
            **config["testing_data"],
        )
        print("Testing on ISAdetect_train_cpu_rec_test")
        ISAdetect_train_cpu_rec_test(
            config,
            ISAdetectDataset=dataset,
            CpuRecDataset=validator_dataset,
            device=device,
            model_class=model,
        )

    elif validator_name == "pass":
        print("Passing validor step")
    else:
        raise ValueError(f"Unknown validator: {validator_name}")

    print("============== Done ==============")


if __name__ == "__main__":
    main()
