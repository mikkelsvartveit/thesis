import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# Import your custom modules
from dataset_loaders import get_dataset
from models import get_model
from transforms import get_transforms
from validators import get_validator


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

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.FileHandler(self.run_dir / "run.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def save_config(self):
        # Save the configuration for reproducibility
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")

        # Also save metrics to a JSON file for easier analysis
        metrics_file = self.run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics[step] = metrics
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)


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
            # Default hyperparameters
            "data": {"batch_size": 32, "num_workers": 4, "dataset": "your_dataset"},
            "model": {
                "name": "your_model",
                "learning_rate": 0.001,
                "optimizer": "adam",
                "weight_decay": 0.0001,
            },
            "training": {
                "epochs": 100,
                "early_stopping_patience": 10,
                "validation_interval": 1,
            },
            "validation": {"method": "kfold", "n_splits": 5},
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
    exp_manager = ExperimentManager(config)
    logger = exp_manager.logger

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get dataset and transforms
    transforms = get_transforms(**config["data"].get("transforms", {}))
    dataset = get_dataset(transform=transforms, **config["data"])

    # Get validator
    validator = get_validator(**config["validation"])

    # Training loop with cross-validation
    for fold, (train_idx, val_idx) in enumerate(validator.split(dataset)):
        logger.info(f"Starting fold {fold + 1}/{config['validation']['n_splits']}")

        # Create data loaders for this fold
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )

        # Initialize model for this fold
        model = get_model(**config["model"]).to(device)

        # Get optimizer
        optimizer = getattr(torch.optim, config["model"]["optimizer"])(
            model.parameters(),
            lr=config["model"]["learning_rate"],
            weight_decay=config["model"]["weight_decay"],
        )

        # Training loop for this fold
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config["training"]["epochs"]):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = model.criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            if (epoch + 1) % config["training"]["validation_interval"] == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        val_loss += model.criterion(output, target).item()
                val_loss /= len(val_loader)

                # Log metrics
                metrics = {"train_loss": train_loss, "val_loss": val_loss}
                exp_manager.log_metrics(metrics, epoch)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), exp_dir / f"model_fold_{fold}.pt")
                else:
                    patience_counter += 1
                    if (
                        patience_counter
                        >= config["training"]["early_stopping_patience"]
                    ):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        logger.info(
            f"Fold {fold + 1} completed. Best validation loss: {best_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
