from datetime import datetime
import numpy as np
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import Subset, random_split
import wandb.wandb_run
from src.validators.train_test_utils import (
    set_seed,
    save_model_as_onnx,
)


def train_test_split(
    config,
    model_class: nn.Module,
    dataset: Dataset,
    device,
):
    wandb_group_name = f"{config["validator"]["name"]} {config["model"]["name"]} {config["target_feature"]} {datetime.now().strftime('%H:%M:%S, %d-%m-%Y')}"
    wandb_project = config["validator"].get("wandb_project_name", None)
    if not wandb_project:
        print("Wandb project name not provided, skipping wandb logging")

    wandb_run = wandb.init(
        project=wandb_project,
        name="train_test_separate_datasets",
        group=wandb_group_name,
        config=config,
        mode=(
            "online" if config["validator"]["wandb_project_name"] else "disabled"
        ),  # disabled = no-op
    )

    # Initial seed setting
    seed = config["validator"]["seed"]
    set_seed(seed)

    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("test_accuracy", step_metric="epoch")

    groups = list(map(lambda x: x["architecture"], dataset.metadata))
    target_features = list(map(lambda x: x[config["target_feature"]], dataset.metadata))

    print(f"group: {set(groups)}")
    print(f"target_feature: {set(target_features)}")

    assert config["validator"][
        "train_split_size"
    ], "train_split_size not provided in config"

    # Split dataset into training and test sets based on config
    train_ratio = config["validator"]["train_split_size"]
    dataset_size = len(dataset)

    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Use PyTorch's random_split
    training_dataset, testing_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config["validator"]["seed"]),
    )

    print(f"Training set size: {len(training_dataset)}")
    print(f"Testing set size: {len(testing_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        training_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        testing_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = model_class(**config["model"]["params"])
    model.to(device)
    wandb_run.watch(model)
    criterion = (
        getattr(model, "criterion", None)
        or getattr(nn, config["training"]["criterion"])()
    )
    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    all_train_labels = [
        dataset.metadata[i][config["target_feature"]]
        for i in range(len(training_dataset))
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_train_labels)

    EPOCHS = config["training"]["epochs"]
    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}:")

        # ========== TRAINING LOOP ==========
        model.train()

        # Training metrics
        total_training_loss = 0
        training_predictions = []
        training_true_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            encoded_labels = torch.from_numpy(
                label_encoder.transform(labels[config["target_feature"]])
            ).to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, encoded_labels)
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

            # Collect predictions and true labels
            _, predicted = torch.max(predictions, 1)
            training_predictions.extend(predicted.cpu().numpy())
            training_true_labels.extend(encoded_labels.cpu().numpy())

            # Log batch metrics
            wandb_run.log(
                {
                    "train_loss": loss.item(),
                },
            )

        # Calculate training metrics
        avg_training_loss = total_training_loss / len(train_loader)

        # ========== TESTING LOOP ==========
        model.eval()

        # Test metrics
        total_testing_loss = 0
        testing_predictions = []
        testing_true_labels = []
        architecture_predictions: dict[str, list] = {}
        architecture_true_labels: dict[str, list] = {}

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                encoded_labels = torch.from_numpy(
                    label_encoder.transform(labels[config["target_feature"]])
                ).to(device)

                outputs = model(images)
                loss = criterion(outputs, encoded_labels)
                total_testing_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                testing_predictions.extend(predicted.cpu().numpy())
                testing_true_labels.extend(encoded_labels.cpu().numpy())

                for i, arch in enumerate(labels["architecture"]):
                    if arch not in architecture_predictions:
                        architecture_predictions[arch] = []
                        architecture_true_labels[arch] = []

                    architecture_predictions[arch].append(predicted[i].cpu().item())
                    architecture_true_labels[arch].append(
                        encoded_labels[i].cpu().item()
                    )

        # Calculate testing metrics
        testing_accuracies = {}
        instance_counts = {}
        for arch in architecture_predictions:
            arch_accuracy = np.mean(
                np.array(architecture_predictions[arch])
                == np.array(architecture_true_labels[arch])
            )

            testing_accuracies[arch] = arch_accuracy
            instance_counts[arch] = len(architecture_predictions[arch])

            print(f"{arch} Accuracy: {100*arch_accuracy:.2f}%")

        avg_testing_loss = total_testing_loss / len(test_loader)
        testing_total_accuracy = np.mean(
            np.array(testing_predictions) == np.array(testing_true_labels)
        )

        # Prepare per-architecture accuracy data for logging
        wandb_log_data = {
            "epoch": epoch,
            "train_loss": avg_training_loss,
            "test_loss": avg_testing_loss,
            "test_accuracy": testing_total_accuracy,
        }
        for arch, acc in testing_accuracies.items():
            wandb_log_data[f"test_accuracy_{arch}"] = acc
        for arch, count in instance_counts.items():
            wandb_log_data[f"instance_count_{arch}"] = count

        wandb_run.log(wandb_log_data)

    # ======== REPORT FINAL RESULTS ========
    wandb_run.log(
        {
            "test_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in testing_accuracies.items()],
                columns=["group", "accuracy"],
            ),
        }
    )

    # try save model if onnx is installed
    if config["validator"].get("save_model", False):
        print("Saving model as ONNX")
        save_model_as_onnx(
            model=model,
            model_name=config["model"]["name"],
            sample_dataset=training_dataset,
        )

    wandb_run.finish()
