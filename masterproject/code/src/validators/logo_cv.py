from datetime import datetime
import wandb
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from .train_test_utils import set_seed
import time


def logo_cv(config, dataset: Dataset, model_class: nn.Module.__class__, device):

    has_wandb_projet_name = (
        "wandb_project_name" in config["validator"]
        and config["validator"]["wandb_project_name"]
    )
    if not has_wandb_projet_name:
        print("Wandb project name not provided, skipping wandb logging")

    # Initial seed setting
    seed = config["validator"]["seed"]
    set_seed(seed)

    # Start timing the entire cross-validation process
    start_time = time.time()

    group_name = f"logo {config['model']['name']} {config['target_feature']} {datetime.now().strftime('%H:%M:%S, %d-%m-%Y')}"

    wandb_project = config["validator"]["wandb_project_name"]

    groups = list(map(lambda x: x["architecture"], dataset.metadata))
    target_features = list(map(lambda x: x[config["target_feature"]], dataset.metadata))

    logo = LeaveOneGroupOut()
    label_encoder = LabelEncoder()
    scaler = torch.cuda.amp.GradScaler()

    fold = 1
    accuracies = {}
    all_predictions = []
    all_true_labels = []

    for train_idx, test_idx in logo.split(
        X=range(len(dataset)), y=target_features, groups=groups
    ):
        # Reset seed at the start of each fold
        set_seed(seed)

        group_left_out = groups[test_idx[0]]
        wandb_run = wandb.init(
            project=wandb_project,
            config=config,
            group=f"{group_name}",
            name=f"fold_{group_left_out}",
            mode=(
                "online" if has_wandb_projet_name else "disabled"
            ),  # disabled = no-op
        )

        wandb_run.define_metric("epoch")
        # define which metrics will be plotted against it
        wandb_run.define_metric("train_loss", step_metric="epoch")
        wandb_run.define_metric("test_loss", step_metric="epoch")
        wandb_run.define_metric("test_accuracy_chunk", step_metric="epoch")
        wandb_run.define_metric("test_accuracy_file", step_metric="epoch")

        print(f"\n=== Fold {fold} leaving out group '{group_left_out}' ===")
        fold += 1

        all_train_labels = [
            dataset.metadata[i][config["target_feature"]] for i in train_idx
        ]
        label_encoder.fit(all_train_labels)

        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )

        model = model_class(**config["model"]["params"])
        model = model.to(device)
        criterion = (
            getattr(model, "criterion", None)
            or getattr(nn, config["training"]["criterion"])()
        )
        optimizer = getattr(torch.optim, config["training"]["optimizer"])(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        wandb_run.log(
            {
                "test_dataset_size": len(test_dataset),
                "train_dataset_size": len(train_dataset),
                "group_left_out": group_left_out,
            }
        )

        # Training loop
        for epoch in range(config["training"]["epochs"]):
            model.train()
            print(f"\nEpoch {epoch+1}:")

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
                        f"batch_loss": loss.item(),
                    },
                )

            # Calculate training metrics
            avg_training_loss = total_training_loss / len(train_loader)

            # Evaluation loop
            model.eval()
            total_test_loss = 0
            chunk_predictions = []
            chunk_true_labels = []
            file_predictions_map = {}
            file_true_labels_map = {}

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    file_paths = labels["file_path"]
                    encoded_labels = torch.from_numpy(
                        label_encoder.transform(labels[config["target_feature"]])
                    ).to(device)

                    outputs = model(images)
                    loss = criterion(outputs, encoded_labels)
                    total_test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    batch_predictions = predicted.cpu().numpy()
                    batch_true_labels = encoded_labels.cpu().numpy()

                    # Store chunk-level predictions
                    chunk_predictions.extend(batch_predictions)
                    chunk_true_labels.extend(batch_true_labels)

                    # Store predictions by parent file for majority voting
                    for pred, true_label, file_path in zip(
                        batch_predictions, batch_true_labels, file_paths
                    ):
                        if file_path not in file_predictions_map:
                            file_predictions_map[file_path] = []
                            file_true_labels_map[file_path] = true_label
                        file_predictions_map[file_path].append(pred)

            # Calculate test metrics
            avg_test_loss = total_test_loss / len(test_loader)

            # Calculate chunk-level accuracy
            chunk_accuracy = np.mean(
                np.array(chunk_predictions) == np.array(chunk_true_labels)
            )

            # Calculate majority voting accuracy
            file_level_predictions = []
            file_level_true_labels = []
            for file_path in file_predictions_map:
                # Get majority vote for this file
                chunk_predictions_for_file = file_predictions_map[file_path]
                vote_distribution = np.bincount(
                    chunk_predictions_for_file, minlength=len(label_encoder.classes_)
                )
                file_prediction = vote_distribution.argmax()
                file_true_label = file_true_labels_map[file_path]

                file_level_predictions.append(file_prediction)
                file_level_true_labels.append(file_true_label)

            file_level_accuracy = np.mean(
                np.array(file_level_predictions) == np.array(file_level_true_labels)
            )

            print(
                f"Training Loss: {avg_training_loss:.4f} | Test loss: {avg_test_loss:.4f}"
            )
            print(f"Chunk-level Test Accuracy: {100*chunk_accuracy:.2f}%")
            print(f"File-level Test Accuracy: {100*file_level_accuracy:.2f}%")

            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_training_loss,
                    "test_loss": avg_test_loss,
                    "test_accuracy_chunk": chunk_accuracy,
                    "test_accuracy_file": file_level_accuracy,
                }
            )

        accuracies[group_left_out] = (
            file_level_accuracy  # Use file-level accuracy for final results
        )
        all_predictions.extend(file_level_predictions)
        all_true_labels.extend(file_level_true_labels)

        wandb_run.finish()

    end_time = time.time()
    elapsed_time = end_time - start_time

    wandb_run = wandb.init(
        project=wandb_project,
        config=config,
        group=f"{group_name}",
        name="overall_metrics",
        mode=("online" if has_wandb_projet_name else "disabled"),  # disabled = no-op
    )

    # try save model if onnx is installed
    try:
        sample_input = DataLoader(dataset, batch_size=1, shuffle=True)
        sample_input = next(iter(sample_input))[0]

        torch.onnx.export(
            model_class(**config["model"]["params"]),
            sample_input,
            "model.onnx",
        )
        wandb_run.save("model.onnx")
    except Exception as e:
        print(f"Failed to save model as onnx: {e}")

    # Calculate and log overall metrics
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    print(f"formatted time: {formatted_time}")

    wandb_run.log(
        {
            "overall_accuracy": overall_accuracy,
            "accuracies_per_group": wandb.Table(
                data=[[group, acc] for group, acc in accuracies.items()],
                columns=["group", "accuracy"],
            ),
            "time_taken": formatted_time,
        }
    )

    # Print final results
    print("\n=== Results ===")
    for group, accuracy in accuracies.items():
        print(f"Group '{group}': {100*accuracy:.2f}%")
    print(f"Average accuracy: {100*sum(accuracies.values())/len(accuracies):.2f}%")

    wandb_run.finish()
    return accuracies
