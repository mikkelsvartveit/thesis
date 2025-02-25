from datetime import datetime
import numpy as np
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch


def ISAdetect_train_cpu_rec_test(
    config,
    model_class: nn.Module,
    ISAdetectDataset: Dataset,
    CpuRecDataset: Dataset,
    device,
):

    wandb_group_name = f"{config["testing"]["name"]} {config["model"]["name"]} {config["target_feature"]} {datetime.now().strftime('%H:%M:%S, %d-%m-%Y')}"

    wandb_project = config["testing"]["wandb_project_name"]

    wandb.init(
        project=wandb_project,
        name="ISAdetect_train",
        group=wandb_group_name,
        config=config,
    )

    wandb.define_metric("epoch")
    wandb.define_metric("validation_loss", step_metric="epoch")
    wandb.define_metric("validation_loss", step_metric="epoch")
    wandb.define_metric("validation_accuracy", step_metric="epoch")

    groups = list(map(lambda x: x["architecture"], ISAdetectDataset.metadata))
    target_features = list(
        map(lambda x: x[config["target_feature"]], ISAdetectDataset.metadata)
    )

    print(f"group: {set(groups)}")
    print(f"target_feature: {set(target_features)}")

    validation_split = config["testing"]["validation_split"]
    test_split = config["testing"]["test_split"]
    train_split = 1 - validation_split - test_split

    train_size = int(train_split * len(ISAdetectDataset))
    validation_size = int(validation_split * len(ISAdetectDataset))
    test_size = len(ISAdetectDataset) - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(
        ISAdetectDataset, [train_size, validation_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = model_class(**config["model"]["params"])
    model.to(device)
    wandb.watch(model)
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
        ISAdetectDataset.metadata[i][config["target_feature"]]
        for i in range(len(ISAdetectDataset))
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_train_labels)

    EPOCHS = config["training"]["epochs"]
    model.train()
    for epoch in range(EPOCHS):
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
            wandb.log(
                {
                    f"batch_loss": loss.item(),
                },
            )

        # Calculate training metrics
        avg_training_loss = total_training_loss / len(train_loader)

        # ========== EPOCH VALIDATION LOOOP ==========
        model.eval()
        total_test_loss = 0
        validation_predictions = []
        validation_true_labels = []
        architecture_predictions = {}
        architecture_true_labels = {}

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                encoded_labels = torch.from_numpy(
                    label_encoder.transform(labels[config["target_feature"]])
                ).to(device)

                outputs = model(images)
                loss = criterion(outputs, encoded_labels)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                validation_predictions.extend(predicted.cpu().numpy())
                validation_true_labels.extend(encoded_labels.cpu().numpy())

                for i, arch in enumerate(labels["architecture"]):
                    if arch not in architecture_predictions:
                        architecture_predictions[arch] = []
                        architecture_true_labels[arch] = []

                    architecture_predictions[arch].append(predicted[i].cpu().item())
                    architecture_true_labels[arch].append(
                        encoded_labels[i].cpu().item()
                    )

        # ===== AFTER EPOCH METRICS =====

        avg_validation_loss = total_test_loss / len(validation_loader)

        validation_total_accuracy = np.mean(
            np.array(validation_predictions) == np.array(validation_true_labels)
        )
        print(
            f"Training Loss: {avg_training_loss:.4f} | Validation loss: {avg_validation_loss:.4f}"
        )
        print(f"Validation Accuracy: {100*validation_total_accuracy:.2f}%")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_training_loss,
                "validation_loss": avg_validation_loss,
                "validation_accuracy": validation_total_accuracy,
            }
        )

        validation_accuracies = {}
        for arch in architecture_predictions:
            arch_accuracy = np.mean(
                np.array(architecture_predictions[arch])
                == np.array(architecture_true_labels[arch])
            )
            validation_accuracies[arch] = arch_accuracy
            print(f"{arch} Accuracy: {100*arch_accuracy:.2f}%")

    # ======== AFTER TRAINING METRICS ========

    # last validation results
    wandb.log(
        {
            "validation_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in validation_accuracies.items()],
                columns=["group", "accuracy"],
            ),
        }
    )

    print("Finished training")
    wandb.finish()

    # ======== ISADETECT TESTING ========
    print("\n===== Testing on ISAdetect =====")
    wandb.init(
        project=wandb_project,
        name="ISAdetect_test",
        group=wandb_group_name,
        config=config,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    model.eval()
    total_test_loss = 0
    testing_predictions = []
    testing_true_labels = []
    architecture_predictions = {}
    architecture_true_labels = {}

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            encoded_labels = torch.from_numpy(
                label_encoder.transform(labels[config["target_feature"]])
            ).to(device)

            outputs = model(images)
            loss = criterion(outputs, encoded_labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            testing_predictions.extend(predicted.cpu().numpy())
            testing_true_labels.extend(encoded_labels.cpu().numpy())

            for i, arch in enumerate(labels["architecture"]):
                if arch not in architecture_predictions:
                    architecture_predictions[arch] = []
                    architecture_true_labels[arch] = []

                architecture_predictions[arch].append(predicted[i].cpu().item())
                architecture_true_labels[arch].append(encoded_labels[i].cpu().item())

    # ====== AFTER TESTING METRICS ======
    testing_accuracies = {}
    for arch in architecture_predictions:
        arch_accuracy = np.mean(
            np.array(architecture_predictions[arch])
            == np.array(architecture_true_labels[arch])
        )
        testing_accuracies[arch] = arch_accuracy
        print(f"{arch} Accuracy: {100*arch_accuracy:.2f}%")

    wandb.log(
        {
            "ISAdetect_testing_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in testing_accuracies.items()],
                columns=["group", "accuracy"],
            ),
        }
    )

    avg_testing_loss = total_test_loss / len(test_loader)
    testing_total_accuracy = np.mean(
        np.array(testing_predictions) == np.array(testing_true_labels)
    )
    print(f"Test Loss: {avg_testing_loss:.4f}")
    print(f"Test Accuracy: {100*testing_total_accuracy:.2f}%")
    print("Finished testing")
    print("Logging to wandb")
    wandb.log(
        {
            "ISAdetect_test_loss": avg_testing_loss,
            "ISAdetect_test_accuracy": testing_total_accuracy,
        }
    )
    wandb.finish()

    # ======= CPU REC TESTING =======
    print("\n===== Testing on CPU REC =====")
    wandb.init(
        project=wandb_project,
        name="CPU_REC_test",
        group=wandb_group_name,
        config=config,
    )

    cpu_rec_loader = DataLoader(
        CpuRecDataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    model.eval()
    total_test_loss = 0
    testing_predictions = []
    testing_true_labels = []
    architecture_predictions = {}
    architecture_true_labels = {}

    with torch.no_grad():
        for images, labels in tqdm(cpu_rec_loader):
            images = images.to(device)
            encoded_labels = torch.from_numpy(
                label_encoder.transform(labels[config["target_feature"]])
            ).to(device)

            outputs = model(images)
            loss = criterion(outputs, encoded_labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            testing_predictions.extend(predicted.cpu().numpy())
            testing_true_labels.extend(encoded_labels.cpu().numpy())

            for i, arch in enumerate(labels["architecture"]):
                if arch not in architecture_predictions:
                    architecture_predictions[arch] = []
                    architecture_true_labels[arch] = []

                architecture_predictions[arch].append(predicted[i].cpu().item())
                architecture_true_labels[arch].append(encoded_labels[i].cpu().item())

    # ====== AFTER TESTING METRICS ======
    testing_accuracies = {}
    for arch in architecture_predictions:
        arch_accuracy = np.mean(
            np.array(architecture_predictions[arch])
            == np.array(architecture_true_labels[arch])
        )
        testing_accuracies[arch] = arch_accuracy
        print(f"{arch} Accuracy: {100*arch_accuracy:.2f}%")

    wandb.log(
        {
            "cpu_rec_testing_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in testing_accuracies.items()],
                columns=["group", "accuracy"],
            ),
        }
    )

    avg_testing_loss = total_test_loss / len(cpu_rec_loader)
    testing_total_accuracy = np.mean(
        np.array(testing_predictions) == np.array(testing_true_labels)
    )
    print(f"Test Loss: {avg_testing_loss:.4f}")
    print(f"Test Accuracy: {100*testing_total_accuracy:.2f}%")
    print("Finished testing")
    print("Logging to wandb")
    wandb.log(
        {
            "cpu_rec_test_loss": avg_testing_loss,
            "cpu_rec_test_accuracy": testing_total_accuracy,
        }
    )
    wandb.finish()
