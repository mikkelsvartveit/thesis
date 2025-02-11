from sklearn.calibration import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


def LOGO_architecture(
    config, dataset: Dataset, model_class: nn.Module.__class__, device
):
    groups = list(map(lambda x: x["architecture"], dataset.metadata))
    target_features = list(map(lambda x: x[config["target_feature"]], dataset.metadata))

    logo = LeaveOneGroupOut()
    label_encoder = LabelEncoder()
    scaler = torch.cuda.amp.GradScaler()

    fold = 1
    accuracies = {}
    for train_idx, test_idx in logo.split(
        X=range(len(dataset)), y=target_features, groups=groups
    ):

        group_left_out = groups[test_idx[0]]

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
        criterion = getattr(model, "criterion", None) or getattr(nn, config["training"]["criterion"])()
        optimizer = getattr(torch.optim, config["training"]["optimizer"])(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Train model
        for epoch in range(config["training"]["epochs"]):
            model.train()
            print(f"\nEpoch {epoch+1}:")

            total_training_loss = 0
            for images, labels in tqdm(train_loader):
                images = images.to(device)

                encoded_labels = torch.from_numpy(
                    label_encoder.transform(labels[config["target_feature"]])
                ).to(device)

                optimizer.zero_grad()

                """ with torch.cuda.amp.autocast():
                    predictions = model(images)
                    loss = criterion(predictions, encoded_labels) """

                """ scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update() """

                predictions = model(images)
                loss = criterion(predictions, encoded_labels)
                loss.backward()
                optimizer.step()

                total_training_loss += loss.item()

            avg_training_loss = total_training_loss / len(train_loader)

            # Evaluate model
            model.eval()
            correct = 0
            total = 0
            total_test_loss = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    encoded_labels = torch.from_numpy(
                        label_encoder.transform(labels[config["target_feature"]])
                    ).to(device)

                    outputs = model(images)
                    loss = criterion(outputs, encoded_labels)
                    total_test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == encoded_labels).sum().item()
                    total += encoded_labels.size(0)

            avg_test_loss = total_test_loss / len(test_loader)
            accuracy = correct / total

            print(
                f"Training Loss: {avg_training_loss:.4f} | Test loss: {avg_test_loss:.4f}"
            )
            print(f"Test Accuracy: {100*accuracy:.2f}%")

        accuracies[group_left_out] = accuracy

    print("\n=== Results ===")
    for group, accuracy in accuracies.items():
        print(f"Group '{group}': {100*accuracy:.2f}%")
    print(f"Average accuracy: {100*sum(accuracies.values())/len(accuracies):.2f}%")
    return accuracies
