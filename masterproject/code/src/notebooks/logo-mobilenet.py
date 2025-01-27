from tqdm import tqdm
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from sklearn.model_selection import LeaveOneGroupOut


def get_device():
    """
    Returns 'cuda' if CUDA is available, else 'mps' if Apple Silicon GPU is available,
    otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class BinaryEndiannessDataset(Dataset):
    """
    Loads files from a directory of subdirectories. Each subdirectory is treated
    as one 'group' for Leave-One-Group-Out cross-validation.
    Converts each binary file to a 150x150 grayscale image (padding/truncating as needed).
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root, which has subdirectories for each instruction set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # will hold (file_path, label, group)

        # Example: define how each folder maps to a label (endianness).
        # You might have subfolders named 'mips', 'powerpc', etc. that are big-endian,
        # and some that are little-endian. Adjust to your actual dataset.
        # Suppose we define a dictionary for big-endian vs. little-endian:
        big_endian_folders = {
            "hppa",
            "m68k",
            "mips",
            "powerpc",
            "powerpcspe",
            "ppc64",
            "s390",
            "s390x",
            "sparc",
            "sparc64",
        }  # Example set
        # If folder name is in big_endian_folders => label=1, else label=0

        # Build the dataset list
        for subfolder in sorted(os.listdir(root_dir)):
            group_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(group_path):
                continue

            # The group is the instruction set (the subfolder name)
            group = subfolder

            # Determine label based on subfolder name. Adjust logic to your data.
            label = 1 if subfolder in big_endian_folders else 0

            # Gather all binary (.bin) files
            for fname in os.listdir(group_path):
                file_path = os.path.join(group_path, fname)
                self.samples.append((file_path, label, group))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, group = self.samples[idx]

        # Read raw bytes
        data = np.fromfile(file_path, dtype=np.uint8)

        # We want exactly 150*150 = 22500 bytes.
        # If less than 22500, pad with zeros; if more, truncate.
        target_size = 224 * 224
        if len(data) < target_size:
            data = np.pad(
                data, (0, target_size - len(data)), "constant", constant_values=0
            )
        else:
            data = data[:target_size]

        # Reshape to 150x150
        data = data.reshape((224, 224))

        # Convert to PIL Image (mode='L' for grayscale)
        image = Image.fromarray(data, mode="L")

        # Apply any augmentations or normalization transforms
        if self.transform:
            image = self.transform(image)

        # Return (tensor_image, label, group)
        # For Pytorch's training, we typically just return (image, label),
        # but we keep group as well for cross-validation splits outside the dataset.
        return image, label, group


def create_transforms():
    """
    Define any image transformations you'd like to apply, e.g. random augmentation.
    For classification, also typical to convert to tensor and normalize.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # If you want to normalize to [mean=0.5, std=0.5], for example:
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training with progress tracking.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # --- Configuration / Hyperparameters ---
    root_dir = (
        "../../dataset/ISAdetect/ISAdetect_full_dataset"  # your dataset root directory
    )
    batch_size = 8
    num_epochs = 1  # Increase for real training

    device = get_device()
    print("Using device:", device)

    # Create dataset with transformations
    dataset = BinaryEndiannessDataset(root_dir, transform=create_transforms())

    # We need to prepare for leave-one-group-out cross validation.
    # We'll extract the groups from the dataset and also the labels.
    groups = [
        sample[2] for sample in dataset.samples
    ]  # The group is at index 2 (file_path, label, group)
    labels = [sample[1] for sample in dataset.samples]  # The label is at index 1

    # Because we’re using scikit-learn’s LeaveOneGroupOut, we pass in:
    logo = LeaveOneGroupOut()

    fold = 1
    all_fold_accuracies = []

    for train_idx, test_idx in logo.split(
        X=range(len(dataset)), y=labels, groups=groups
    ):
        print(f"\n=== Fold {fold} ===")
        fold += 1

        # Build Subset for train and test
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Create a fresh MobileNetV3-Small model for each fold
        model = mobilenet_v3_small(num_classes=2, weights=None)
        # Modify first conv layer to accept 1 channel (grayscale) instead of 3 (RGB)
        model.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # --- Training Loop ---
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

        # --- Evaluation on this fold's test set ---
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Fold Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        all_fold_accuracies.append(test_acc)

    # Print overall performance across folds
    mean_acc = np.mean(all_fold_accuracies)
    std_acc = np.std(all_fold_accuracies)
    print(f"\nOverall LOGO CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()
