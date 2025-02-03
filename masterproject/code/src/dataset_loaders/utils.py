import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Subset


def random_train_test_split(
    dataset: Dataset, test_split=0.2, seed=420
) -> tuple[Subset, Subset]:

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size

    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.manual_seed(seed),
    )

    return train_dataset, test_dataset


def architecture_metadata_info(architecture_path, architecture_name):
    architecture_path = Path(architecture_path)
    # get file with achitecture_name.json
    architecture_path = architecture_path / f"{architecture_name}.json"

    with open(architecture_path, "r") as f:
        metadata = json.load(f)
        # get first as relevant metadata are the same for all files
        metadata = metadata[0]

    metadata = {
        key: metadata[key] for key in ["architecture", "endianness", "wordsize"]
    }

    return metadata


def get_architecture_features(csv_file, architecture):
    """
    Takes a CSV file containing ISA features and an architecture name,
    and returns a dictionary of features for that architecture.

    Parameters:
        csv_file (str): Path to the CSV file with ISA features.
        architecture (str): Name of the architecture to lookup.

    Returns:
        dict: Dictionary containing features for the given architecture.
              Returns an empty dictionary if the architecture is not found.
    """
    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(csv_file, delimiter=";")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Ensure the architecture exists in the data
    if architecture not in df.iloc[:, 0].values:
        return {}

    # Filter the row corresponding to the given architecture
    row = df[df.iloc[:, 0] == architecture].iloc[0]

    # Convert the row to a dictionary
    features_dict = row.to_dict()

    return features_dict
