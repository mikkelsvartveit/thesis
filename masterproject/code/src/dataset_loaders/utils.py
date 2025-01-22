import torch
from torch.utils.data import Dataset


def random_train_test_split(dataset: Dataset, test_split=0.2, seed=420):

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.manual_seed(seed),
    )

    return train_dataset, test_dataset
