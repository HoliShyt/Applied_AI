from config import BATCH_SIZE, NUM_WORKERS

from typing import Tuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as tdata


# Download and pre-process MNIST dataset
# Argument trainval_split is the fraction of samples in the training set used for pure training
# (the rest will be used for validation of models)
def MNIST_dataset(trainval_split: float = 0.8):
    # Pre-processing for MNIST (tensorize and normalize between 0 and 1)
    MNIST_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # Getting train, validation and test sets
    MNIST_trainval = datasets.MNIST('./datasets/', train=True, download=True, transform=MNIST_transform)
    MNIST_train, MNIST_val = tdata.random_split(MNIST_trainval, [trainval_split, 1 - trainval_split])
    MNIST_test = datasets.MNIST('./datasets/', train=False, download=True, transform=MNIST_transform)

    return MNIST_train, MNIST_val, MNIST_test


# Download and pre-process EMNIST dataset (digits split)
# Argument trainval_split is the fraction of samples in the training set used for pure training
# (the rest will be used for validation of models)
def EMNIST_digits_dataset(trainval_split: float = 0.8):
    # Pre-processing for EMNIST (tensorize and normalize between 0 and 1)
    EMNIST_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])

    # Getting train, validation and test sets
    EMNIST_trainval = datasets.EMNIST('./datasets/', train=True, download=True, transform=EMNIST_transform, split='digits')
    EMNIST_train, EMNIST_val = tdata.random_split(EMNIST_trainval, [trainval_split, 1 - trainval_split])
    EMNIST_test = datasets.EMNIST('./datasets/', train=False, download=True, transform=EMNIST_transform, split='digits')

    return EMNIST_train, EMNIST_val, EMNIST_test


# Mappings of dataset names to corresponding download function
DATASET_MAPPINGS = {'mnist': MNIST_dataset,
                    'emnist_digits': EMNIST_digits_dataset}

# List of available datasets
AVAILABLE_DATASETS = list(DATASET_MAPPINGS.keys())


# Initialize all 3 dataloaders (train, validation, test) for a given dataset
def initialize_dataloaders(dataset_name: str, *args, **kwargs) -> Tuple[tdata.DataLoader, tdata.DataLoader, tdata.DataLoader]:
    print(f"Loading dataset '{dataset_name}'")

    # Check name, get associated download function and get samples
    if dataset_name.lower() not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.\n"
                         f"Supported values are {AVAILABLE_DATASETS}.")
    get_dataset = DATASET_MAPPINGS[dataset_name.lower()]
    data_train, data_val, data_test = get_dataset(*args, **kwargs)

    # Train dataloader, validation dataloader, testing dataloader
    trainloader = tdata.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validloader = tdata.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    testloader = tdata.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return trainloader, validloader, testloader
