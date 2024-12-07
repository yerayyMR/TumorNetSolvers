import numpy as np
from typing import List, Tuple

def train_val_test_split_fx(data_identifiers: List[str], train_size: int, val_size: int = 2000, test_size: int = 2000, seed: int = 12345, fixed_val_test: Tuple[List[str], List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the data into training, validation, and test sets.
    
    :param data_identifiers: List of identifiers for the dataset.
    :param train_size: Number of samples to be used for training.
    :param val_size: Number of samples to be used for validation.
    :param test_size: Number of samples to be used for testing.
    :param seed: Seed for random number generator for reproducibility.
    :param fixed_val_test: A tuple of (validation_set, test_set) if you want to keep them fixed.
    :return: A tuple containing the training, validation, and test identifiers.
    """
    
    np.random.seed(seed)

    shuffled_identifiers = np.random.permutation(data_identifiers)

    if fixed_val_test is not None:
        fixed_val, fixed_test = fixed_val_test

        # Filter out the fixed validation and test sets from the shuffled data
        shuffled_identifiers = [item for item in shuffled_identifiers if item not in fixed_val and item not in fixed_test]

        # Combine the remaining data with the fixed val/test sets
        train_keys = list(shuffled_identifiers[:train_size])
        val_keys = list(fixed_val)
        test_keys = list(fixed_test)
        
    else:
        # Ensure the total requested sizes do not exceed the number of available data points
        total_size = train_size + val_size + test_size
        if total_size > len(data_identifiers):
            raise ValueError(f"Total size (train + val + test) exceeds the number of available data samples ({len(data_identifiers)}).")

        train_keys = list(shuffled_identifiers[:train_size])
        val_keys = list(shuffled_identifiers[train_size:train_size + val_size])
        test_keys = list(shuffled_identifiers[train_size + val_size:train_size + val_size + test_size])
    
    return train_keys, val_keys, test_keys


def train_val_test_split_size(data_identifiers: List[str], train_size: int, val_size: int = 2000, test_size: int = 2000, seed: int = 12345) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the data into training, validation, and test sets.
    
    :param data_identifiers: List of identifiers for the dataset.
    :param train_size: Number of samples to be used for training.
    :param val_size: Number of samples to be used for validation.
    :param test_size: Number of samples to be used for testing.
    :param seed: Seed for random number generator for reproducibility.
    :return: A tuple containing the training, validation, and test identifiers.
    """
    #random seed for reproducibility
    np.random.seed(seed)

    shuffled_identifiers = np.random.permutation(data_identifiers)

    # Ensure the total requested sizes do not exceed the number of available data points
    total_size = train_size + val_size + test_size
    if total_size > len(data_identifiers):
        raise ValueError(f"Total size (train + val + test) exceeds the number of available data samples ({len(data_identifiers)}).")

    train_keys = list(shuffled_identifiers[:train_size])
    val_keys = list(shuffled_identifiers[train_size:train_size + val_size])
    test_keys = list(shuffled_identifiers[train_size + val_size:train_size + val_size + test_size])
    
    return train_keys, val_keys, test_keys



def train_val_test_split_ratio(data_identifiers: List[str], val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 12345) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits the data into training, validation, and test sets.
    
    :param data_identifiers: List of identifiers for the dataset.
    :param val_ratio: Ratio of the data to be used for validation.
    :param test_ratio: Ratio of the data to be used for testing.
    :param seed: Seed for random number generator for reproducibility.
    :return: A tuple containing the training, validation, and test identifiers.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Shuffle the data identifiers
    shuffled_identifiers = np.random.permutation(data_identifiers)

    # Calculate the split indices
    val_split_index = int(len(shuffled_identifiers) * (1 - val_ratio - test_ratio))
    test_split_index = int(len(shuffled_identifiers) * (1 - test_ratio))

    # Create training, validation, and test sets
    train_keys = list(shuffled_identifiers[:val_split_index])
    val_keys = list(shuffled_identifiers[val_split_index:test_split_index])
    test_keys = list(shuffled_identifiers[test_split_index:])

    return train_keys, val_keys, test_keys




#From nnUnet Framework
from typing import List

import numpy as np
from sklearn.model_selection import KFold


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits
