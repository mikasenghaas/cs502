"""
Module for data loading and preprocessing.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset


# Utils
#######


def load_json(filname):
    """Load the data from a json file."""
    with open(filname) as f:
        data = json.load(f)
    return data


def get_token_freq(sequences):
    """Get the frequency of each token in the sequences."""
    token_freq = {}
    for seq in sequences:
        for token in seq:
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
    return token_freq


def split_sequences(sequences, train_ratio, valid_ratio, subsample=None, seed=42):
    """Get the train, valid and test datasets."""
    # Data shuffling
    np.random.seed(seed)
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)

    # Sub-sample the data
    if subsample is not None:
        indices = indices[:subsample]

    # Data splitting into train, validation and test sets
    train_size = int(len(indices) * train_ratio)
    valid_size = int(len(indices) * valid_ratio)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    test_indices = indices[train_size + valid_size :]
    train_sequences = [sequences[i] for i in train_indices]
    valid_sequences = [sequences[i] for i in valid_indices]
    test_sequences = [sequences[i] for i in test_indices]

    return train_sequences, valid_sequences, test_sequences


# Dataset
#########


class ProteinDataset(Dataset):
    def __init__(self, sequences, token2idx):
        self.token2idx = token2idx
        # Prepare the data by adding the special tokens <bos> and <eos>
        self.data, self.data_id = [], []
        for seq in sequences:
            tokens = ["<bos>"] + [token for token in seq] + ["<eos>"]
            self.data.append(tokens)
            self.data_id.append([token2idx[token] for token in self.data[-1]])

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Return the input and target token indices.

        Args:
            index (int): index of the sequence in the dataset.
        Returns:
            input_ids (list): list of input token indices.
            target_ids (list): list of target token indices.
        """
        input_ids = self.data_id[index][:-1]
        target_ids = self.data_id[index][1:]

        return input_ids, target_ids

    def padding_batch(self, batch):
        """
        Pad the batch to the longest sequence.

        Args:
            batch (list): list of (input_ids, target_ids) tuples.
        Returns:
            input_ids (torch.LongTensor): tensor of padded input sequences.
            target_ids (torch.LongTensor): tensor of padded target sequences.
        """
        # Get all input and target sequences
        input_ids = [d[0] for d in batch]
        target_ids = [d[1] for d in batch]

        # Get the max length in the batch
        max_len = max([len(i) for i in input_ids])

        # Pad the sequences
        # Hint: use self.token2idx['<pad>'] to get the padding token index
        for i in range(len(input_ids)):
            input_ids[i] += [self.token2idx["<pad>"]] * (max_len - len(input_ids[i]))
            target_ids[i] += [self.token2idx["<pad>"]] * (max_len - len(target_ids[i]))

        # Transform into tensors (useful for PyTorch DataLoaders)
        input_ids = torch.LongTensor(input_ids)
        target_ids = torch.LongTensor(target_ids)

        return input_ids, target_ids
