import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import re
from prettytable import PrettyTable
import matplotlib.pyplot as plt


# Define a dataset class for transliteration data
class TransliterationDataset(Dataset):
    def __init__(self, data_path, input_lang, target_lang, max_len=50):
        """
        Args:
            data_path: Path to the data file with pairs of words
            input_lang: The input language/script processor object
            target_lang: The target language/script processor object
            max_len: Maximum sequence length
        """
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.max_len = max_len
        self.pairs = []

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Format may vary depending on dataset
                    latin, native = parts[0], parts[1]
                    if latin and native and len(latin) <= max_len and len(native) <= max_len:
                        self.pairs.append((latin, native))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        latin_word, native_word = self.pairs[idx]

        # Convert characters to indices
        input_tensor = self.input_lang.word_to_tensor(latin_word, self.max_len)
        target_tensor = self.target_lang.word_to_tensor(native_word, self.max_len)

        return input_tensor, target_tensor

# Language class to handle character-level processing
class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_chars = 3  # Count PAD, SOS and EOS

    def add_word(self, word):
        for char in word:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def word_to_tensor(self, word, max_len):
        """Convert a word to tensor of character indices with padding"""
        indices = [self.char2index.get(char, 0) for char in word]
        indices = [self.char2index["SOS"]] + indices + [self.char2index["EOS"]]

        # Pad sequence to max_len
        if len(indices) < max_len + 2:  # +2 for SOS and EOS
            indices += [self.char2index["PAD"]] * (max_len + 2 - len(indices))
        else:
            indices = indices[:max_len + 2]

        return torch.tensor(indices, dtype=torch.long)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# Function to prepare data for a language pair
def prepare_data(data_path, max_len=50, input_lang=None, output_lang=None): # Add input_lang and output_lang arguments with default None
    if input_lang is None:  # If not provided, create new Lang objects
        input_lang = Lang("latin")
    if output_lang is None:
        output_lang = Lang("native")

    # Read data and build vocabulary
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                latin, native = parts[0], parts[1]
                if latin and native and len(latin) <= max_len and len(native) <= max_len:
                    input_lang.add_word(latin)
                    output_lang.add_word(native)
                    pairs.append((latin, native))

    print(f"Prepared {len(pairs)} pairs")
    print(f"Input vocabulary size: {input_lang.n_chars}")
    print(f"Output vocabulary size: {output_lang.n_chars}")

    return input_lang, output_lang, pairs