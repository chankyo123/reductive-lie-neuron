import sys
sys.path.append('.')

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ElasticityDataset(Dataset):
    def __init__(self, data_dir, split='train', train_ratio=0.8, seed=42, device='cuda'):
        """
        Args:
            data_dir (str): Directory containing CSV files (e.g., 'solid/elasticity_tensors').
            split (str): 'train' or 'test'.
            train_ratio (float): Fraction of unique rows for training.
            seed (int): Random seed for shuffling reproducibility.
            device (str): Device to load tensors onto.
        """
        self.data_dir = data_dir
        self.device = device
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.files.sort()

        # Load and concatenate all CSV files
        data_list = []
        for file in self.files:
            df = pd.read_csv(os.path.join(data_dir, file))
            data_list.append(df)
        self.full_data = pd.concat(data_list, ignore_index=True)

        # Remove duplicate rows (excluding 't' column, as it’s a timestamp)
        # Compare all columns except 't'
        cols_to_compare = [col for col in self.full_data.columns if col != 't']
        self.unique_data = self.full_data.drop_duplicates(subset=cols_to_compare, keep='first').reset_index(drop=True)

        # Shuffle and split rows
        np.random.seed(seed)
        indices = np.random.permutation(len(self.unique_data))
        num_train = int(train_ratio * len(self.unique_data))
        if split == 'train':
            self.indices = indices[:num_train]
        elif split == 'test':
            self.indices = indices[num_train:]
        else:
            raise ValueError("split must be 'train' or 'test'")
        self.data = self.unique_data.iloc[self.indices].reset_index(drop=True)

        # Define column groups
        self.strain_cols = [
            'eps_1_xx', 'eps_1_xy', 'eps_1_yx', 'eps_1_yy',
            'eps_2_xx', 'eps_2_xy', 'eps_2_yx', 'eps_2_yy',
            'eps_3_xx', 'eps_3_xy', 'eps_3_yx', 'eps_3_yy'
        ]
        self.stress_cols = [
            'sig_1_xx', 'sig_1_xy', 'sig_1_yx', 'sig_1_yy',
            'sig_2_xx', 'sig_2_xy', 'sig_2_yx', 'sig_2_yy',
            'sig_3_xx', 'sig_3_xy', 'sig_3_yx', 'sig_3_yy'
        ]
        self.bbr_col = ['bbr']
        self.c_cols = ['C_1111', 'C_1112', 'C_1122', 'C_1222', 'C_2222', 'C_1212']

        # Convert to tensors
        self.inputs = torch.tensor(
            self.data[self.strain_cols + self.stress_cols + self.bbr_col].values,
            dtype=torch.float32
        ).to(device)  # Shape: [N, 25]
        self.targets = torch.tensor(
            self.data[self.c_cols].values,
            dtype=torch.float32
        ).to(device)  # Shape: [N, 6]

        self.num_data = len(self.data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'inputs': self.inputs[idx],  # [25]
            'targets': self.targets[idx]  # [6]
        }
        return sample

if __name__ == "__main__":
    # Test the dataset
    dataset = ElasticityDataset('solid/elasticity_tensors', split='train', device='cpu')
    print(f"Total unique samples: {len(dataset.unique_data)}")
    print(f"Training samples after split: {len(dataset)}")
    print(f"Input shape: {dataset.inputs.shape}")
    print(f"Target shape: {dataset.targets.shape}")
    sample = dataset[0]
    print(f"Sample inputs: {sample['inputs']}")
    print(f"Sample targets: {sample['targets']}")