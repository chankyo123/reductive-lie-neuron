import sys
sys.path.append('.')

import argparse
import os
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.lie_neurons_layers import *
from experiment.gl2_solid_layers import *
from data_loader.gl2_solid_data_loader import *
import time

def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            inputs = sample['inputs'].to(device)  # [batch_size, 25]
            targets = sample['targets'].to(device)  # [batch_size, 6]
            outputs = model(inputs)  # [batch_size, 6]
            loss = criterion(outputs, targets)
            loss_sum += loss.item()
        loss_avg = loss_sum / len(test_loader)
    return loss_avg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    parser = argparse.ArgumentParser(description='Test the elasticity prediction network')
    parser.add_argument('--training_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/gl2_solid/testing_param.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.test_config))

    test_dataset = ElasticityDataset(config['data_dir'], split='test', device=device)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle']
    )

    # Placeholder for your neural network
    model = None  # Replace with your model definition, e.g., ElasticityNN().to(device)
    if model is None:
        raise NotImplementedError("Please define your neural network model in the code.")

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}, test loss: {checkpoint['test loss']}")

    criterion = nn.MSELoss().to(device)
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()