import sys
sys.path.append('.')

import argparse
import os
import yaml
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.lie_neurons_layers import *
from experiment.gl2_solid_layers import *
from data_loader.gl2_solid_data_loader import *
import time

def init_writer(config):
    writer = SummaryWriter(
        config['log_writer_path'] + "_" + str(time.localtime()),
        comment=config['model_description']
    )
    writer.add_text("data_dir", config['data_dir'])
    writer.add_text("model_save_path", config['model_save_path'])
    writer.add_text("log_writer_path", config['log_writer_path'])
    writer.add_text("shuffle", str(config['shuffle']))
    writer.add_text("batch_size", str(config['batch_size']))
    writer.add_text("init_lr", str(config['initial_learning_rate']))
    writer.add_text("num_epochs", str(config['num_epochs']))
    return writer

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

def train(model, train_loader, test_loader, config, device='cpu'):
    writer = init_writer(config)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['initial_learning_rate'],
        weight_decay=config['weight_decay_rate']
    )

    if config['resume_training']:
        print(f"Resuming training from {config['resume_model_path']}")
        checkpoint = torch.load(config['resume_model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Start from epoch {start_epoch}, loss: {checkpoint['loss']}, test loss: {checkpoint['test loss']}")
    else:
        start_epoch = 0

    best_loss = float("inf")
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        loss_sum = 0.0
        running_loss = 0.0
        optimizer.zero_grad()
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            inputs = sample['inputs'].to(device)  # [batch_size, 25]
            targets = sample['targets'].to(device)  # [batch_size, 6]
            outputs = model(inputs)  # [batch_size, 6]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()
            running_loss += loss.item()
            if i % config['print_freq'] == 0 and i > 0:
                print(f"Epoch {epoch}/{config['num_epochs']}, Iter {i}/{len(train_loader)}, Loss: {running_loss/config['print_freq']:.8f}")
                running_loss = 0.0

        train_loss = loss_sum / len(train_loader)
        test_loss = test(model, test_loader, criterion, device)

        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('test loss', test_loss, epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'test loss': test_loss
            }
            torch.save(state, config['model_save_path'] + '_best_test_loss.pt')

        print(f"Epoch {epoch}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'test loss': test_loss
        }
        torch.save(state, config['model_save_path'] + '_last_epoch.pt')

    writer.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    parser = argparse.ArgumentParser(description='Train the elasticity prediction network')
    parser.add_argument('--training_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/gl2_solid/training_param.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.training_config))

    train_dataset = ElasticityDataset(config['data_dir'], split='train', device=device)
    test_dataset = ElasticityDataset(config['data_dir'], split='test', device=device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle']
    )
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
    train(model, train_loader, test_loader, config, device)

if __name__ == "__main__":
    main()