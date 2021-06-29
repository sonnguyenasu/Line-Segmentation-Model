from trainer import Trainer
from models import UNet
from datasets import ImageDataset
import torch
import argparse
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--log_frequency', default=10,
                        type=int, help='Frequency to print out info')
    parser.add_argument('-sf', '--save-frequency', default=1000,
                        type=int, help='Frequency to save data')
    parser.add_argument('-e', '--epoch', default=10,
                        type=int, help='Number of epochs')
    parser.add_argument('-d', '--device', default='cuda',
                        type=str, help='device')
    parser.add_argument('-sp', '--save-path', default='output', type=str, help='Path to save model weights')
    parser.add_argument('--resume', default=None,
                        help='path to load weight, default to None means no weight loaded')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model = UNet(3,1,bilinear=False).to('cuda')
    data_path = './data/'
    
    loader = torch.utils.data.DataLoader(ImageDataset(
        data_path, transforms=None, mode="train"), batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ImageDataset(
        data_path, transforms=None, mode="test"), batch_size=1, shuffle=True)
    args = get_args()
    trainer = Trainer(args, model, loader, test_loader)
    trainer.train()
