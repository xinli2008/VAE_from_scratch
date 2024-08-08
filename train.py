from time import time
import torch
import torch.nn as nn
from model import VAE
from dataset import get_dataloader
import torch.nn.functional as F
from torchvision import transforms
import argparse
import os
import torch.optim as optim
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for training VAE model")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--kl_weight", type=float, default=0.00025, help="Weight for the KL divergence loss component")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", ], help="Optimizer type (e.g., 'adam', 'sgd', 'rmsprop', 'adagrad')")
    parser.add_argument("--data_path", type=str, default="/mnt/VAE_from_scratch/data", help="data root path")
    parser.add_argument("--output_path", type=str, default="/mnt/VAE_from_scratch/models", help="output model path")
    args = parser.parse_args()
    return args

def loss_fn(y, output, mean, var):
    reconstruction_loss = F.mse_loss(y, output)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mean**2 - torch.exp(var), 1), 0)
    loss = reconstruction_loss + kl_loss * args.kl_weight
    return loss


def train(model, dataloader, args, device):
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    else:
        raise ValueError(f"unsupported optimizer type with {args.optimizer}")
    
    length_dataset = len(dataloader.dataset)
    begin_time = time()

    # train
    for i in range(args.epoch):
        loss_sum = 0
        model.train()
        for x in tqdm(dataloader, desc = f'Epoch {i + 1}/{args.epoch}'):
            x = x.to(device)
            output, mean, var = model(x)
            loss = loss_fn(x, output, mean, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum = loss + loss_sum
        
        avg_loss = loss_sum / length_dataset
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {avg_loss} {minute}:{second}')
        output_path = os.path.join(args.output_path, f"model_{i}.pth")
        torch.save(model.state_dict(), output_path)

def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader(args.data_path, args.batch_size, args.num_workers)
    model = VAE().to(device = device)
    train(model, dataloader, args, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)