import torch
import torch.nn as nn
from model import VAE
from torchvision import transforms
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter for sampling image")
    parser.add_argument("--model_path", type=str, default="/mnt/VAE_from_scratch/models/model_29.pth", help="model_path for loading")
    parser.add_argument("--output_path", type=str, default="/mnt/VAE_from_scratch/output", help="folder path for save images")
    parser.add_argument("--num_samples", type=int, default=4, help="number of samples for generating images")
    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained VAE model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"the provided path: {model_path} not exists.")
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()
    return model

def save_images(model, output_path, num_samples, device):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with torch.no_grad():
        for i in range(num_samples):
            sample = model.sample(device = device)
            sample = sample.squeeze(0).cpu()
            sample_img = transforms.ToPILImage()(sample)
            sample_img.save(os.path.join(output_path, f"sample_{i + 1}.png"))
            print(f"sample {i+1} saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)
    save_images(model, args.output_path, args.num_samples, device)