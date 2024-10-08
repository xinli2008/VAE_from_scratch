import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    """
    Custom dataset for loading images

    Args:
        root(str): root directory containing image files.
        image_shape(tuple): desired shape of the output images.
    """
    def __init__(self, root, image_shape = (64, 64)) ->None:
        super().__init__()
        self.root = root
        self.image_shape = image_shape
        self.filenames = sorted(os.listdir(self.root))
        self.pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.image_shape),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        try:
            image = Image.open(path).convert("RGB")
            return self.pipeline(image)
        except Exception as e:
            raise IOError(f"Error loading image {path}: {e}")
        
    
def get_dataloader(root, batch_size, num_workers, **kwargs):
    """
    Create a dataloader for the custom dataset.

    Args:
        root(str): root directionary containing image files.
        batch_size(int): number of images per batch.
        num_workers(int): number of subprocess to use for data loading.
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"the provided root path {root} does not exists")
    
    dataset = MyDataset(root = root, **kwargs)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True)
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(root = "/mnt/VAE_from_scratch/data")
    img = next(iter(dataloader))  # img.shape: [b, c, h, w]
    b, c, h, w = img.shape
    assert b == 16

    img = torch.permute(img, (1, 0, 2, 3)) # c, b, h, w
    img = torch.reshape(img, (c, 4, 4 * h, w))  # c, 4, 4h, w
    img = torch.permute(img, (0, 2, 1, 3))  # c, 4h, 4, w
    img = torch.reshape(img, (c, 4*h, 4*w))
    img = transforms.ToPILImage()(img)
    img.save("./visualization.jpg")