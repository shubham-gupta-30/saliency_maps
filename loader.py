

from torch.utils.data import Dataset, Dataloader, random_split
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from datasets import load_dataset
import os



class TinyImageNetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
