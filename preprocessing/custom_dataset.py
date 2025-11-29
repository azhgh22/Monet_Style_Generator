from typing import List
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class CustomDataset(Dataset):
    def __init__(self, data_files:List[str])->None:
        self.paths = data_files
        self.length = len(self.paths)

        self.transform = T.Compose([
            T.ToTensor(),                     # [0,1]
          ])

    def __len__(self):
        # unpaired: use max length
        return self.length

    def __getitem__(self, idx:int):
        path = self.paths[idx % self.length]
        
        img = Image.open(path).convert("RGB")
        
        return self.transform(img)