from typing import List
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class CustomDataset(Dataset):
    def __init__(self, data_files:List[str],transforms = None,size=-1)->None:
        self.paths = data_files
        if size==-1:
          self.length = len(self.paths)
        else:
          self.length = size
        if transforms==None:
          self.transform = T.Compose([
              T.ToTensor(),
              T.ConvertImageDtype(torch.float)
            ]) 
        else:
          self.transform = transforms

    def __len__(self):
        # unpaired: use max length
        return self.length

    def __getitem__(self, idx:int):
        path = self.paths[idx % self.length]
        
        img = Image.open(path).convert("RGB")
        
        return self.transform(img)