from typing import List
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class DatasetAssembly(Dataset):
  def __init__(self, picture_dataset:Dataset, monet_dataset:Dataset) -> None:
    self.monet = monet_dataset
    self.picture = picture_dataset

  def __len__(self):
    # unpaired: use max length
    return max(len(self.monet),len(self.picture))

  def __getitem__(self, idx:int):
    monet_item = self.monet[idx]
    picutre_item = self.picture[idx]

    return picutre_item, monet_item  