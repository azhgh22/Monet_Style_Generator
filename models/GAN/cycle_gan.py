import torch
import torch.nn as nn

class CycleGAN(nn.Module):
  def __init__(self,y_gen, x_gen, y_desc, x_desc) -> None:
    super().__init__()

    self.y_gen = y_gen
    self.x_gen = x_gen
    self.y_desc = y_desc
    self.x_desc = x_desc


  def forward(self, x, y):
    """
    x: batch from domain X
    y: batch from domain Y
    Returns generated images
    """
    fake_y = self.y_gen(x)
    fake_x = self.x_gen(y)
    rec_x = self.x_gen(fake_y)
    rec_y = self.y_gen(fake_x)

    return fake_y, fake_x, rec_y, rec_x