import torch
import torch.nn as nn

class CycleGAN(nn.Module):
  def __init__(self,
        y_gen, x_gen, y_desc, x_desc,
        generator_optimizer,
        y_desc_optimizer,
        x_desc_optimizer,
        gan_criterion,
        consistency_criterion,
        identiry_criterion 
      ) -> None:
    super().__init__()

    self.y_gen = y_gen
    self.x_gen = x_gen
    self.y_desc = y_desc
    self.x_desc = x_desc
    self.generator_optimizer = generator_optimizer
    self.y_desc_optimizer = y_desc_optimizer
    self.x_desc_optimizer = x_desc_optimizer
    self.gan_criterion = gan_criterion
    self.consistency_criterion = consistency_criterion
    

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


  def train_step(self, batch):
    x, y = batch[0], batch[1]
    gen_y, gen_x, rec_y, rec_x = self(x,y)

    #  L_Gan loss = 

