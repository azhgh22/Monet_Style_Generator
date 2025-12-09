import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    Input: 3 x H x W image
    Output: 1 x H' x W' feature map of "realness" per patch
    """
    def __init__(self, input_channels=3, ndf=64):
        """
        input_channels: usually 3 (RGB)
        ndf: base number of filters
        """
        super().__init__()

        # ---- C64 ----
        model = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # ---- C128 ----
        model += [
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # ---- C256 ----
        model += [
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # ---- C512 ----
        model += [
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # ---- Output Layer ----
        model += [
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)