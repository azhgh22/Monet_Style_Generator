import torch
import torch.nn as nn

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


# -----------------------------
# CUT-friendly Generator
# -----------------------------
class ResnetGeneratorCut(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super().__init__()

        # ----- Encoder -----
        self.enc_conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.resblocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh()
        )

    def encode(self, x):
        feats = []

        x = self.enc_conv1(x)
        feats.append(x)

        x = self.enc_down1(x)
        feats.append(x)

        x = self.enc_down2(x)
        feats.append(x)

        for i, block in enumerate(self.resblocks):
            x = block(x)
            if i in {0, 4}:  # use 1st and 5th residual blocks
                feats.append(x)

        return x, feats

    def forward(self, x):
        enc_out, feats = self.encode(x)
        out = self.decoder(enc_out)

        return out