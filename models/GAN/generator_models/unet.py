import torch
import torch.nn as nn

class UNetGeneratorCUT(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = self.conv_block(in_ch, base)
        self.enc2 = self.conv_block(base, base * 2)
        self.enc3 = self.conv_block(base * 2, base * 4)
        self.enc4 = self.conv_block(base * 4, base * 8)

        self.pool = nn.AvgPool2d(2)

        # ---------- Bottleneck ----------
        self.bottleneck = self.conv_block(base * 8, base * 8)

        # ---------- Decoder ----------
        self.up4 = self.up_block(base * 8, base * 8)
        self.dec4 = self.conv_block(base * 8, base * 4)

        self.up3 = self.up_block(base * 4, base * 4)
        self.dec3 = self.conv_block(base * 4, base * 2)

        self.up2 = self.up_block(base * 2, base * 2)
        self.dec2 = self.conv_block(base * 2, base)

        self.up1 = self.up_block(base, base)
        self.dec1 = self.conv_block(base, base)

        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    # ---------- Blocks ----------
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_c, out_c):
        return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

    # ---------- CUT requirement ----------
    def encode(self, x):
        feats = []

        e1 = self.enc1(x)
        feats.append(e1)

        e2 = self.enc2(self.pool(e1))
        feats.append(e2)

        e3 = self.enc3(self.pool(e2))
        feats.append(e3)

        e4 = self.enc4(self.pool(e3))
        feats.append(e4)

        bottleneck = self.bottleneck(self.pool(e4))
        feats.append(bottleneck)

        return bottleneck, feats

    def forward(self, x):
        dim = 1 if len(x.shape)==4 else 0
        bottleneck, feats = self.encode(x)

        d4 = self.up4(bottleneck)
        concat = torch.cat([d4, feats[3]], dim=dim)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        return self.out(d1)
