import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from utils.image_pool import ImagePool

# -----------------------------
# MLP Head for PatchNCE
# -----------------------------
class PatchSampleMLP(nn.Module):
    """
    MLP projection head for a single encoder layer
    """
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, C, N) where N = H*W flattened
        x = x.transpose(1, 2)  # (B, N, C)
        x = self.model(x)      # (B, N, output_dim)
        x = F.normalize(x, dim=2)
        return x

# -----------------------------
# CUT Model
# -----------------------------
class Cut(nn.Module):
    def __init__(self, generator, discriminator, nce_layers=[0,1,2,3,4], lambda_nce=1.0, lambda_identity=0.5,layer_chanels_maping=None):
        """
        generator: CUT-friendly generator
        discriminator: PatchGAN discriminator
        nce_layers: which encoder features to use for PatchNCE
        """
        super().__init__()

        if layer_chanels_maping==None:
          self.mapping = mapping = {
            0: 64,   # enc_conv1
            1: 128,  # enc_down1
            2: 256,  # enc_down2
            3: 256,  # resblock 0
            4: 256   # resblock 4
          }
        else:
          self.mapping = layer_chanels_maping


        self.G = generator
        self.D = discriminator

        self.nce_layers = nce_layers
        self.lambda_nce = lambda_nce
        self.lambda_identity = lambda_identity

        # Create MLPs for each selected layer
        self.mlps = nn.ModuleList([PatchSampleMLP(self._get_layer_channels(i)) for i in nce_layers])

        # Losses
        self.criterion_gan = nn.MSELoss()
        self.criterion_nce = nn.CrossEntropyLoss()

        # Image buffers
        self.fake_B_pool = ImagePool(50)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda epoch: 1.0)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda epoch: 1.0)

    def _get_layer_channels(self, layer_idx):
        # channels for selected encoder layers
        return self.mapping[layer_idx]

    def sample_patches_same(self, feat_q, feat_k, num_patches=256):
        """
        Sample same patch locations from both query (fake) and key (real) features
        """
        B, C, H, W = feat_q.shape
        N = H * W
        idx = torch.randint(0, N, (num_patches,), device=feat_q.device)
        feat_q = feat_q.view(B, C, -1)[:, :, idx]
        feat_k = feat_k.view(B, C, -1)[:, :, idx]
        return feat_q, feat_k

    def patchnce_loss(self, feats_q, feats_k):
        """
        feats_q: list of sampled fake features, each (B, C, N)
        feats_k: list of sampled real features, each (B, C, N)
        """
        total_loss = 0.0
        temperature = 0.07

        for f_q, f_k, mlp in zip(feats_q, feats_k, self.mlps):
            # Project + normalize
            q = mlp(f_q)  # (B, N, C)
            k = mlp(f_k)  # (B, N, C)

            # IMPORTANT: stop gradient on keys
            k = k.detach()

            B, N, C = q.shape

            # Flatten across batch → global negatives
            q = q.reshape(B * N, C)  # (BN, C)
            k = k.reshape(B * N, C)  # (BN, C)

            # InfoNCE logits
            logits = torch.mm(q, k.t()) / temperature  # (BN, BN)

            # Positive pairs are diagonal
            labels = torch.arange(B * N, device=q.device)

            loss = self.criterion_nce(logits, labels)
            total_loss += loss

        return total_loss / len(feats_q)


    def forward(self, x):
        return self.G(x)

    def train_step(self, real_X, real_Y):
        # ----------------------
        # Train Generator
        # ----------------------
        self.optimizer_G.zero_grad()
        fake_Y = self.G(real_X)

        # ----- GAN Loss -----
        pred_fake = self.D(fake_Y)
        target_real = torch.ones_like(pred_fake)
        loss_gan = self.criterion_gan(pred_fake, target_real)

        # ----- PatchNCE Loss -----
        _, feats_real = self.G.encode(real_X)
        _, feats_fake = self.G.encode(fake_Y)

        # Sample patches
        feats_real_sampled = []
        feats_fake_sampled = []
        for i in self.nce_layers:
            fq, fk = self.sample_patches_same(feats_fake[i], feats_real[i])
            feats_fake_sampled.append(fq)
            feats_real_sampled.append(fk)

        loss_nce = self.patchnce_loss(feats_fake_sampled, feats_real_sampled)

        # ----- Identity Loss -----
        idt_Y = self.G(real_Y)
        loss_idt = F.l1_loss(idt_Y, real_Y)

        loss_G = loss_gan + self.lambda_nce * loss_nce + self.lambda_identity * loss_idt
        loss_G.backward()
        self.optimizer_G.step()

        # ----------------------
        # Train Discriminator
        # ----------------------
        self.optimizer_D.zero_grad()

        # Real images
        pred_real = self.D(real_Y)
        target_real = torch.ones_like(pred_real)

        # Fake images (from pool)
        fake_Y_pool = self.fake_B_pool.query(fake_Y.detach())
        pred_fake = self.D(fake_Y_pool)
        target_fake = torch.zeros_like(pred_fake)

        loss_D = 0.5 * (
            self.criterion_gan(pred_real, target_real) +
            self.criterion_gan(pred_fake, target_fake)
        )

        loss_D.backward()
        self.optimizer_D.step()


        # self.optimizer_D.zero_grad()
        # pred_real = self.D(real_Y)
        # pred_fake = self.D(fake_Y.detach())
        # target_real = torch.ones_like(pred_real)
        # target_fake = torch.zeros_like(pred_fake)
        # loss_D = 0.5 * (self.criterion_gan(pred_real, target_real) +
        #                 self.criterion_gan(pred_fake, target_fake))
        # loss_D.backward()
        # self.optimizer_D.step()

        return {
            "G": loss_G.item(),
            "GAN": loss_gan.item(),
            "PatchNCE": loss_nce.item(),
            "Identity": loss_idt.item(),
            "D": loss_D.item()
        }

    def epoch_step(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()

    def get_init_loss_dict(self):
      return {
            "G": 0.0,
            "GAN": 0.0,
            "PatchNCE": 0.0,
            "Identity": 0.0,
            "D": 0.0
        }

    def get_model_state(self, epoch):
      return {
        "epoch": epoch,
        "G": self.G.state_dict(),
        "D": self.D.state_dict(),
        "MLP" : self.mlps.state_dict(),
        "opt_G": self.optimizer_G.state_dict(),
        "opt_D": self.optimizer_D.state_dict(),
        "sched_G": self.lr_scheduler_G.state_dict(),
        "sched_D": self.lr_scheduler_D.state_dict(),
      }

    def load_state(self, checkpoint):
      # Load model weights
      self.G.load_state_dict(checkpoint["G"])
      self.D.load_state_dict(checkpoint["D"])
      self.mlps.load_state_dict(checkpoint["MLP"])

      # Load optimizer states
      self.optimizer_G.load_state_dict(checkpoint["opt_G"])
      self.optimizer_D.load_state_dict(checkpoint["opt_D"])

      # Load schedulers
      self.lr_scheduler_G.load_state_dict(checkpoint["sched_G"])
      self.lr_scheduler_D.load_state_dict(checkpoint["sched_D"])