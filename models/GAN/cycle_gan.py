import torch
import itertools
from utils.image_pool import ImagePool

# ---------------------------------------------------
#              CycleGAN Class
# ---------------------------------------------------
class CycleGAN(torch.nn.Module):
    def __init__(self, G_AB, G_BA, D_A, D_B):
        super().__init__()
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A  = D_A
        self.D_B  = D_B

        # self.device = device

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Buffers for improved discriminator stability
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # One optimizer for both generators
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

        # One optimizer per discriminator
        self.optimizer_D_A = torch.optim.Adam(
            D_A.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        self.optimizer_D_B = torch.optim.Adam(
            D_B.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Cycle consistency weight
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5 * self.lambda_cycle


    # ---------------------------------------------------
    # Forward pass (used only for evaluation/inference)
    # ---------------------------------------------------
    def forward(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        return fake_A, fake_B


    # ---------------------------------------------------
    #          Single training step
    # ---------------------------------------------------
    def train_step(self, real_A, real_B):

        # ===============================
        #   Train Generators G_AB + G_BA
        # ===============================
        self.optimizer_G.zero_grad()

        # Identity loss (optional but improves color preservation)
        idt_B = self.G_AB(real_B)
        idt_A = self.G_BA(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_A) * self.lambda_identity
        loss_idt_B = self.criterion_identity(idt_B, real_B) * self.lambda_identity

        # GAN loss
        fake_B = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B)
        valid_label = torch.ones_like(pred_fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, valid_label)

        fake_A = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, valid_label)

        # Cycle loss
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        # Total generator loss
        loss_G = (
            loss_GAN_AB + loss_GAN_BA +
            loss_cycle_A + loss_cycle_B +
            loss_idt_A + loss_idt_B
        )

        loss_G.backward()
        self.optimizer_G.step()

        # ===============================
        #       Train Discriminator A
        # ===============================
        self.optimizer_D_A.zero_grad()

        # Real
        pred_real = self.D_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        # Fake (buffered)
        fake_A_buffered = self.fake_A_pool.query(fake_A.detach())
        pred_fake = self.D_A(fake_A_buffered)
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # ===============================
        #       Train Discriminator B
        # ===============================
        self.optimizer_D_B.zero_grad()

        pred_real = self.D_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

        fake_B_buffered = self.fake_B_pool.query(fake_B.detach())
        pred_fake = self.D_B(fake_B_buffered)
        loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()

        # ------------------------------------------
        # return all components for logging
        # ------------------------------------------
        return {
            "loss_G": loss_G.item(),
            "loss_GAN_AB": loss_GAN_AB.item(),
            "loss_GAN_BA": loss_GAN_BA.item(),
            "loss_cycle_A": loss_cycle_A.item(),
            "loss_cycle_B": loss_cycle_B.item(),
            "loss_idt_A": loss_idt_A.item(),
            "loss_idt_B": loss_idt_B.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
        }