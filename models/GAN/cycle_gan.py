import torch
import itertools
from utils.image_pool import ImagePool


# ---------------------------------------------------
#               Linear decay scheduler
# ---------------------------------------------------
def lambda_rule(epoch, start_decay=100, total=200):
    """Linear decay after start_decay epoch."""
    epoch = max(0, epoch - start_decay)
    return 1.0 - epoch / (total - start_decay)


# ---------------------------------------------------
#               CycleGAN Model Class
# ---------------------------------------------------
class CycleGAN(torch.nn.Module):
    def __init__(self, G_AB, G_BA, D_A, D_B, lambda_func=lambda_rule):
        super().__init__()
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A  = D_A
        self.D_B  = D_B

        self.labmda_func = lambda_func

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Image buffers
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # Cycle consistency loss weight
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5 * self.lambda_cycle

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

        self.optimizer_D_A = torch.optim.Adam(
            D_A.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        self.optimizer_D_B = torch.optim.Adam(
            D_B.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Learning rate schedulers (linear decay after epoch 100)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lambda epoch: self.labmda_func(epoch)
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=lambda epoch: self.labmda_func(epoch)
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=lambda epoch: self.labmda_func(epoch)
        )


    # ---------------------------------------------------
    #                  Forward for inference
    # ---------------------------------------------------
    def forward(self, real_A, real_B):
        return self.G_AB(real_A), self.G_BA(real_B)


    # ---------------------------------------------------
    #                  One training iteration
    # ---------------------------------------------------
    def train_step(self, real_A, real_B):
        # ==========================
        #    Train Generators
        # ==========================
        self.optimizer_G.zero_grad()

        # ----- Identity Loss -----
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)

        loss_idt_A = self.criterion_identity(idt_A, real_A) * self.lambda_identity
        loss_idt_B = self.criterion_identity(idt_B, real_B) * self.lambda_identity

        # ----- GAN Loss -----
        fake_B = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # ----- Cycle Loss -----
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)

        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        # Total G loss
        loss_G = (
            loss_GAN_AB + loss_GAN_BA +
            loss_cycle_A + loss_cycle_B +
            loss_idt_A + loss_idt_B
        )

        loss_G.backward()
        self.optimizer_G.step()

        # ==========================
        #    Train D_A
        # ==========================
        self.optimizer_D_A.zero_grad()

        pred_real_A = self.D_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))

        fake_A_buffered = self.fake_A_pool.query(fake_A.detach())
        pred_fake_A = self.D_A(fake_A_buffered)
        loss_D_fake = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))

        loss_D_A = 0.5 * (loss_D_real + loss_D_fake)
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # ==========================
        #    Train D_B
        # ==========================
        self.optimizer_D_B.zero_grad()

        pred_real_B = self.D_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

        fake_B_buffered = self.fake_B_pool.query(fake_B.detach())
        pred_fake_B = self.D_B(fake_B_buffered)
        loss_D_fake = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

        loss_D_B = 0.5 * (loss_D_real + loss_D_fake)
        loss_D_B.backward()
        self.optimizer_D_B.step()

        return {
            "G": loss_G.item(),
            "GAN_AB": loss_GAN_AB.item(),
            "GAN_BA": loss_GAN_BA.item(),
            "cycle_A": loss_cycle_A.item(),
            "cycle_B": loss_cycle_B.item(),
            "idt_A": loss_idt_A.item(),
            "idt_B": loss_idt_B.item(),
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
        }
    
    def epoch_step(self):
      self.lr_scheduler_G.step()
      self.lr_scheduler_D_A.step()
      self.lr_scheduler_D_B.step()

    def get_init_loss_dict(self):
      return {
        "G": 0.0, "GAN_AB": 0.0, "GAN_BA": 0.0,
        "cycle_A": 0.0, "cycle_B": 0.0,
        "idt_A": 0.0, "idt_B": 0.0,
        "D_A": 0.0, "D_B": 0.0
      }

    def get_model_state(self, epoch):
        return {
            "epoch": epoch,
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_G": self.optimizer_G.state_dict(),
            "opt_D_A": self.optimizer_D_A.state_dict(),
            "opt_D_B": self.optimizer_D_B.state_dict(),
            "sched_G": self.lr_scheduler_G.state_dict(),
            "sched_D_A": self.lr_scheduler_D_A.state_dict(),
            "sched_D_B": self.lr_scheduler_D_B.state_dict(),
        }

    def load_state(self, checkpoint):
      # Load model weights
      self.G_AB.load_state_dict(checkpoint["G_AB"])
      self.G_BA.load_state_dict(checkpoint["G_BA"])
      self.D_A.load_state_dict(checkpoint["D_A"])
      self.D_B.load_state_dict(checkpoint["D_B"])

      # Load optimizers
      self.optimizer_G.load_state_dict(checkpoint["opt_G"])
      self.optimizer_D_A.load_state_dict(checkpoint["opt_D_A"])
      self.optimizer_D_B.load_state_dict(checkpoint["opt_D_B"])

      # Load schedulers
      self.lr_scheduler_G.load_state_dict(checkpoint["sched_G"])
      self.lr_scheduler_D_A.load_state_dict(checkpoint["sched_D_A"])
      self.lr_scheduler_D_B.load_state_dict(checkpoint["sched_D_B"])


