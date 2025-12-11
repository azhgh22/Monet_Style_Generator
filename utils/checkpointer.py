import os
import torch

class Checkpointer:
    def __init__(self, checkpoint_dir, model_name, save_every,del_prev=True) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_every = save_every
        self.del_prev = del_prev

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, epoch, state_dict):
        # Only save every N epochs
        if epoch % self.save_every != 0:
            return

        filename = f"{self.model_name}_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save(state_dict, path)

        if self.del_prev:
          for fl in os.listdir(self.checkpoint_dir):
            if fl != filename:
                os.remove(os.path.join(self.checkpoint_dir, fl))



        print(f"Checkpoint saved: {path}")

    def load(self):
        # List all checkpoint files
        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.model_name) and f.endswith(".pt")
        ]

        if not files:
            print("No checkpoints found.")
            return None  # (state_dict, epoch)

        # Sort by epoch number
        def extract_epoch(fname):
            # model_name_epoch_XX.pt → extract XX
            parts = fname.replace(".pt", "").split("_")
            return int(parts[-1])

        files.sort(key=extract_epoch)
        latest = files[-1]

        path = os.path.join(self.checkpoint_dir, latest)
        checkpoint = torch.load(path, map_location="cpu")
        epoch = extract_epoch(latest)

        print(f"Loaded checkpoint from: {path}")
        return checkpoint
