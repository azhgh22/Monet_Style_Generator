import torch
import random

# -------------------------------------------
#  Image Buffer (for discriminator smoothing)
# -------------------------------------------


class ImagePool:
    def __init__(self, size=50):
        self.size = size
        self.pool = []

    def query(self, images):
        """Return images from history or add new ones."""
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.pool) < self.size:
                self.pool.append(img)
                result.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.size - 1)
                    tmp = self.pool[idx].clone()
                    self.pool[idx] = img
                    result.append(tmp)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)