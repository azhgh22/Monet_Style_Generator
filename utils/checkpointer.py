class Checkpointer:
  def __init__(self,checkpoint_dir, model_name) -> None:
    self.checkpoint_dir = checkpoint_dir
    self.model_name = model_name

  def save(self, epoch ,state_dict):
    pass

  def load(self):
    pass
