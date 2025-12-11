class Train:
  def __init__(self, model, num_epochs, train_loader,checkpointer,device) -> None:
    self.model = model
    self.num_epochs = num_epochs
    self.train_loader = train_loader
    self.chechpointer = checkpointer
    self.device = device

    self.epoch_losses = []
    self.current_epoch = 1

  def train(self):
    for epoch in range(self.current_epoch, self.num_epochs+1):
      epoch_losses = {
        "G": 0.0, "GAN_AB": 0.0, "GAN_BA": 0.0,
        "cycle_A": 0.0, "cycle_B": 0.0,
        "idt_A": 0.0, "idt_B": 0.0,
        "D_A": 0.0, "D_B": 0.0
      }

      for p,m in self.train_loader:
        # Send data to device
        p = p.to(self.device)
        m = m.to(self.device)

        # Perform one training step
        losses = self.model.train_step(p, m)

        # Accumulate losses for logging
        for k in epoch_losses.keys():
            epoch_losses[k] += losses[k]

        num_batches += 1

      # Average losses over epoch
      for k in epoch_losses:
          epoch_losses[k] /= num_batches

      self.epoch_losses.append(epoch_losses)
      self.model.epoch_step()

      self.chechpointer.save(epoch,{
        "model_state" : self.model.get_model_state(),
        "epoch_losses" : self.epoch_losses,
        "current_epoch" : epoch
      })

    return self.epoch_losses

  def load_checkpoint(self):
    persist_dict = self.chechpointer.load()
    self.epoch_losses = persist_dict["epoch_losses"]
    self.current_epoch = persist_dict["current_epoch"]
    self.model.load_state(persist_dict["model_state"])


