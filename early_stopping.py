import logging
import torch


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        :param patience: How long to wait after last time validation loss improved.
                         Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
                      Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, val_loss, model, filename="model_checkpoint.pth"):
        """
        Saves model when validation loss decrease.
        :param val_loss: Current epoch's validation loss
        :param model: The PyTorch model to save
        :param filename: Filename for the saved model checkpoint (optional)
        """
        logging.info(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {filename}..."
        )
        torch.save(model.state_dict(), filename)  # Save the model's state_dict
        self.val_loss_min = val_loss  # Update the minimum validation loss
