import logging
import torch


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        :param patience: How long to wait after the last time validation loss improved.
                         Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
                      Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model = None  # Optional: To keep track of the best model

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def save_checkpoint(self, val_loss, model, filename="model_checkpoint.pth"):
        """
        Saves model when validation loss decreases.
        :param val_loss: Current epoch's validation loss
        :param model: The PyTorch model to save
        :param filename: Filename for the saved model checkpoint (optional)
        """
        if val_loss < self.min_validation_loss:
            logging.info(
                f"Validation loss decreased ({self.min_validation_loss:.6f} --> {val_loss:.6f}). Saving model to {filename}..."
            )
            torch.save(model.state_dict(), filename)
            self.min_validation_loss = val_loss
            self.best_model = model
