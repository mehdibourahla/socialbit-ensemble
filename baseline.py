import torch.nn as nn
import torch
from utils import log_message, EarlyStopping
import time


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        class_weights_tensor=None,
        input_size=1024,
        hidden_size=150,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.25,
    ):
        super(BiLSTMModel, self).__init__()
        self.class_weights_tensor = class_weights_tensor
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.class_weights_tensor = class_weights_tensor

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.dropout(out[:, -1, :])  # Take the last time step
        out = self.fc(out)

        return out

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_start_training = time.time()
        for i, batch_x in enumerate(train_loader):
            _, inputs_x, labels_x, domains_x = batch_x
            inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

            optimizer.zero_grad()
            outputs = self(inputs_x)

            loss = criterion(outputs, labels_x.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels_x).float().sum().item()
            total += labels_x.numel()
        train_accuracy = correct / total
        train_loss /= len(train_loader)
        epoch_end_training = time.time()

        return train_loss, train_accuracy, epoch_end_training - epoch_start_training

    def evaluate(
        self,
        data_loader,
        device,
        criterion=None,
    ):
        self.eval()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        loss = 0
        correct = 0
        total = 0
        epoch_start_validation = time.time()
        with torch.no_grad():
            for i, batch_x in enumerate(data_loader):
                _, inputs_x, labels_x, _ = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                outputs = self(inputs_x)

                if criterion is not None:
                    loss += criterion(outputs, labels_x.float()).item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                # Compute confusion matrix components
                TP += ((preds == 1) & (labels_x == 1)).float().sum().item()
                TN += ((preds == 0) & (labels_x == 0)).float().sum().item()
                FP += ((preds == 1) & (labels_x == 0)).float().sum().item()
                FN += ((preds == 0) & (labels_x == 1)).float().sum().item()

                correct += (preds == labels_x).float().sum().item()
                total += labels_x.numel()
        epoch_end_validation = time.time()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = correct / total
        loss /= len(data_loader)

        return (
            loss,
            accuracy,
            sensitivity,
            specificity,
            epoch_end_validation - epoch_start_validation,
        )

    def train_model(
        self,
        train_loader,
        val_loader,
        device,
        output_dir,
        num_epochs=100,
        early_stopping_patience=10,
    ):
        criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        early_stopping = EarlyStopping(patience=early_stopping_patience, delta=0.001)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(num_epochs):
            train_loss, train_accuracy, training_time = self.train_epoch(
                train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy, _, _, validation_time = self.evaluate(
                val_loader, device, criterion
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            log_message(
                {
                    "Epoch": epoch + 1,
                    "Training Loss": train_loss,
                    "Training Accuracy": train_accuracy,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                    "Training Time": training_time,
                    "Validation Time": validation_time,
                }
            )

            if early_stopping.early_stop(val_loss):
                print(
                    f"Validation loss did not decrease for {early_stopping.patience} epochs. Training stopped."
                )
                early_stopping.save_checkpoint(
                    val_loss, self, filename=f"{output_dir}/model_checkpoint.pth"
                )
                break
        return train_losses, val_losses, train_accuracies, val_accuracies
