import torch
import torch.nn as nn
from early_stopping import EarlyStopping


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        class_weights_tensor,
        input_size=1024,
        hidden_size=150,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.25,
    ):
        super(BiLSTMModel, self).__init__()
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

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.early_stopping = EarlyStopping(patience=10, delta=0)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.dropout(out[:, -1, :])  # Take the last time step
        out = self.fc(out)

        return out

    def train_model(
        self,
        train_loader_x,
        validation_loader,
        device,
        output_dir,
        epochs=10,
    ):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            # Training phase
            self.train()
            for i, batch_x in enumerate(train_loader_x):
                inputs_x, labels_x, _ = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                self.optimizer.zero_grad()
                # Transpose between dim 1 and 2 because LSTM expects the sequence length to be the second dimension
                inputs_x = inputs_x.transpose(1, 2)
                outputs_x = self(inputs_x)
                loss_x = self.criterion(outputs_x, labels_x.float())
                loss_x.backward()
                self.optimizer.step()

                total_loss += loss_x.item()
                probs = torch.sigmoid(outputs_x)
                preds = (probs > 0.5).float()
                correct += (preds == labels_x).float().sum().item()
                total += labels_x.numel()

            train_accuracy = correct / total
            total_loss /= len(train_loader_x)
            print(
                f"End of Epoch {epoch+1}, Training Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
            )

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():  # No gradient calculation for validation
                for i, batch in enumerate(validation_loader):
                    inputs_v, labels_v, _ = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)
                    inputs_v = inputs_v.transpose(1, 2)
                    outputs_v = self(inputs_v)
                    loss_v = self.criterion(outputs_v, labels_v.float())

                    val_loss += loss_v.item()
                    probs_v = torch.sigmoid(outputs_v)
                    preds_v = (probs_v > 0.5).float()
                    val_correct += (preds_v == labels_v).float().sum().item()
                    val_total += labels_v.numel()

            val_accuracy = val_correct / val_total
            val_loss /= len(validation_loader)
            print(
                f"End of Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

            # Save the loss and accuracy for plotting
            train_losses.append(total_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Early stopping
            if self.early_stopping.early_stop(val_loss):
                print(
                    f"Validation loss did not decrease for {self.early_stopping.patience} epochs. Training stopped."
                )

                # Save the model checkpoint
                self.early_stopping.save_checkpoint(
                    val_loss, self, filename=f"{output_dir}/model_checkpoint.pth"
                )
                break
        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate_model(self, test_loader, device):
        self.eval()  # Set the model to evaluation mode
        test_loss = 0
        TP = 0  # True Positives
        TN = 0  # True Negatives
        FP = 0  # False Positives
        FN = 0  # False Negatives

        with torch.no_grad():  # No gradient calculation for evaluation
            for batch in test_loader:
                inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs.transpose(1, 2))
                loss = self.criterion(outputs, labels.float())

                test_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                # Compute confusion matrix components
                TP += ((preds == 1) & (labels == 1)).float().sum().item()
                TN += ((preds == 0) & (labels == 0)).float().sum().item()
                FP += ((preds == 1) & (labels == 0)).float().sum().item()
                FN += ((preds == 0) & (labels == 1)).float().sum().item()

        # Calculate sensitivity and specificity
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        test_loss /= len(test_loader)

        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(
            f"Sensitivity (Recall): {sensitivity:.2f}, Specificity: {specificity:.2f}"
        )

        return test_loss, accuracy, sensitivity, specificity
