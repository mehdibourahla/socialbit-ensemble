import torch
import torch.nn as nn
import logging
from early_stopping import EarlyStopping


class StandardModel(nn.Module):
    def __init__(
        self,
        class_weights_tensor=None,
        num_classes=2,
        num_experts=2,
        transpose_input=False,
    ):
        super(StandardModel, self).__init__()
        self.class_weights_tensor = class_weights_tensor
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.transpose_input = transpose_input

    def train_model(
        self,
        train_loader_x,
        validation_loader,
        device,
        output_dir,
        epochs=10,
    ):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        early_stopping = EarlyStopping(patience=10, delta=0)

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
                _, inputs_x, labels_x, domains_x = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                optimizer.zero_grad()
                if self.transpose_input:
                    inputs_x = inputs_x.transpose(1, 2)

                outputs_x = (
                    self(inputs_x, domains_x)
                    if self.num_experts > 1
                    else self(inputs_x)
                )
                loss_x = criterion(outputs_x, labels_x.float())
                loss_x.backward()
                optimizer.step()

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
            logging.info(
                f"End of Epoch {epoch+1}, Training Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
            )

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():  # No gradient calculation for validation
                for i, batch in enumerate(validation_loader):
                    _, inputs_v, labels_v, domains_v = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

                    if self.transpose_input:
                        inputs_v = inputs_v.transpose(1, 2)

                    outputs_v = (
                        self(inputs_v, domains_v)
                        if self.num_experts > 1
                        else self(inputs_v)
                    )
                    loss_v = criterion(outputs_v, labels_v.float())

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
            logging.info(
                f"End of Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )
            # Save the loss and accuracy for plotting
            train_losses.append(total_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Early stopping
            if early_stopping.early_stop(val_loss):
                print(
                    f"Validation loss did not decrease for {early_stopping.patience} epochs. Training stopped."
                )
                logging.info(
                    f"Validation loss did not decrease for {early_stopping.patience} epochs. Training stopped."
                )
                # Save the model checkpoint
                early_stopping.save_checkpoint(
                    val_loss, self, filename=f"{output_dir}/model_checkpoint.pth"
                )
                break
        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate_model(self, test_loader, device):
        self.eval()  # Set the model to evaluation mode
        criterion = nn.BCEWithLogitsLoss()
        test_loss = 0
        TP = 0  # True Positives
        TN = 0  # True Negatives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        predictions = []
        with torch.no_grad():  # No gradient calculation for evaluation
            for batch in test_loader:
                filename, inputs, labels, domains = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if self.transpose_input:
                    inputs = inputs.transpose(1, 2)

                outputs = (
                    self(inputs, domains) if self.num_experts > 1 else self(inputs)
                )
                loss = criterion(outputs, labels.float())

                test_loss += loss.item()
                probs = torch.sigmoid(outputs)
                for i in range(len(filename)):
                    prediction = probs[i].cpu().numpy()
                    predictions.append(
                        {
                            "filename": filename[i],
                            "true_label": labels[i].cpu().numpy()[1],
                            "positive": prediction[1],
                            "negative": prediction[0],
                        }
                    )

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

        return test_loss, accuracy, sensitivity, specificity, predictions
