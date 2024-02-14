import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from early_stopping import EarlyStopping
import numpy as np


class SharedFeatureExtractor(nn.Module):
    def __init__(self):
        super(SharedFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1024, out_channels=512, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 9, 1024)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = self.pool(x1)
        x2 = F.relu(self.conv2(x))
        x = self.pool(x2)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x, x1, x2


class ExpertModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ExpertModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MasterModel(nn.Module):
    def __init__(
        self, num_experts, class_weights_tensor, num_classes=2, skip_connection=True
    ):
        super(MasterModel, self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.skip_connection = skip_connection
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList(
            [ExpertModel(num_classes) for _ in range(num_experts)]
        )
        self.aggregation_layer = nn.Linear(num_experts * num_classes, num_classes)
        self.final_classification_layer = nn.Linear(
            19458, num_classes
        )  # This should be calculated based on the actual sizes
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.early_stopping = EarlyStopping(patience=10, delta=0)

    def forward(self, x, expert_idx):
        shared_features, intermediate1, intermediate2 = self.shared_extractor(x)

        # Get the aggregated output from the experts
        batch_size = x.size(0)
        expert_outputs = torch.zeros(
            batch_size, self.num_experts * self.num_classes, device=x.device
        )
        for i, idx in enumerate(expert_idx):
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            expert_output = self.experts[idx](shared_features[i].unsqueeze(0))
            expert_outputs[i, idx * self.num_classes : (idx + 1) * self.num_classes] = (
                expert_output.squeeze(0)
            )

        aggregated_output = self.aggregation_layer(expert_outputs)

        if self.skip_connection:
            intermediate1_flat = intermediate1.view(x.size(0), -1)
            intermediate2_flat = intermediate2.view(x.size(0), -1)
            combined_features = torch.cat(
                [aggregated_output, intermediate1_flat, intermediate2_flat], dim=1
            )
            final_output = self.final_classification_layer(combined_features)
        else:
            final_output = aggregated_output

        return final_output

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
                inputs_x, labels_x, domains_x = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                self.optimizer.zero_grad()
                outputs_x = self(inputs_x, domains_x)
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
                    inputs_v, labels_v, domains_v = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

                    outputs_v = self(inputs_v, domains_v)
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
            logging.info(
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
                logging.info(
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
                inputs, labels, domains = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs, domains)
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
