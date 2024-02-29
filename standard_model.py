import torch
import torch.nn as nn
import logging
from early_stopping import EarlyStopping
import torch.nn.functional as F
import random


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

    def get_positive_example(self, representations, domains, i):
        """
        Select a positive example (same domain but different instance) for the anchor.
        """
        anchor_domain = domains[i]
        same_domain_indices = [
            idx
            for idx, domain in enumerate(domains)
            if domain == anchor_domain and idx != i
        ]

        if not same_domain_indices:
            return representations[i]

        positive_idx = random.choice(same_domain_indices)
        return representations[positive_idx]

    def get_negative_example(self, representations, domains, i):
        """
        Select a negative example (different domain) for the anchor.
        """
        anchor_domain = domains[i]
        # Filter indices of a different domain
        different_domain_indices = [
            idx for idx, domain in enumerate(domains) if domain != anchor_domain
        ]

        # Edge case: If no samples from a different domain are present, this is a critical error,
        # as it suggests the batch does not contain enough domain variety for effective learning.
        if not different_domain_indices:
            return None

        # Randomly select one of the different domain indices
        negative_idx = random.choice(different_domain_indices)
        return representations[negative_idx]

    def contrastive_loss(self, representations, domains, margin=1):
        loss = 0.0
        valid_triplets = 0
        for i in range(representations.size(0)):
            anchor = representations[i]
            positive = self.get_positive_example(representations, domains, i)
            negative = self.get_negative_example(representations, domains, i)

            pos_dist = F.pairwise_distance(anchor, positive, keepdim=True)
            if negative is not None:
                neg_dist = F.pairwise_distance(anchor, negative, keepdim=True)
                triplet_loss = F.relu(pos_dist - neg_dist + margin)
            else:
                triplet_loss = F.relu(pos_dist + margin)
                print(
                    "Warning: Negative example not found. This may be due to insufficient domain variety in the batch."
                )
                continue
            valid_triplets += 1
            loss += triplet_loss.mean()

        if valid_triplets > 0:
            loss /= valid_triplets
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        return loss

    def train_model(
        self,
        train_loader_x,
        validation_loader,
        device,
        output_dir,
        epochs=10,
    ):
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights_tensor)
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        early_stopping = EarlyStopping(patience=10, delta=0)
        signature_matrix = torch.zeros(self.num_experts, 64, 3, device=device)

        # TODO: Transform these into hyperparameters
        alpha = 1
        beta = 1

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            signature_sums = torch.zeros_like(signature_matrix)
            counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
            # Training phase
            self.train()
            for _, batch_x in enumerate(train_loader_x):
                _, inputs_x, labels_x, domains_x = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                optimizer.zero_grad()
                if self.transpose_input:
                    inputs_x = inputs_x.transpose(1, 2)

                final_output_x, representations_x, expert_idx_x = (
                    self(inputs_x, domains_x)
                    if self.num_experts > 1
                    else self(inputs_x)
                )

                # Update signature sums and counts
                for idx in range(self.num_experts):
                    mask = expert_idx_x == idx
                    if mask.any():
                        signature_sums[idx] += representations_x[mask].mean(dim=0)
                        counts[idx] += mask.sum()

                loss_bce_x = bce_loss_fn(
                    final_output_x, labels_x.float()
                )  # Social Interaction loss
                loss_cl_x = self.contrastive_loss(
                    representations_x, expert_idx_x
                )  # Contrastive loss

                loss_x = alpha * loss_bce_x + beta * loss_cl_x
                loss_x.backward()
                optimizer.step()

                total_loss += loss_x.item()
                probs = torch.sigmoid(final_output_x)
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
            # After the epoch, update the signatures
            for idx in range(self.num_experts):
                if counts[idx] > 0:
                    signature_matrix[idx] = signature_sums[idx] / counts[idx]

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():  # No gradient calculation for validation
                for i, batch in enumerate(validation_loader):
                    _, inputs_v, labels_v, _ = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

                    if self.transpose_input:
                        inputs_v = inputs_v.transpose(1, 2)

                    final_output_v, representations_v, expert_idx_v = (
                        self(inputs_v, signature_matrix=signature_matrix)
                        if self.num_experts > 1
                        else self(inputs_v)
                    )

                    loss_bce_v = bce_loss_fn(final_output_v, labels_v.float())
                    loss_cl_v = self.contrastive_loss(representations_v, expert_idx_v)

                    loss_v = alpha * loss_bce_v + beta * loss_cl_v

                    val_loss += loss_v.item()
                    probs_v = torch.sigmoid(final_output_v)
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
        return (
            signature_matrix,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
        )

    def evaluate_model(self, test_loader, signature_matrix, device):
        self.eval()  # Set the model to evaluation mode
        TP = 0  # True Positives
        TN = 0  # True Negatives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        predictions = []
        with torch.no_grad():  # No gradient calculation for evaluation
            for batch in test_loader:
                filename, inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if self.transpose_input:
                    inputs = inputs.transpose(1, 2)

                (
                    outputs,
                    _,
                    _,
                ) = (
                    self(inputs, signature_matrix=signature_matrix)
                    if self.num_experts > 1
                    else self(inputs)
                )

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

        print(
            f"Sensitivity (Recall): {sensitivity:.2f}, Specificity: {specificity:.2f}"
        )

        return accuracy, sensitivity, specificity, predictions
