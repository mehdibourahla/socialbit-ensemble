import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import torch.nn.functional as F
import random
from utils import representative_cluster, log_message
import numpy as np
import time
import pandas as pd
import os


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            # Ensure alpha is a float tensor
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Ensure inputs are softmax probabilities (F.log_softmax to get log-probs for numerical stability)
        log_probs = F.log_softmax(inputs, dim=1)
        # Convert targets to the same format as log_probs
        targets = targets.float()

        # Calculate the Cross-Entropy component of the Focal Loss
        ce_loss = -1 * torch.sum(targets * log_probs, dim=1)

        # Calculate pt as the probability of the target class
        pt = torch.exp(-ce_loss)

        # Compute alpha factor
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            # Apply alpha factor to each class (assumes alpha is provided as [alpha_negative, alpha_positive])
            at = torch.sum(self.alpha * targets, dim=1)
        else:
            at = 1.0

        # Calculate the final focal loss
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


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

    def get_positive_example(self, representations, domains, labels, i):
        anchor_domain = domains[i].item()
        anchor_label = labels[i].item()

        positive_indices = [
            idx
            for idx, (domain, label) in enumerate(zip(domains, labels))
            if (domain.item() == anchor_domain or label.item() == anchor_label)
            and idx != i
        ]

        if not positive_indices:
            return representations[i]

        positive_idx = random.choice(positive_indices)
        return representations[positive_idx]

    def get_negative_example(self, representations, domains, labels, i):
        """
        Select a negative example for the anchor, considering both domain and label.
        An example is negative if it has a different domain and a different label from the anchor.
        """
        anchor_domain = domains[i].item()
        anchor_label = labels[i].item()
        negative_indices = [
            idx
            for idx, (domain, label) in enumerate(zip(domains, labels))
            if domain.item() != anchor_domain and label.item() != anchor_label
        ]

        if not negative_indices:
            return None  # Indicate no suitable negative example was found

        negative_idx = random.choice(negative_indices)
        return representations[negative_idx]

    def contrastive_loss(self, representations, domains, labels, margin=0.5):
        loss = 0.0
        valid_triplets = 0
        for i in range(representations.size(0)):
            anchor = representations[i].unsqueeze(0)
            positive = self.get_positive_example(
                representations, domains, labels, i
            ).unsqueeze(0)
            pos_sim = F.cosine_similarity(anchor, positive)
            pos_dist = 1 - pos_sim

            negative = self.get_negative_example(representations, domains, labels, i)
            if negative is not None:
                negative = negative.unsqueeze(0)
                neg_sim = F.cosine_similarity(anchor, negative)
                neg_dist = 1 - neg_sim

                # Triplet loss calculation with cosine distance
                triplet_loss = F.relu(pos_dist - neg_dist + margin)
            else:
                # Handle cases where no suitable negative example is found
                triplet_loss = F.relu(pos_dist + margin)
                # Optionally log or handle this situation as needed
                continue

            valid_triplets += 1
            loss += triplet_loss.mean()

        if valid_triplets > 0:
            loss /= valid_triplets
        else:
            # Create a grad-able zero tensor if no valid triplets are found
            loss = torch.tensor(0.0, requires_grad=True)

        return loss

    def train_model(
        self,
        train_loader_x,
        validation_loader,
        device,
        output_dir,
        epochs=10,
        use_metadata=False,
        alpha=0.5,
    ):
        bce_loss_fn = FocalLoss(alpha=self.class_weights_tensor, gamma=2)
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        early_stopping = EarlyStopping(patience=10, delta=0)
        signature_matrix = torch.rand(self.num_experts * 2, 64 * 3, device=device)

        beta = 1 - alpha

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        meta_over_epochs = []

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            pos_representations_for_clustering = [[] for _ in range(self.num_experts)]
            neg_representations_for_clustering = [[] for _ in range(self.num_experts)]

            # Training phase
            self.train()
            epoch_start_training = time.time()
            for _, batch_x in enumerate(train_loader_x):
                _, inputs_x, labels_x, domains_x = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                optimizer.zero_grad()

                final_output_x, representations_x = self.forward_train(
                    inputs_x,
                    domains_x,
                )

                pos_labels = labels_x[:, 1] > 0.5
                neg_labels = ~pos_labels
                # Update signature sums and counts
                for idx in range(self.num_experts):
                    mask_pos = (domains_x == idx) & pos_labels
                    mask_neg = (domains_x == idx) & neg_labels
                    if mask_pos.any():
                        sampled_pos_representations = representations_x[
                            mask_pos, idx, :
                        ].detach()
                        pos_representations_for_clustering[idx].extend(
                            sampled_pos_representations
                        )

                    if mask_neg.any():
                        sampled_neg_representations = representations_x[
                            mask_neg, idx, :
                        ].detach()

                        neg_representations_for_clustering[idx].extend(
                            sampled_neg_representations
                        )

                loss_cl_x = self.contrastive_loss(
                    representations_x, domains_x, labels_x[:, 1]
                )  # Contrastive loss
                loss_bce_x = bce_loss_fn(
                    final_output_x, labels_x.float()
                )  # Social Interaction loss

                loss_x = (
                    alpha * loss_bce_x + beta * loss_cl_x
                    if self.num_experts > 1
                    else loss_bce_x
                )
                loss_x.backward()
                optimizer.step()

                total_loss += loss_x.item()
                probs = torch.sigmoid(final_output_x)
                preds = (probs > 0.5).float()
                correct += (preds == labels_x).float().sum().item()
                total += labels_x.numel()
            train_accuracy = correct / total
            total_loss /= len(train_loader_x)

            for idx in range(self.num_experts):
                min_pos_len = min(
                    [len(reps) for reps in pos_representations_for_clustering]
                )
                min_neg_len = min(
                    [len(reps) for reps in neg_representations_for_clustering]
                )

                # Cut the list of representations to the minimum length
                pos_representations_for_clustering[idx] = (
                    pos_representations_for_clustering[idx][:min_pos_len]
                )
                neg_representations_for_clustering[idx] = (
                    neg_representations_for_clustering[idx][:min_neg_len]
                )

                pos_representations_for_clustering[idx] = torch.stack(
                    pos_representations_for_clustering[idx]
                )
                neg_representations_for_clustering[idx] = torch.stack(
                    neg_representations_for_clustering[idx]
                )

            pos_medoids = representative_cluster(pos_representations_for_clustering)
            neg_medoids = representative_cluster(neg_representations_for_clustering)

            # Update the signature matrix with separate embeddings for positive and negative samples
            for idx in range(self.num_experts):
                signature_matrix[idx * 2] = pos_medoids[idx].to(device)  # Positive
                signature_matrix[idx * 2 + 1] = neg_medoids[idx].to(device)  # Negative

            if use_metadata:
                meta_over_epochs.append(signature_matrix.clone())
            epoch_end_training = time.time()

            # Validation phase
            self.eval()  # Set the model to evaluation mode
            epoch_start_validation = time.time()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():  # No gradient calculation for validation
                for i, batch in enumerate(validation_loader):
                    _, inputs_v, labels_v, domains_v = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

                    (final_output_v, representations_v, _, _, _, _) = (
                        self.forward_inference(
                            inputs_v,
                            signature_matrix=signature_matrix,
                        )
                    )

                    preds_v = (final_output_v > 0.5).float()
                    loss_bce_v = bce_loss_fn(preds_v, labels_v.float())
                    if self.num_experts > 1:
                        loss_cl_v = self.contrastive_loss(
                            representations_v, domains_v, labels_v[:, 1]
                        )

                    loss_v = (
                        alpha * loss_bce_v + beta * loss_cl_v
                        if self.num_experts > 1
                        else loss_bce_v
                    )

                    val_loss += loss_v.item()
                    val_correct += (preds_v == labels_v).float().sum().item()
                    val_total += labels_v.numel()

            val_accuracy = val_correct / val_total
            val_loss /= len(validation_loader)
            epoch_end_validation = time.time()

            training_time = epoch_end_training - epoch_start_training
            validation_time = epoch_end_validation - epoch_start_validation

            print(
                {
                    "Epoch": epoch + 1,
                    "Training Loss": total_loss,
                    "Training Accuracy": train_accuracy,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                    "Training start": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(epoch_start_training)
                    ),
                    "Training end": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(epoch_end_training)
                    ),
                    "Validation start": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(epoch_start_validation)
                    ),
                    "Validation end": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(epoch_end_validation)
                    ),
                }
            )
            log_message(
                {
                    "Epoch": epoch + 1,
                    "Training Loss": total_loss,
                    "Training Accuracy": train_accuracy,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                    "Training Time": training_time,
                    "Validation Time": validation_time,
                }
            )
            # Save the loss and accuracy for plotting
            train_losses.append(total_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            # Early stopping
            if early_stopping.early_stop(val_loss):

                log_message(
                    {
                        "Early Stopping": f"Validation loss did not decrease for {early_stopping.patience} epochs. Training stopped."
                    }
                )
                # Save the model checkpoint
                early_stopping.save_checkpoint(
                    val_loss, self, filename=f"{output_dir}/model_checkpoint.pth"
                )
                break
        return (
            signature_matrix,
            meta_over_epochs,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
        )

    def evaluate_model(self, dataset_name, test_loader, signature_matrix, device):
        self.eval()  # Set the model to evaluation mode
        TP = 0  # True Positives
        TN = 0  # True Negatives
        FP = 0  # False Positives
        FN = 0  # False Negatives

        with torch.no_grad():  # No gradient calculation for evaluation
            for batch in test_loader:
                filename, inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)

                (outputs, representations, expert_idx, _, _, _) = (
                    self.forward_inference(inputs, signature_matrix=signature_matrix)
                )

                preds = (outputs > 0.5).float()

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
            {
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Accuracy": accuracy,
            }
        )
        log_message(
            {
                f"{dataset_name}_Sensitivity": sensitivity,
                f"{dataset_name}_Specificity": specificity,
                f"{dataset_name}_Accuracy": accuracy,
            }
        )

        return accuracy, sensitivity, specificity
