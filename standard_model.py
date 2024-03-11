import torch
import torch.nn as nn
from early_stopping import EarlyStopping
import torch.nn.functional as F
import random
from utils import representative_cluster, log_message
import numpy as np
import time
import pandas as pd


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

    def preprocess_representations(self, representations):
        representations = [
            torch.cat(rep, dim=0).cpu().numpy() for rep in representations
        ]  # Shape: (num_experts, N, 64 * 3)

        return representations

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
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights_tensor)
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
                if self.transpose_input:
                    inputs_x = inputs_x.transpose(1, 2)
                # Check if signature matrix is diffenrent than zeros

                final_output_x, representations_x, expert_idx_x, _, _ = (
                    self(
                        inputs_x,
                        domains_x,
                        inference_mode=False,
                        signature_matrix=signature_matrix,
                    )
                    if self.num_experts > 1
                    else self(inputs_x)
                )
                if self.num_experts > 1:
                    pos_labels = labels_x[:, 1] > 0.5
                    neg_labels = ~pos_labels
                    # Update signature sums and counts
                    for idx in range(self.num_experts):
                        mask_pos = (expert_idx_x == idx) & pos_labels
                        mask_neg = (expert_idx_x == idx) & neg_labels
                        if mask_pos.any():
                            sampled_pos_representations = representations_x[
                                mask_pos
                            ].detach()
                            pos_representations_for_clustering[idx].append(
                                sampled_pos_representations
                            )

                        if mask_neg.any():
                            sampled_neg_representations = representations_x[
                                mask_neg
                            ].detach()
                            neg_representations_for_clustering[idx].append(
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

            if self.num_experts > 1:
                # TODO: Try to not update the signature matrix at every epoch
                pos_representations_for_clustering = self.preprocess_representations(
                    pos_representations_for_clustering
                )
                neg_representations_for_clustering = self.preprocess_representations(
                    neg_representations_for_clustering
                )

                # Compute medoids separately for positive and negative representations
                pos_medoids = [
                    representative_cluster([rep], check=False)[0]
                    for rep in pos_representations_for_clustering
                ]
                neg_medoids = [
                    representative_cluster([rep], check=False)[0]
                    for rep in neg_representations_for_clustering
                ]

                # Update the signature matrix with separate embeddings for positive and negative samples
                for idx in range(self.num_experts):
                    signature_matrix[idx * 2] = torch.from_numpy(pos_medoids[idx]).to(
                        device
                    )  # Positive
                    signature_matrix[idx * 2 + 1] = torch.from_numpy(
                        neg_medoids[idx]
                    ).to(
                        device
                    )  # Negative

            if use_metadata:
                meta_over_epochs.append(signature_matrix.clone())
            epoch_end_training = time.time()
            # Validation phase
            self.eval()  # Set the model to evaluation mode
            epoch_start_validation = time.time()
            val_loss = 0
            val_correct = 0
            val_total = 0
            expert_selection_counts = [0] * self.num_experts
            expert_correct_counts = [0] * self.num_experts
            with torch.no_grad():  # No gradient calculation for validation
                for i, batch in enumerate(validation_loader):
                    _, inputs_v, labels_v, domains_v = batch
                    inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)

                    if self.transpose_input:
                        inputs_v = inputs_v.transpose(1, 2)

                    (
                        final_output_v,
                        representations_v,
                        expert_idx_v,
                        expert_output_v,
                        _,
                    ) = (
                        self(
                            inputs_v,
                            inference_mode=True,
                            signature_matrix=signature_matrix,
                        )
                        if self.num_experts > 1
                        else self(inputs_v)
                    )
                    # Update expert selection counts
                    for expert_idx in expert_idx_v.tolist():
                        expert_selection_counts[expert_idx] += 1

                        # Compute accuracy for each expert
                        expert_output = expert_output_v[:, expert_idx, :]
                        expert_probs = torch.sigmoid(expert_output)
                        expert_preds = (expert_probs > 0.5).float()
                        expert_correct = (expert_preds == labels_v).float().sum().item()
                        expert_correct_counts[expert_idx] += expert_correct

                    loss_bce_v = bce_loss_fn(final_output_v, labels_v.float())
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
                    probs_v = torch.sigmoid(final_output_v)
                    preds_v = (probs_v > 0.5).float()
                    val_correct += (preds_v == labels_v).float().sum().item()
                    val_total += labels_v.numel()

            val_accuracy = val_correct / val_total
            val_loss /= len(validation_loader)
            epoch_end_validation = time.time()
            total_samples = sum(expert_selection_counts)
            for i, count in enumerate(expert_selection_counts):
                count = expert_selection_counts[i]
                correct = expert_correct_counts[i]
                accuracy = correct / count if count > 0 else 0

                log_message(
                    {
                        f"Expert {i+1}": {
                            "Selections": count,
                            "Selection Ratio": count / total_samples,
                            "Accuracy": accuracy,
                        }
                    }
                )
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

    def evaluate_model(self, test_loader, signature_matrix, device):
        self.eval()  # Set the model to evaluation mode
        TP = 0  # True Positives
        TN = 0  # True Negatives
        FP = 0  # False Positives
        FN = 0  # False Negatives
        predictions = []
        df = []
        expert_selection_counts = [0] * self.num_experts
        expert_correct_counts = [0] * self.num_experts
        with torch.no_grad():  # No gradient calculation for evaluation
            for batch in test_loader:
                filename, inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)

                if self.transpose_input:
                    inputs = inputs.transpose(1, 2)

                (outputs, _, expert_idx, expert_outputs, expert_weights) = (
                    self(inputs, inference_mode=True, signature_matrix=signature_matrix)
                    if self.num_experts > 1
                    else self(inputs)
                )

                for idx in expert_idx.tolist():
                    expert_selection_counts[idx] += 1

                    expert_output = expert_outputs[:, idx, :]
                    expert_probs = torch.sigmoid(expert_output)
                    expert_preds = (expert_probs > 0.5).float()
                    expert_correct = (expert_preds == labels).float().sum().item()
                    # For each expert, check the misclassification.
                    expert_correct_counts[idx] += expert_correct

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
                # Get the idx of the misclassified samples
                misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
                misclassified_idx = torch.unique(misclassified_idx)
                for idx in misclassified_idx:
                    obj = {"filename": filename[idx]}
                    for i in range(self.num_experts):
                        expert_preds = torch.sigmoid(expert_outputs[idx, i, :])
                        expert_weight = expert_weights[idx, i]
                        obj[f"expert_{i+1}"] = (
                            expert_preds.cpu().numpy()[1],
                            expert_weight.cpu().numpy(),
                        )
                    obj["true_label"] = labels[idx, 1].cpu().numpy()
                    obj["prediction"] = probs[idx, 1].cpu().numpy()
                    df.append(obj)

                # Compute confusion matrix components
                TP += ((preds == 1) & (labels == 1)).float().sum().item()
                TN += ((preds == 0) & (labels == 0)).float().sum().item()
                FP += ((preds == 1) & (labels == 0)).float().sum().item()
                FN += ((preds == 0) & (labels == 1)).float().sum().item()
        df = pd.DataFrame(df)
        # Calculate sensitivity and specificity
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        total_samples = sum(expert_selection_counts)
        for i in range(self.num_experts):
            count = expert_selection_counts[i]
            correct = expert_correct_counts[i]
            accuracy_expert = correct / count if count > 0 else 0

            print(
                f"Expert {i+1}: Selections: {count}, Selection Ratio: {count / total_samples}, Accuracy: {accuracy_expert}"
            )

            log_message(
                {
                    f"Expert {i+1}": {
                        "Selections": count,
                        "Selection Ratio": count / total_samples,
                        "Accuracy": accuracy_expert,
                    }
                }
            )
        print(
            {
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Accuracy": accuracy,
            }
        )
        log_message(
            {
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Accuracy": accuracy,
            }
        )

        return accuracy, sensitivity, specificity, df
