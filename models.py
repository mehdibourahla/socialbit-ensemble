import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import representative_cluster, log_message, EarlyStopping
import time


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=dilation_size,
                    dilation=dilation_size,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        return torch.matmul(attention, value)


class SharedFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_size=1024,
        seq_length=30,
        num_channels=[512, 256],
        kernel_size=3,
        dropout=0.25,
    ):
        super(SharedFeatureExtractor, self).__init__()
        self.temporal_conv_net = TemporalConvNet(
            input_size, num_channels, kernel_size, dropout
        )
        self.attention = ScaledDotProductAttention(scale=seq_length**0.5)

    def forward(self, x):
        # Input x shape: (batch, features, seq_length)
        conv_output = self.temporal_conv_net(x)  # Apply Temporal Convolution
        # Attention expects (batch, seq_length, features), so transpose conv_output
        conv_output = conv_output.transpose(
            1, 2
        )  # Now shape: (batch, seq_length, features)
        attention_output = self.attention(
            conv_output, conv_output, conv_output
        )  # Self-attention
        attention_output = attention_output.transpose(1, 2)
        return attention_output


class ExpertModel(nn.Module):
    def __init__(
        self,
        input_size=1024,
        seq_length=30,
        num_channels=[512, 256, 128, 64],
        kernel_size=3,
        dropout=0.25,
    ):
        super(ExpertModel, self).__init__()
        self.temporal_conv_net = TemporalConvNet(
            input_size, num_channels, kernel_size, dropout
        )
        self.attention = ScaledDotProductAttention(scale=seq_length**0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        # Input x shape: (batch, features, seq_length)
        conv_output = self.temporal_conv_net(x)  # Apply Temporal Convolution
        # Attention expects (batch, seq_length, features), so transpose conv_output
        conv_output = conv_output.transpose(
            1, 2
        )  # Now shape: (batch, seq_length, features)
        attention_output = self.attention(
            conv_output, conv_output, conv_output
        )  # Self-attention
        attention_output = attention_output.transpose(1, 2)
        embedding = self.avg_pool(attention_output)
        output = self.fc(embedding.squeeze(-1))
        return output, embedding


class MasterModel(nn.Module):
    def __init__(
        self,
        num_experts=2,
        num_classes=2,
        class_weights_tensor=None,
        signature_matrix=None,
        representation_size=64,
        alpha=0.5,
    ):
        super(MasterModel, self).__init__()
        self.class_weights_tensor = class_weights_tensor
        self.signature_matrix = signature_matrix
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.representation_size = representation_size
        self.alpha = alpha
        self.beta = 1 - alpha
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList([ExpertModel() for _ in range(num_experts)])

    def process_experts(self, shared_features):
        expert_outputs = torch.zeros(
            shared_features.size(0),
            self.num_experts,
            self.num_classes,
            device=shared_features.device,
        )
        expert_representations = torch.zeros(
            shared_features.size(0),
            self.num_experts,
            self.representation_size,
            device=shared_features.device,
        )  # Adjust the size accordingly

        for i, expert in enumerate(self.experts):
            output, representation = expert(shared_features)
            expert_outputs[:, i, :] = output
            expert_representations[:, i, :] = representation.reshape(
                shared_features.size(0), -1
            )  # Flatten the representation

        return expert_outputs, expert_representations

    def compute_similarity_weights(self, expert_representations, signature_matrix):
        batch_size, num_experts, _ = expert_representations.size()

        similarities_pos = torch.empty(
            batch_size, num_experts, device=expert_representations.device
        )
        similarities_neg = torch.empty(
            batch_size, num_experts, device=expert_representations.device
        )

        for i in range(num_experts):

            pos_index = 2 * i
            neg_index = 2 * i + 1

            rep = expert_representations[:, i, :]
            signature_pos = signature_matrix[pos_index, :].unsqueeze(0)
            signature_neg = signature_matrix[neg_index, :].unsqueeze(0)

            similarities_pos[:, i] = F.cosine_similarity(
                rep, signature_pos.expand_as(rep), dim=1
            )
            similarities_neg[:, i] = F.cosine_similarity(
                rep, signature_neg.expand_as(rep), dim=1
            )

        summed_similarities_expert = [
            similarities_pos[:, i] + similarities_neg[:, i] for i in range(num_experts)
        ]

        similarity_weights = F.softmax(
            torch.stack(summed_similarities_expert, dim=1), dim=1
        )

        return similarity_weights, similarities_pos, similarities_neg

    def forward_train(self, x, expert_idx):
        # shared_features = self.shared_extractor(x)
        expert_outputs, expert_representations = self.process_experts(x)
        output = torch.zeros(x.size(0), self.num_classes, device=x.device)
        representation = torch.zeros(
            x.size(0), self.representation_size, device=x.device
        )
        newbie_output = torch.zeros(x.size(0), self.num_classes, device=x.device)

        # Get the expert output
        for i in range(x.size(0)):
            output[i, :] = expert_outputs[i, expert_idx[i], :]
            representation[i, :] = expert_representations[i, expert_idx[i], :]

        # Get the newbie output
        for i in range(self.num_experts):
            mask = expert_idx != i
            newbie_output[mask, :] += expert_outputs[mask, i, :]
        newbie_output = newbie_output / (self.num_experts - 1)

        return output, newbie_output, representation, expert_representations

    def forward_inference(self, x):
        # shared_features = self.shared_extractor(x)
        expert_outputs, expert_representations = self.process_experts(x)
        output = torch.zeros(x.size(0), self.num_classes, device=x.device)
        newbie_output = torch.zeros(x.size(0), self.num_classes, device=x.device)

        similarity_weights, similarities_pos, similarities_neg = (
            self.compute_similarity_weights(
                expert_representations, self.signature_matrix
            )
        )

        # Get the most similar expert output
        expert_idx = torch.argmax(similarity_weights, dim=1)
        for i in range(x.size(0)):
            output[i, :] = expert_outputs[i, expert_idx[i], :]

        # Get the newbie output
        for i in range(self.num_experts):
            mask = expert_idx != i
            newbie_output[mask, :] += expert_outputs[mask, i, :]
        newbie_output = newbie_output / (self.num_experts - 1)

        return (
            output,
            expert_representations,
            newbie_output,
            expert_idx,
            similarity_weights,
            similarities_pos,
            similarities_neg,
        )

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

    def update_signature_matrix(self, pos_representations, neg_representations, device):

        pos_medoids = representative_cluster(pos_representations)
        neg_medoids = representative_cluster(neg_representations)

        # Update the signature matrix with separate embeddings for positive and negative samples
        for idx in range(self.num_experts):
            self.signature_matrix[idx * 2] = pos_medoids[idx].to(device)  # Positive
            self.signature_matrix[idx * 2 + 1] = neg_medoids[idx].to(device)  # Negative

    def train_epoch(self, train_loader, criterion, optimizer, device):
        self.train()
        train_loss = 0
        correct = 0
        total = 0

        pos_representations_for_clustering = [[] for _ in range(self.num_experts)]
        neg_representations_for_clustering = [[] for _ in range(self.num_experts)]

        epoch_start_training = time.time()
        for i, batch_x in enumerate(train_loader):
            _, inputs_x, labels_x, domains_x = batch_x
            inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

            optimizer.zero_grad()
            outputs, newbie, representations, all_representations = self.forward_train(
                inputs_x,
                domains_x,
            )
            # Separate positive and negative embeddings for clustering
            pos_labels = labels_x[:, 1] > 0.5
            neg_labels = ~pos_labels
            # Update signature sums and counts
            for idx in range(self.num_experts):
                mask_pos = (domains_x == idx) & pos_labels
                mask_neg = (domains_x == idx) & neg_labels
                if mask_pos.any():
                    sampled_pos_representations = (
                        representations[mask_pos, :].detach().cpu().numpy().tolist()
                    )
                    pos_representations_for_clustering[idx].extend(
                        sampled_pos_representations
                    )

                if mask_neg.any():
                    sampled_neg_representations = (
                        representations[mask_neg, :].detach().cpu().numpy().tolist()
                    )

                    neg_representations_for_clustering[idx].extend(
                        sampled_neg_representations
                    )

            triplet_loss = self.contrastive_loss(
                all_representations, domains_x, labels_x[:, 1]
            )
            bce_loss = criterion(outputs, labels_x.float())
            loss_cr = ((newbie - outputs) ** 2).sum(1).mean()

            loss = bce_loss + triplet_loss + loss_cr
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels_x).float().sum().item()
            total += labels_x.numel()
        train_accuracy = correct / total
        train_loss /= len(train_loader)

        # Update the signature matrix
        self.update_signature_matrix(
            pos_representations_for_clustering,
            neg_representations_for_clustering,
            device,
        )
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
                _, inputs_x, labels_x, domains_x = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                outputs, representations, newbie, _, _, _, _ = self.forward_inference(
                    inputs_x,
                )

                if criterion is not None:
                    loss_cr = ((newbie - outputs) ** 2).sum(1).mean()
                    bce_loss = criterion(outputs, labels_x.float()).item()
                    triplet_loss = self.contrastive_loss(
                        representations, domains_x, labels_x[:, 1]
                    ).item()
                    loss += bce_loss + triplet_loss + loss_cr

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
        self.signature_matrix = torch.rand(
            self.num_experts * 2, self.representation_size, device=device
        )

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
