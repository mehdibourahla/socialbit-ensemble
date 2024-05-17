import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import representative_cluster, log_message, EarlyStopping
import time


class ExpertModel(nn.Module):
    def __init__(
        self,
        input_size=1024,
        hidden_size=150,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.25,
    ):
        super(ExpertModel, self).__init__()
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

        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, requires_grad=True
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, requires_grad=True
        ).to(x.device)

        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        out = out.squeeze()
        return out, lstm_out


class MasterModel(nn.Module):
    def __init__(
        self,
        num_experts=2,
        num_classes=2,
        class_weights_tensor=None,
        signature_matrix=None,
        representation_size=300,
        coefficents=(0.5, 0.25, 0.25),
    ):
        super(MasterModel, self).__init__()
        self.class_weights_tensor = class_weights_tensor
        self.signature_matrix = signature_matrix
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.representation_size = representation_size
        (
            self.alpha,
            self.beta,
            self.gamma,
        ) = coefficents
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
            expert_outputs[:, i] = output
            expert_representations[:, i, :] = representation.reshape(
                shared_features.size(0), -1
            )  # Flatten the representation

        return expert_outputs, expert_representations

    def compute_similarity_weights(self, expert_representations):
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
            signature_pos = self.signature_matrix[pos_index, :].unsqueeze(0)
            signature_neg = self.signature_matrix[neg_index, :].unsqueeze(0)

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
            output[i] = expert_outputs[i, expert_idx[i]]
            representation[i, :] = expert_representations[i, expert_idx[i], :]

        # Get the newbie output
        for i in range(self.num_experts):
            mask = expert_idx != i
            newbie_output[mask] += expert_outputs[mask, i]
        newbie_output = newbie_output / (self.num_experts - 1)

        return output, newbie_output, representation, expert_representations

    def forward_inference(self, x):
        # shared_features = self.shared_extractor(x)
        expert_outputs, expert_representations = self.process_experts(x)
        return (
            expert_outputs,
            expert_representations,
        )

    def get_positive_example(self, representations, sources, domains, labels, i):
        anchor_domain = domains[i].item()
        anchor_label = labels[i].item()
        anchor_source = sources[i]

        positive_indices = [
            idx
            for idx, (domain, label, source) in enumerate(zip(domains, labels, sources))
            if (
                source != anchor_source
                and (domain.item() == anchor_domain or label.item() == anchor_label)
            )
            and idx != i
        ]

        if not positive_indices:
            return representations[i]

        positive_idx = random.choice(positive_indices)
        return representations[positive_idx]

    def get_negative_example(self, representations, sources, domains, labels, i):
        """
        Select a negative example for the anchor, considering both domain and label.
        An example is negative if it has a different domain and a different label from the anchor.
        """
        anchor_domain = domains[i].item()
        anchor_label = labels[i].item()
        anchor_source = sources[i]

        negative_indices = [
            idx
            for idx, (domain, label, source) in enumerate(zip(domains, labels, sources))
            if (
                source != anchor_source
                and (domain.item() != anchor_domain and label.item() != anchor_label)
            )
        ]

        if not negative_indices:
            return representations[i]

        negative_idx = random.choice(negative_indices)
        return representations[negative_idx]

    def contrastive_loss(self, representations, sources, domains, labels, margin=0.05):
        # Representations shape is (batch_size, num_experts, representation_size)
        loss = torch.zeros(representations.size(0))
        for i in range(representations.size(0)):
            anchor = representations[i]  # Shape: (num_experts, representation_size)
            positive = self.get_positive_example(
                representations, sources, domains, labels, i
            )
            negative = self.get_negative_example(
                representations, sources, domains, labels, i
            )

            pos_sim = F.cosine_similarity(anchor, positive)
            pos_dist = 1 - pos_sim  # 0 if postive is the anchor

            neg_sim = F.cosine_similarity(anchor, negative)
            neg_dist = 1 - neg_sim
            # Triplet loss calculation with cosine distance
            triplet_loss = F.relu(abs(pos_dist - neg_dist) + margin)
            loss[i] = triplet_loss.mean()

        return loss

    def update_signature_matrix(self, pos_representations, neg_representations, device):

        pos_medoids = representative_cluster(pos_representations)
        neg_medoids = representative_cluster(neg_representations)

        # Update the signature matrix with separate embeddings for positive and negative samples
        for idx in range(self.num_experts):
            self.signature_matrix[idx * 2] = pos_medoids[idx].to(device)  # Positive
            self.signature_matrix[idx * 2 + 1] = neg_medoids[idx].to(device)  # Negative

    def train_epoch(self, train_loader, optimizer, device):
        self.train()
        train_loss = 0
        correct = 0
        total = 0

        pos_representations_for_clustering = [[] for _ in range(self.num_experts)]
        neg_representations_for_clustering = [[] for _ in range(self.num_experts)]

        epoch_start_training = time.time()
        for i, batch_x in enumerate(train_loader):
            # Get the inputs and labels and get batch weights
            _, source_x, inputs_x, labels_x, domains_x = batch_x
            inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)
            label_decoded = torch.argmax(labels_x, dim=1)

            batch_weight = self.class_weights_tensor[label_decoded].to(device)

            optimizer.zero_grad()
            outputs, newbie, representations, all_representations = self.forward_train(
                inputs_x, domains_x
            )

            # Separate positive and negative embeddings for clustering
            pos_labels = label_decoded == 1
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

            # Apply the batch weights to combined losses
            triplet_loss = (
                self.contrastive_loss(
                    all_representations, source_x, domains_x, label_decoded
                )
                if self.gamma > 0
                else 0
            )
            loss_cr = ((newbie - outputs) ** 2).sum() if self.beta > 0 else 0
            bce_loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels_x.float())

            loss = (
                self.alpha * bce_loss + self.beta * loss_cr + self.gamma * triplet_loss
            )
            loss = (loss * batch_weight).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            labels_x = torch.argmax(labels_x, dim=1)
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

        del (
            pos_representations_for_clustering,
            neg_representations_for_clustering,
            outputs,
            newbie,
            representations,
            all_representations,
            bce_loss,
            loss,
            preds,
        )

        return train_loss, train_accuracy, epoch_end_training - epoch_start_training

    def evaluate(
        self,
        data_loader,
        device,
    ):
        TP, TN, FP, FN, loss, correct, total = 0, 0, 0, 0, 0, 0, 0
        # Initialize tensors for storage on device

        epoch_start_validation = time.time()
        with torch.no_grad():
            for _, (_, _, inputs_x, labels_x, _) in enumerate(data_loader):
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)

                expert_outputs = torch.zeros(
                    inputs_x.size(0),
                    self.num_experts,
                    self.num_classes,
                    device=device,
                    requires_grad=False,
                )
                expert_representations = torch.zeros(
                    inputs_x.size(0),
                    self.num_experts,
                    self.representation_size,
                    device=device,
                    requires_grad=False,
                )

                for idx, expert in enumerate(self.experts):
                    expert.eval()
                    outputs, representations = expert(inputs_x)
                    expert_outputs[:, idx] = outputs
                    expert_representations[:, idx, :] = representations
                    expert.zero_grad()

                signature_similarities, _, _ = self.compute_similarity_weights(
                    expert_representations
                )

                weighted_similarities = F.softmax(signature_similarities, dim=1)
                chosen_outputs = torch.zeros(
                    inputs_x.size(0), self.num_classes, device=device
                )

                for idx in range(self.num_experts):
                    chosen_outputs += (
                        weighted_similarities[:, idx].unsqueeze(1)
                        * expert_outputs[:, idx]
                    )
                bce_loss = nn.CrossEntropyLoss()(chosen_outputs, labels_x.float())
                loss += bce_loss

                preds = torch.argmax(chosen_outputs, dim=1)
                labels_x = torch.argmax(labels_x, dim=1)
                # Compute confusion matrix components
                TP += ((preds == 1) & (labels_x == 1)).sum().item()
                TN += ((preds == 0) & (labels_x == 0)).sum().item()
                FP += ((preds == 1) & (labels_x == 0)).sum().item()
                FN += ((preds == 0) & (labels_x == 1)).sum().item()

                correct += (preds == labels_x).float().sum().item()
                total += labels_x.numel()

        epoch_end_validation = time.time()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = correct / total
        loss /= len(data_loader)
        loss = loss.item()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=0.001,
            output_dir=output_dir,
            verbose=True,
        )
        self.signature_matrix = torch.zeros(
            self.num_experts * 2, self.representation_size, device=device
        )

        for epoch in range(num_epochs):
            train_loss, train_accuracy, training_time = self.train_epoch(
                train_loader, optimizer, device
            )
            val_loss, val_accuracy, _, _, validation_time = self.evaluate(
                val_loader, device
            )

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

            early_stopping(val_loss, self)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.load_state_dict(torch.load(early_stopping.model_path))
        self.signature_matrix = torch.load(early_stopping.signature_matrix_path)
