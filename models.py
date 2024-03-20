import torch
import torch.nn as nn
from standard_model import StandardModel
import torch.nn.functional as F


class BiLSTMModel(StandardModel):
    def __init__(
        self,
        class_weights_tensor=None,
        input_size=1024,
        hidden_size=150,
        num_layers=2,
        num_classes=2,
        dropout_rate=0.25,
    ):
        super(BiLSTMModel, self).__init__(
            class_weights_tensor=class_weights_tensor,
            num_classes=num_classes,
            num_experts=1,
            transpose_input=True,
        )
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
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        out = self.dropout(out[:, -1, :])  # Take the last time step
        out = self.fc(out)

        return out, None, None


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
        self.shared_norm = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.shared_norm(x)
        return x


class ExpertModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ExpertModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(64 * 3, num_classes)
        self.shared_norm = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)
        x1 = self.shared_norm(x1)
        x = x1.view(x1.size(0), -1)
        x = self.fc(x)
        return x, x1


class MasterModel(StandardModel):
    def __init__(
        self,
        num_experts=2,
        class_weights_tensor=None,
        num_classes=2,
    ):
        super(MasterModel, self).__init__(
            class_weights_tensor=class_weights_tensor,
            num_classes=num_classes,
            num_experts=num_experts,
        )
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList(
            [ExpertModel(num_classes) for _ in range(num_experts)]
        )

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
            64 * 3,
            device=shared_features.device,
        )  # Adjust the size accordingly

        for i, expert in enumerate(self.experts):
            output, representation = expert(shared_features)
            expert_outputs[:, i, :] = output
            expert_representations[:, i, :] = representation.view(
                shared_features.size(0), -1
            )  # Flatten the representation

        return expert_outputs, expert_representations

    def compute_similarity_weights(self, shared_featrues, signature_matrix):
        num_experts, _ = signature_matrix.size()
        batch_size, _ = shared_featrues.size()

        similarities_pos = torch.empty(
            batch_size, num_experts // 2, device=shared_featrues.device
        )
        similarities_neg = torch.empty(
            batch_size, num_experts // 2, device=shared_featrues.device
        )

        for i in range(batch_size):
            rep = shared_featrues[i, :]
            for j in range(num_experts):
                if j % 2 == 0:
                    signature_pos = signature_matrix[j, :]
                    similarities_pos[i, j // 2] = F.cosine_similarity(
                        rep, signature_pos, dim=0
                    )
                else:
                    signature_neg = signature_matrix[j, :]
                    similarities_neg[i, j // 2] = F.cosine_similarity(
                        rep, signature_neg, dim=0
                    )

        summed_similarities_expert = [
            similarities_pos[:, i] + similarities_neg[:, i]
            for i in range(num_experts // 2)
        ]

        summed_similarities_expert = torch.stack(summed_similarities_expert, dim=1)

        similarity_weights = F.softmax(summed_similarities_expert, dim=1)

        return similarity_weights

    def forward(self, x, expert_idx=None, inference_mode=False, signature_matrix=None):
        similarity_weights = None
        final_output = torch.zeros(x.size(0), self.num_classes, device=x.device)
        representations = torch.zeros(
            x.size(0), 64 * 3, device=x.device
        )  # Adjust the size accordingly

        shared_features = self.shared_extractor(x)  # Shape: (batch, 256, 9)
        # I am currently passing the shared features to the experts according to the domain id of the input data x
        flatten_shared_features = shared_features.view(
            shared_features.size(0), -1
        )  # Shape: (batch, 256*9)

        if inference_mode:
            # Signature Matrix has shape (2*num_experts, 256*9)
            # I need to compute the similarity weights between the flattened shared features and the signature matrix
            # to determine which expert to use
            similarity_weights = self.compute_similarity_weights(
                flatten_shared_features, signature_matrix
            )
            expert_idx = torch.argmax(similarity_weights, dim=1)

        # Initialize tensors for outputs and representations
        expert_outputs, expert_representations = self.process_experts(shared_features)

        for i in range(x.size(0)):
            if not inference_mode:
                final_output[i, :] = expert_outputs[i, expert_idx[i], :]
                representations[i, :] = expert_representations[i, expert_idx[i], :]
            else:
                for j in range(self.num_experts):
                    final_output[i, :] += (
                        similarity_weights[i, j] * expert_outputs[i, j, :]
                    )
                    representations[i, :] += (
                        similarity_weights[i, j] * expert_representations[i, j, :]
                    )

        return (
            final_output,
            representations,
            expert_idx,
            expert_outputs,
            similarity_weights,
            flatten_shared_features,
        )
