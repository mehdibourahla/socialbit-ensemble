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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
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

    def forward(self, x, expert_idx=None, signature_matrix=None):
        shared_features = self.shared_extractor(x)

        expert_outputs = torch.zeros(
            x.size(0), self.num_experts, self.num_classes, device=x.device
        )
        expert_representations = torch.zeros(
            x.size(0),
            self.num_experts,
            64 * 3,
            device=x.device,  # Flattened representation
        )

        # Process input through each expert's network
        for i in range(self.num_experts):
            output, representation = self.experts[i](shared_features)
            expert_outputs[:, i, :] = output
            expert_representations[:, i, :] = representation.view(
                x.size(0), -1
            )  # Flatten the representation

        final_output = torch.zeros(x.size(0), self.num_classes, device=x.device)
        representations = torch.zeros(x.size(0), 64 * 3, device=x.device)

        if signature_matrix is not None:  # Inference mode
            expert_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            similarities = torch.empty(
                x.size(0), self.num_experts, signature_matrix.size(0), device=x.device
            )

            for i in range(self.num_experts):
                rep = expert_representations[:, i, :]
                for j in range(signature_matrix.size(0)):
                    signature = signature_matrix[j, :].unsqueeze(0)
                    similarities[:, i, j] = F.cosine_similarity(rep, signature, dim=1)

            # Normalize the weights for each expert across positive and negative signature
            similarity_weights = F.softmax(similarities, dim=1)
            expert_idx = torch.argmax(similarity_weights.sum(dim=2), dim=1)
            similarity_weights = F.softmax(similarity_weights.sum(dim=2), dim=1)
            # print(similarities.sum(dim=2))

            for i in range(self.num_experts):
                weighted_output = expert_outputs[:, i, :] * similarity_weights[
                    :, i
                ].unsqueeze(1)
                final_output += weighted_output
                weighted_rep = expert_representations[:, i, :] * similarity_weights[
                    :, i
                ].unsqueeze(1)
                representations += weighted_rep

        else:  # Training mode
            # Select the expert output based on the provided expert_idx for each sample
            for sample in range(x.size(0)):
                final_output[sample, :] = expert_outputs[sample, expert_idx[sample], :]
                representations[sample, :] = expert_representations[
                    sample, expert_idx[sample], :
                ]

        return final_output, representations, expert_idx, expert_outputs
