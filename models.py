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

        return out


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


class MasterModel(StandardModel):
    def __init__(
        self,
        num_experts=2,
        class_weights_tensor=None,
        num_classes=2,
        skip_connection=True,
    ):
        super(MasterModel, self).__init__(
            class_weights_tensor=class_weights_tensor,
            num_classes=num_classes,
            num_experts=num_experts,
        )
        self.skip_connection = skip_connection
        self.shared_extractor = SharedFeatureExtractor()
        self.experts = nn.ModuleList(
            [ExpertModel(num_classes) for _ in range(num_experts)]
        )
        self.aggregation_layer = nn.Linear(num_experts * num_classes, num_classes)
        self.final_classification_layer = (
            nn.Linear(19458, num_classes) if skip_connection else None
        )
        self.expert_selector = nn.Linear(num_experts * num_classes, self.num_experts)

    def forward(self, x, expert_idx=None):
        shared_features, intermediate1, intermediate2 = self.shared_extractor(x)
        expert_outputs = torch.zeros(
            x.size(0), self.num_experts, self.num_classes, device=x.device
        )
        all_expert_outputs = [
            self.experts[i](shared_features) for i in range(self.num_experts)
        ]
        all_expert_outputs = torch.stack(
            all_expert_outputs, dim=1
        )  # Shape: [batch_size, num_experts, num_classes]

        expert_selection_logits = self.expert_selector(
            all_expert_outputs.view(x.size(0), -1)
        )
        # Get the expert_idx with the highest confidence score
        expert_selection_logits = expert_selection_logits.argmax(dim=1)
        if expert_idx is not None:
            for i, idx in enumerate(expert_idx):
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                expert_output = self.experts[idx](shared_features[i].unsqueeze(0))
                expert_outputs[i, idx] = expert_output.squeeze(0)
        else:
            # Case for unknown expert_idx, such as in a test set
            for i in range(x.size(0)):
                best_expert_idx = expert_selection_logits[i].item()
                expert_outputs[i, :] = all_expert_outputs[i, best_expert_idx, :]

        # Continue with aggregation of expert outputs
        expert_outputs_flattened = expert_outputs.view(x.size(0), -1)
        final_output = self.aggregation_layer(expert_outputs_flattened)
        if self.skip_connection:
            intermediate1_flat = intermediate1.view(x.size(0), -1)
            intermediate2_flat = intermediate2.view(x.size(0), -1)
            combined_features = torch.cat(
                [final_output, intermediate1_flat, intermediate2_flat], dim=1
            )
            final_output = self.final_classification_layer(combined_features)

        return final_output, expert_outputs, expert_idx, expert_selection_logits
