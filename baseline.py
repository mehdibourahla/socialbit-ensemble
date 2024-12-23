import torch.nn as nn
import torch
from utils import log_message, EarlyStopping
import time
import torch.nn.functional as F


class BaselineModel(nn.Module):

    def __init__(self, class_weights_tensor=None):
        super(BaselineModel, self).__init__()
        self.class_weights_tensor = class_weights_tensor

    def train_epoch(self, train_loader, optimizer, device):
        # CHANGES:
        # - Changed CrossEntropy to BCELoss
        # - Applied class weights at batch level
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_start_training = time.time()
        for i, batch_x in enumerate(train_loader):
            _, inputs_x, labels_x, domains_x = batch_x
            inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)
            batch_weight = self.class_weights_tensor[labels_x.long()].to(device)

            optimizer.zero_grad()
            outputs = self(inputs_x)
            loss = nn.BCELoss(reduction="none")(outputs, labels_x)
            loss = (loss * batch_weight).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.round(outputs).float()
            correct += (preds == labels_x).float().sum().item()
            total += labels_x.numel()
        train_accuracy = correct / total
        train_loss /= len(train_loader)
        epoch_end_training = time.time()

        return train_loss, train_accuracy, epoch_end_training - epoch_start_training

    def evaluate(self, data_loader, device):
        self.eval()
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        val_loss = 0
        correct = 0
        total = 0
        epoch_start_validation = time.time()
        with torch.no_grad():
            for i, batch_x in enumerate(data_loader):
                _, inputs_x, labels_x, _ = batch_x
                inputs_x, labels_x = inputs_x.to(device), labels_x.to(device)
                outputs = self(inputs_x)
                loss = nn.BCELoss()(outputs, labels_x)
                preds = torch.round(outputs).float()

                # Compute confusion matrix components
                TP += ((preds == 1) & (labels_x == 1)).float().sum().item()
                TN += ((preds == 0) & (labels_x == 0)).float().sum().item()
                FP += ((preds == 1) & (labels_x == 0)).float().sum().item()
                FN += ((preds == 0) & (labels_x == 1)).float().sum().item()

                correct += (preds == labels_x).float().sum().item()
                val_loss += loss.item()
                total += labels_x.numel()
        epoch_end_validation = time.time()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = correct / total
        val_loss /= len(data_loader)

        return (
            val_loss,
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
        # CHANGES:
        # - Changed learning rate to 3e-4 instead of 1e-5
        # - Increased early stopping patience to 20 instead of 10
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            delta=0.001,
            output_dir=output_dir,
            verbose=True,
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


class TransformerBlock(nn.Module):
    def __init__(self, feature_length, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_length,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_length, eps=1e-6)
        self.norm2 = nn.LayerNorm(feature_length, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Conv1d(in_channels=feature_length, out_channels=ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=ff_dim, out_channels=feature_length, kernel_size=1),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Adjust input shape from (batch, feature, seq_len) to (batch, seq_len, feature)
        x = x.permute(0, 2, 1)

        # Multi-Head Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + self.dropout1(x)

        # Feed-Forward Network
        residual = x
        x = self.norm2(x)
        # Adjust x back to (batch, feature, seq_len) for Conv1D, then revert
        x = self.ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = residual + self.dropout2(x)

        # Revert shape to (batch, feature, seq_len) for downstream processing
        x = x.permute(0, 2, 1)
        return x


class TransformerModel(BaselineModel):
    def __init__(
        self,
        feature_length=1024,
        num_transformers=2,
        num_heads=8,
        ff_dim=9,
        mlp_units=32,
        dropout=0.35,
        class_weights_tensor=None,
    ):
        super(TransformerModel, self).__init__(
            class_weights_tensor=class_weights_tensor
        )
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(feature_length, num_heads, ff_dim, dropout)
                for _ in range(num_transformers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_length, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, 2),
        )

    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        x = self.pool(x).squeeze(-1)

        x = self.mlp(x)
        return x


class BiLSTMModel(BaselineModel):
    def __init__(
        self,
        class_weights_tensor=None,
        input_size=1024,
        hidden_size=150,
        num_layers=2,
        num_classes=1,
        dropout_rate=0.25,
    ):
        super(BiLSTMModel, self).__init__(class_weights_tensor=class_weights_tensor)
        self.class_weights_tensor = class_weights_tensor
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # CHANGES:
        # - Pooling instead of last time step
        # - Sigmoid activation

        # Initialize hidden state and cell state
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, requires_grad=True
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size, requires_grad=True
        ).to(x.device)

        # Forward propagate the LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.squeeze()
        return out
