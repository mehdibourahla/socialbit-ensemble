from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import seaborn as sns
import wandb


def log_message(message):
    line = ""
    for key, value in message.items():
        line += f"{key}: {round(value,4) if type(value) == float else value} | "
    print(line)
    if wandb.run is not None:
        wandb.log(message)


def setup_wandb(args):
    wandb.login(key=args.wandb_key)
    config = {}
    for key, value in vars(args).items():
        config[key] = value
    wandb.init(project="socialbit-ensemble", config=config)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, output_dir="/", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            output_dir (str): Directory to save the model.
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = os.path.join(output_dir, "early_stopping.pth")
        self.signature_matrix_path = os.path.join(output_dir, "signature_matrix.pth")
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.model_path)
        if hasattr(model, "signature_matrix"):
            torch.save(model.signature_matrix, self.signature_matrix_path)
        self.val_loss_min = val_loss


def representative_cluster(X, check=False):
    medoids = []

    for expert_index, expert in enumerate(X):
        # Compute intra-cluster distances once
        pairwise_distances = squareform(pdist(expert, "cosine"))
        intra_cluster_distances = pairwise_distances.sum(axis=0)

        # Initialize inter-cluster distances
        inter_cluster_distances = np.zeros(pairwise_distances.shape[0])

        # Use KDTree for efficient distance calculation
        expert_tree = cKDTree(expert)

        for other_index, other_expert in enumerate(X):
            if other_index != expert_index:
                # Using KDTree query to find the minimum distance to points in the other cluster
                dist, _ = expert_tree.query(other_expert, k=1)
                inter_cluster_distances += dist.sum()

        # Adjust score
        score = intra_cluster_distances - inter_cluster_distances
        medoid_index = np.argmin(score)
        medoids.append(expert[medoid_index])

    medoids = torch.tensor(medoids, dtype=torch.float32)

    # Optional: Check the distance between medoids
    if check:
        pairwise_distances = squareform(pdist(np.array(medoids), "cosine"))
        sns.heatmap(pairwise_distances, annot=True, cmap="viridis", square=True)
        plt.title("Pairwise Distance Matrix")
        plt.xlabel("Index")
        plt.ylabel("Index")
        plt.show()

    return medoids


def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    output_dir,
):
    output_path = os.path.join(output_dir, "training_curves.png")
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)


def process_data(representations_over_epochs):
    filtered_representations = []
    filtered_domains = []
    filtered_labels = []
    for epoch in representations_over_epochs:
        epoch_representations = []
        epoch_domains = []
        epoch_labels = []
        for sample in epoch:
            for key, batch in sample.items():
                if batch.shape[0] != 32:
                    continue
                if key == "domains":
                    epoch_domains.append(batch)
                elif key == "labels":
                    epoch_labels.append(batch)
                else:
                    epoch_representations.append(batch)
        filtered_representations.append(epoch_representations)
        filtered_domains.append(epoch_domains)
        filtered_labels.append(epoch_labels)
    filtered_representations = np.array(filtered_representations)
    num_experts = filtered_representations.shape[3]
    num_features = filtered_representations.shape[4]

    filtered_representations = filtered_representations.reshape(
        -1, num_experts * num_features
    )

    filtered_labels = np.array(filtered_labels)
    filtered_labels = filtered_labels[
        :, :, :, 1
    ]  # Get the labels for the social interaction
    filtered_labels = filtered_labels.flatten()

    filtered_domains = np.array(filtered_domains)
    filtered_domains = filtered_domains.flatten()

    tsne_results = TSNE(n_components=2, random_state=42).fit_transform(
        filtered_representations
    )

    return tsne_results, filtered_domains, filtered_labels


def plot_tsne_by_domain_epoch(
    tsne_results, domains, num_epochs, samples_per_epoch, output_dir
):
    output_path = os.path.join(output_dir, "tsne_by_domain_epoch.png")
    colors = plt.cm.rainbow(np.linspace(0, 1, num_epochs))
    unique_domains = np.unique(domains)
    marker_styles = ["o", "v", "*", "x"]
    domain_to_marker = {
        domain: marker for domain, marker in zip(unique_domains, marker_styles)
    }
    epoch_patches = [
        mpatches.Patch(color=colors[i], label=f"Epoch {i+1}") for i in range(num_epochs)
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(num_epochs):
        start_idx = i * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        epoch_domains = domains[start_idx:end_idx]

        for domain, marker in domain_to_marker.items():
            domain_indices = np.where(epoch_domains == domain)[0] + start_idx
            ax.scatter(
                tsne_results[domain_indices, 0],
                tsne_results[domain_indices, 1],
                marker=marker,
                color=colors[i],
                label=f"Domain {domain+1}" if i == 0 else "",
                s=25,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = {
        label: handle for label, handle in zip(labels, handles) if "Domain" in label
    }
    domain_patches = list(unique.values())

    leg1 = ax.legend(handles=domain_patches, bbox_to_anchor=(1, 1), title="Domains")

    ax.add_artist(leg1)

    ax.legend(handles=epoch_patches, bbox_to_anchor=(1, 0.5), title="Epochs")

    plt.title("t-SNE visualization of learned representations by Domain and Epoch")
    plt.xlabel("t-SNE axis 1")
    plt.ylabel("t-SNE axis 2")
    plt.savefig(output_path)


def plot_tsne_by_label_epoch(
    tsne_results, labels, num_epochs, samples_per_epoch, output_dir
):
    output_path = os.path.join(output_dir, "tsne_by_label_epoch.png")

    colors = plt.cm.rainbow(np.linspace(0, 1, num_epochs))
    unique_labels = np.unique(labels)
    marker_styles = ["o", "x"]  # For binary labels
    epoch_patches = [
        mpatches.Patch(color=colors[i], label=f"Epoch {i+1}") for i in range(num_epochs)
    ]
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(num_epochs):
        start_idx = i * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        epoch_labels = labels[start_idx:end_idx]

        for label, marker in zip(unique_labels, marker_styles):
            label_indices = np.where(epoch_labels == label)[0] + start_idx
            ax.scatter(
                tsne_results[label_indices, 0],
                tsne_results[label_indices, 1],
                marker=marker,
                color=colors[i],
                label=f"Label {label+1}" if i == 0 else "",
                s=25,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = {
        label: handle for label, handle in zip(labels, handles) if "Label" in label
    }
    label_patches = list(unique.values())

    leg1 = ax.legend(handles=label_patches, bbox_to_anchor=(1, 1), title="Labels")

    ax.add_artist(leg1)

    ax.legend(handles=epoch_patches, bbox_to_anchor=(1, 0.5), title="Epochs")

    plt.title("t-SNE visualization of learned representations by label and Epoch")
    plt.xlabel("t-SNE axis 1")
    plt.ylabel("t-SNE axis 2")
    plt.savefig(output_path)


def load_csv_files(directory_path):
    # Define the file names
    file_names = ["train.csv", "val.csv", "test.csv"]

    # Initialize an empty dictionary to store the dataframes
    dataframes = {}

    # Iterate through the file names and load each into a DataFrame
    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        if os.path.exists(file_path):
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Transform "dataset" column into a categorical variable
            df["dataset"] = pd.Categorical(df["dataset"]).codes
            dataframes[file_name[:-4]] = df
        else:
            print(f"File {file_name} not found in directory.")

    # Return the DataFrames
    return dataframes.get("train"), dataframes.get("val"), dataframes.get("test")


def setup_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def load_and_prepare_data(file_path, i_fold, j_subfold, balance_data=False):
    df = pd.read_csv(file_path)
    # Encode the 'dataset' column as categorical codes
    df["source"] = df["dataset"].apply(lambda x: x.split("_")[0])
    # df["source"] = df["dataset"]
    df["dataset"] = pd.Categorical(df["dataset"]).codes

    test_fold = df[(df["dataset"] == i_fold) | (df["dataset"] == i_fold + 5)]

    validation_fold = df[
        (df["dataset"] == j_subfold) | (df["dataset"] == j_subfold + 5)
    ]

    training_fold = df[
        ~df["dataset"].isin([i_fold, i_fold + 5, j_subfold, j_subfold + 5])
    ]

    training_fold_codes = pd.Categorical(training_fold["dataset"]).codes
    training_fold.loc[:, "dataset"] = training_fold_codes

    if balance_data:
        num_experts = len(training_fold["dataset"].unique())
        training_fold_balanced = []
        for expert_idx in range(num_experts):
            expert_data = training_fold[training_fold["dataset"] == expert_idx]
            pos_class = expert_data[expert_data["is_social"] == 1]
            neg_class = expert_data[expert_data["is_social"] == 0]
            if len(neg_class) > len(pos_class):
                neg_class = neg_class.sample(
                    n=len(pos_class), random_state=42, replace=False
                )
            else:
                pos_class = pos_class.sample(
                    n=len(neg_class), random_state=42, replace=False
                )
            training_fold_balanced.append(pos_class)
            training_fold_balanced.append(neg_class)
        training_fold = pd.concat(training_fold_balanced)

    # Print the number of samples in each dataset
    num_experts = len(training_fold["dataset"].unique())
    for i in range(num_experts):
        expert_samples = len(training_fold[training_fold["dataset"] == i])
        log_message(
            {
                f"Expert {i+1} Samples": expert_samples,
            }
        )

    log_message(
        {
            "Training Fold Size": len(training_fold),
            "Validation Fold Size": len(validation_fold),
            "Test Fold Size": len(test_fold),
            "Training Fold Positives": len(
                training_fold[training_fold["is_social"] == 1]
            ),
            "Training Fold Negatives": len(
                training_fold[training_fold["is_social"] == 0]
            ),
            "Validation Fold Positives": len(
                validation_fold[validation_fold["is_social"] == 1]
            ),
            "Validation Fold Negatives": len(
                validation_fold[validation_fold["is_social"] == 0]
            ),
            "Test Fold Positives": len(test_fold[test_fold["is_social"] == 1]),
            "Test Fold Negatives": len(test_fold[test_fold["is_social"] == 0]),
        }
    )

    return training_fold, validation_fold, test_fold
