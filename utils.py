from sklearn.manifold import TSNE
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import wandb


def setup_wandb(args):
    wandb.login(key=args.wandb_key)
    config = {
        "commit_id": args.commit_id,
        "i_fold": args.i_fold,
        "j_subfold": args.j_subfold,
        "num_experts": args.num_experts,
        "is_baseline": args.baseline,
    }
    wandb.init(project="socialbit-ensemble", config=config)


def representative_cluster(X, check=False):
    # TODO: Optimize this function, it is slowing down the training process
    medoids = []
    for expert in X:
        pairwise_distances = squareform(pdist(expert, "cosine"))
        n = pairwise_distances.shape[0]  # Number of points in the current cluster

        # Calculate the sum of distances to points in the same cluster
        intra_cluster_distances = pairwise_distances.sum(axis=0)

        # Calculate the sum of distances to points in other clusters
        inter_cluster_distances = np.zeros(n)
        for other_expert in X:
            if not np.array_equal(other_expert, expert):
                for point in expert:
                    dist_to_other_cluster = np.min(
                        [
                            np.linalg.norm(point - other_point)
                            for other_point in other_expert
                        ]
                    )
                    inter_cluster_distances += dist_to_other_cluster

        # Adjust score to find a balance between intra and inter cluster distances
        score = intra_cluster_distances - inter_cluster_distances
        medoid_index = np.argmin(score)
        medoids.append(expert[medoid_index])

    # Check the distance between medoids
    if check:
        pairwise_distances = squareform(pdist(medoids, "cosine"))
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


def setup_logging(output_dir):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f"{output_dir}/training.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Logger is configured.")


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


def load_and_prepare_data(data_dir, i_fold, j_subfold, num_folds=5):
    train_data, val_data, test_data = load_csv_files(data_dir)
    data = pd.concat([train_data, val_data, test_data])
    fold_size = len(data) // num_folds
    folds = [data[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)]
    test_fold = folds[i_fold]
    validation_fold = folds[j_subfold]
    training_fold = pd.concat(
        folds[i] for i in range(num_folds) if i not in [i_fold, j_subfold]
    )
    return training_fold, validation_fold, test_fold
