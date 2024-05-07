import os
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import wandb
from utils import (
    plot_training_curves,
    setup_directories,
    load_and_prepare_data,
    setup_wandb,
    log_message,
)
from train import compute_weights, initialize_model, get_data_loaders
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import YAMNetFeaturesDatasetDavid


def compute_metrics(preds, labels):

    preds = np.array(preds)
    labels = np.array(labels)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    accuracy = np.mean(preds == labels)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return accuracy, sensitivity, specificity


def fusion(models, test_gen, device):
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probabilities = []

    for idx, model in enumerate(models):
        model.to(device)
        model_predictions = []
        model_confidences = []
        model_probabilities = []
        model.eval()
        with torch.no_grad():
            for batch in test_gen:
                _, inputs, labels, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                positive_class_probabilities = outputs
                negative_class_probabilities = 1 - outputs
                probabilities = torch.cat(
                    (
                        negative_class_probabilities.unsqueeze(1),
                        positive_class_probabilities.unsqueeze(1),
                    ),
                    dim=1,
                )
                confidences, predicted_classes = torch.max(probabilities, dim=1)
                model_predictions.append(predicted_classes.cpu().numpy())
                model_confidences.append(confidences.cpu().numpy())
                model_probabilities.append(probabilities.cpu().numpy())

                if idx == 0:  # Collect labels only once
                    all_labels.extend(labels.numpy())

        model_predictions = np.concatenate(model_predictions, axis=0)
        model_confidences = np.concatenate(model_confidences, axis=0)
        model_probabilities = np.concatenate(model_probabilities, axis=0)
        all_predictions.append(model_predictions)
        all_confidences.append(model_confidences)
        all_probabilities.append(model_probabilities)

    all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples)
    all_confidences = np.array(all_confidences)  # Shape: (num_models, num_samples)
    all_probabilities = np.array(
        all_probabilities
    )  # Shape: (num_models, num_samples, 2)
    all_labels = np.array(all_labels)

    # Create a heatmap of the probabilities for positive and negative classes
    positive_indices = np.where(all_labels == 1)[0]
    negative_indices = np.where(all_labels == 0)[0]

    positive_probabilities = all_probabilities[
        :, positive_indices, 1
    ]  # Class 1 probabilities for true positive samples
    negative_probabilities = all_probabilities[
        :, negative_indices, 0
    ]  # Class 0 probabilities for true negative samples

    # Visualizing the confidences of the models for each sample as a heatmap
    positive_probabilities = positive_probabilities.T
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        positive_probabilities,
        cmap="RdYlGn",
        fmt=".2f",
        cbar_kws={"label": "Probability"},
    )
    plt.title("Positive Class Probabilities for True Positive Samples")
    plt.xlabel("Expert Index")
    plt.ylabel("Sample Index")
    ax.set_xticklabels(np.arange(1, len(models) + 1))

    # Save the heatmap as pdf
    plt.savefig("positive_prob_heatmap.pdf")

    negative_probabilities = negative_probabilities.T
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        negative_probabilities,
        cmap="RdYlGn",
        fmt=".2f",
        cbar_kws={"label": "Probability"},
    )
    plt.title("Negative Class Probabilities for True Negative Samples")
    plt.xlabel("Expert Index")
    plt.ylabel("Sample Index")
    ax.set_xticklabels(np.arange(1, len(models) + 1))

    # Save the heatmap as pdf
    plt.savefig("negative_prob_heatmap.pdf")

    # Implementing Dynamic Soft Voting
    final_predictions = np.zeros(all_predictions.shape[1], dtype=int)
    for i in range(all_predictions.shape[1]):
        sample_predictions = all_predictions[:, i]
        sample_confidences = all_confidences[:, i]
        weighted_votes = np.zeros(np.max(sample_predictions) + 1)

        for pred, conf in zip(sample_predictions, sample_confidences):
            weighted_votes[pred] += conf

        final_predictions[i] = np.argmax(weighted_votes)

    # Compute metrics
    accuracy, sensitivity, specificity = compute_metrics(
        final_predictions, np.array(all_labels)
    )
    return accuracy, sensitivity, specificity


def main(args):

    # Initialize wandb and create output directories
    setup_wandb(args)
    setup_directories(args.output_dir)
    current_output_dir = os.path.join(
        args.output_dir,
        f"fold_{args.i_fold + 1}",
        f"subfold_{args.j_subfold + 1}",
    )
    setup_directories(current_output_dir)

    # Load and prepare data
    training_fold, validation_fold, test_fold = load_and_prepare_data(
        args.data_dir,
        args.i_fold,
        args.j_subfold,
        True if args.balance_data > 0 else False,
    )

    models = []
    for expert_idx in range(args.num_experts):
        # Select the training_fold
        training_fold_idx = training_fold[training_fold["dataset"] == expert_idx]

        # Compute class weights
        class_weights = compute_weights(training_fold_idx)

        # Initialize model and device
        model, device = initialize_model(args, class_weights)

        # Get loaders Train the model
        train_gen, val_gen, test_gen = get_data_loaders(
            args.dataset, training_fold_idx, validation_fold, test_fold
        )
        model.train_model(
            train_gen, val_gen, device, current_output_dir
        )

        models.append(model)

    # Evaluate Fusion on test data
    print("Evaluating Fusion on test data...")
    accuracy, sensitivity, specificity = fusion(models, test_gen, device)
    log_message(
        {
            f"{args.dataset}_Accuracy": accuracy,
            f"{args.dataset}_Sensitivity": sensitivity,
            f"{args.dataset}_Specificity": specificity,
        }
    )

    # Evaluate on external test data
    ext_test_df = pd.read_csv(os.path.join(args.ext_test_data_dir, "metadata.csv"))
    seq_len = 30 if args.dataset == "EAR" else 60
    ext_test_gen = DataLoader(
        YAMNetFeaturesDatasetDavid(
            ext_test_df, args.ext_test_data_dir, seq_len=seq_len
        ),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )
    print("Evaluating on external test data...")
    accuracy, sensitivity, specificity = fusion(models, ext_test_gen, device)
    log_message(
        {
            "LAB_Accuracy": accuracy,
            "LAB_Sensitivity": sensitivity,
            "LAB_Specificity": specificity,
        }
    )

    wandb.finish()


def initialize_args(parser):
    parser.add_argument("--wandb_key", type=str, help="Wandb API key")
    parser.add_argument("--commit_id", type=str, help="Commit ID")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument(
        "--balance_data", type=int, default=1, help="Use downsampled data"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=2,
        help="Number of experts to use in the MasterModel",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the directory containing train, val and test datasets",
    )

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )

    parser.add_argument(
        "--ext_test_data_dir",
        required=True,
        help="Path to the directory containing external test dataset",
    )

    parser.add_argument("--i_fold", type=int, help="Fold number")
    parser.add_argument("--j_subfold", type=int, help="Subfold number")

    parser.add_argument("--model", type=str, required=True, help="Model to use")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
