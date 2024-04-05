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

    for idx, model in enumerate(models):
        model.to(device)
        model_predictions = []
        model_confidences = []
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

                if idx == 0:  # Collect labels only once
                    all_labels.extend(labels.numpy())

        model_predictions = np.concatenate(model_predictions, axis=0)
        model_confidences = np.concatenate(model_confidences, axis=0)
        all_predictions.append(model_predictions)
        all_confidences.append(model_confidences)

    all_predictions = np.array(all_predictions)  # Shape: (num_experts, num_samples)
    all_confidences = np.array(all_confidences)  # Shape: (num_experts, num_samples)

    # Weighted fusion based on confidence scores
    weighted_predictions = np.zeros_like(
        all_predictions[0], dtype=float
    )  # Shape: (num_samples,)
    for i in range(all_predictions.shape[1]):
        sample_predictions = all_predictions[:, i]
        sample_confidences = all_confidences[:, i]
        unique_predictions = np.unique(sample_predictions)

        weighted_sum = {pred: 0 for pred in unique_predictions}
        for pred, conf in zip(sample_predictions, sample_confidences):
            weighted_sum[pred] += conf

        final_prediction = max(weighted_sum, key=weighted_sum.get)
        weighted_predictions[i] = final_prediction

    # Calculate and return metrics for fused predictions
    weighted_predictions = torch.eye(2)[weighted_predictions].numpy()
    all_labels = torch.eye(2)[all_labels].numpy()
    accuracy, sensitivity, specificity = compute_metrics(
        weighted_predictions, all_labels
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
        args.data_dir, args.i_fold, args.j_subfold
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
        train_losses, val_losses, train_accuracies, val_accuracies = model.train_model(
            train_gen, val_gen, device, current_output_dir
        )

        # Save model and log results
        result_dir = os.path.join(current_output_dir, f"expert_{expert_idx}")
        os.makedirs(result_dir, exist_ok=True)
        plot_training_curves(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            result_dir,
        )
        torch.save(model.state_dict(), os.path.join(result_dir, "model.pth"))

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
