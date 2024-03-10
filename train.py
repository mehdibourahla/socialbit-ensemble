import os
import argparse
from data_loader import YAMNetFeaturesDatasetEAR, YAMNetFeaturesDatasetDavid
from sklearn.utils.class_weight import compute_class_weight
from models import MasterModel, BiLSTMModel
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb
from utils import (
    process_data,
    plot_training_curves,
    plot_tsne_by_domain_epoch,
    plot_tsne_by_label_epoch,
    setup_directories,
    load_and_prepare_data,
    setup_wandb,
)


def compute_weights(training_fold):
    return compute_class_weight(
        "balanced",
        classes=np.unique(training_fold["is_social"]),
        y=training_fold["is_social"],
    )


def initialize_model(args, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    if args.baseline:
        model = BiLSTMModel(class_weights_tensor=class_weights_tensor).to(device)
    else:
        model = MasterModel(
            num_experts=args.num_experts,
            class_weights_tensor=class_weights_tensor,
        ).to(device)
    return model, device


def get_data_loaders(training_fold, validation_fold, test_fold):
    train_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(training_fold), batch_size=32, shuffle=True
    )
    val_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(validation_fold), batch_size=32, shuffle=True
    )
    test_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(test_fold), batch_size=32, shuffle=True
    )
    return train_gen, val_gen, test_gen


def train_and_evaluate_model(
    model, train_gen, val_gen, test_gen, device, current_output_dir, args
):
    epochs = args.epochs
    output_dir = args.output_dir
    use_metadata = args.metadata
    alpha = float(args.alpha / 10)
    (
        signature_matrix,
        signature_matrix_over_epochs,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
    ) = model.train_model(
        train_gen,
        val_gen,
        device,
        output_dir,
        epochs=epochs,
        use_metadata=use_metadata,
        alpha=alpha,
    )
    # Save the model and signature matrix
    torch.save(model.state_dict(), os.path.join(current_output_dir, "model.pth"))
    torch.save(
        signature_matrix, os.path.join(current_output_dir, "signature_matrix.pth")
    )

    print("Starting evaluation on test set...")
    evaluate_and_save_results(
        model,
        test_gen,
        signature_matrix,
        device,
        os.path.join(current_output_dir, "test_results.txt"),
        "test_predictions.csv",
    )
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, current_output_dir
    )
    # print("Starting plotting t-SNE...")
    # tsne_results, domains, labels = process_data(meta_over_epochs)
    # num_epochs = len(meta_over_epochs)
    # samples_per_epoch = len(meta_over_epochs[0])
    # plot_tsne_by_domain_epoch(
    #     tsne_results, domains, num_epochs, samples_per_epoch, current_output_dir
    # )
    # plot_tsne_by_label_epoch(
    #     tsne_results, labels, num_epochs, samples_per_epoch, current_output_dir
    # )
    return model, signature_matrix, signature_matrix_over_epochs


def evaluate_and_save_results(
    model,
    data_loader,
    signature_matrix,
    device,
    results_file_path,
    predictions_file_name,
):
    test_accuracy, sensitivity, specificity, predictions = model.evaluate_model(
        data_loader, signature_matrix, device
    )
    with open(results_file_path, "w") as f:
        f.write(
            f"Test Accuracy: {test_accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n"
        )
    pd.DataFrame(predictions).to_csv(
        os.path.join(os.path.dirname(results_file_path), predictions_file_name),
        index=False,
    )

    return predictions


# List of hyperparameters/variants to try
# 1. The way that I am splitting the data into domains
#   - Currently, I am using KMeans to split the data into 8 domains
# 2. The distance metric used in Triplet loss
#   - Currently, I am using cosine distance
# 3. The distance metric used to compare the signature matrix and the input data
#   - Currently, I am using Pearson correlation
# 4. The way that I constructing the signature matrix
#   - Currently, I am using the medoid of each expert representation as the signature
#   - This uses Eucledian distance to find the medoid


def main(args):
    setup_wandb(args)
    setup_directories(args.output_dir)

    current_output_dir = os.path.join(
        args.output_dir,
        f"fold_{args.i_fold + 1}",
        f"subfold_{args.j_subfold + 1}",
    )
    setup_directories(current_output_dir)

    training_fold, validation_fold, test_fold = load_and_prepare_data(
        args.data_dir, args.i_fold, args.j_subfold
    )
    class_weights = compute_weights(training_fold)
    model, device = initialize_model(args, class_weights)
    train_gen, val_gen, test_gen = get_data_loaders(
        training_fold, validation_fold, test_fold
    )
    model, signature_matrix, _ = train_and_evaluate_model(
        model,
        train_gen,
        val_gen,
        test_gen,
        device,
        current_output_dir,
        args=args,
    )

    # Evaluate on external test data
    ext_test_df = pd.read_csv(os.path.join(args.ext_test_data_dir, "metadata.csv"))
    ext_test_gen = DataLoader(
        YAMNetFeaturesDatasetDavid(ext_test_df, args.ext_test_data_dir),
        batch_size=32,
        shuffle=False,
    )

    evaluate_and_save_results(
        model,
        ext_test_gen,
        signature_matrix,
        device,
        os.path.join(current_output_dir, "ext_test_results.txt"),
        "ext_test_predictions.csv",
    )
    wandb.finish()


def initialize_args(parser):
    parser.add_argument("--wandb_key", type=str, help="Wandb API key")
    parser.add_argument("--commit_id", type=str, help="Commit ID")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
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

    parser.add_argument("--baseline", action="store_true", help="Use Baseline model")
    parser.add_argument("--metadata", action="store_true", help="Use metadata")
    parser.add_argument("--alpha", type=int, default=5, help="Alpha for the loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
