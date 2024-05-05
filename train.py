import os
import argparse
from data_loader import (
    YAMNetFeaturesDatasetEAR,
    YAMNetFeaturesDatasetDavid,
    YAMNetFeaturesSINS,
)
from sklearn.utils.class_weight import compute_class_weight
from models import MasterModel
from baseline import BiLSTMModel, TransformerModel
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb
from utils import (
    plot_training_curves,
    setup_directories,
    load_and_prepare_data,
    setup_wandb,
    log_message,
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
    if args.model == "bilstm":
        model = BiLSTMModel(class_weights_tensor=class_weights_tensor).to(device)
    elif args.model == "transformer":
        model = TransformerModel(class_weights_tensor=class_weights_tensor).to(device)
    else:
        coefficents = (args.alpha, args.beta, args.gamma)

        model = MasterModel(
            num_experts=args.num_experts,
            class_weights_tensor=class_weights_tensor,
            coefficents=coefficents,
        ).to(device)
    return model, device


def get_data_loaders(dataset, training_fold, validation_fold, test_fold):
    dataloader_class = (
        YAMNetFeaturesDatasetEAR if dataset == "EAR" else YAMNetFeaturesSINS
    )

    train_gen = DataLoader(
        dataloader_class(training_fold),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    val_gen = DataLoader(
        dataloader_class(validation_fold),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )
    test_gen = DataLoader(
        dataloader_class(test_fold),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )
    return train_gen, val_gen, test_gen


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

    # Compute class weights
    class_weights = compute_weights(training_fold)

    # Initialize model and device
    model, device = initialize_model(args, class_weights)

    # Get loaders Train the model
    train_gen, val_gen, test_gen = get_data_loaders(
        args.dataset, training_fold, validation_fold, test_fold
    )
    model.train_model(train_gen, val_gen, device, current_output_dir)

    if args.model == "master":
        torch.save(
            model.signature_matrix,
            os.path.join(current_output_dir, "signature_matrix.pth"),
        )
    _, accuracy, sensitivity, specificity, _ = model.evaluate(test_gen, device)
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
    _, accuracy, sensitivity, specificity, _ = model.evaluate(ext_test_gen, device)
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
    parser.add_argument("--metadata", action="store_true", help="Use metadata")
    parser.add_argument(
        "--balance_data", type="int", default=1, help="Use downsampled data"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha BCE loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta CR loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma Triplet loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
