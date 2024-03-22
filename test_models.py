import os
import pandas as pd
from models import MasterModel
from data_loader import YAMNetFeaturesDatasetEAR, YAMNetFeaturesDatasetDavid
import torch
from torch.utils.data import DataLoader
from train import evaluate_and_save_results
import argparse
import wandb


def initialize_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signature_matrix_path = os.path.join(args.output_dir, "signature_matrix.pth")
    model_path = os.path.join(args.output_dir, "model.pth")
    model = MasterModel(
        num_experts=args.num_experts,
    ).to(device)
    signature_matrix = torch.load(signature_matrix_path, map_location=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, signature_matrix, device


def test_models(model, test_gen, signature_matrix, device, args):

    results = evaluate_and_save_results(
        model,
        test_gen,
        signature_matrix,
        device,
        os.path.join(args.output_dir, "test_results.txt"),
        "test_predictions.csv",
    )
    return results


def get_data_loaders(args):
    test_df = pd.read_csv(args.ear_ground_truth)
    test_df["dataset"] = pd.Categorical(test_df["dataset"]).codes
    ear_test_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(test_df),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    ext_test_df = pd.read_csv(os.path.join(args.lab_ground_truth, "metadata.csv"))
    ext_test_gen = DataLoader(
        YAMNetFeaturesDatasetDavid(ext_test_df, args.lab_ground_truth),
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    return ear_test_gen, ext_test_gen


def main(args):
    wandb.login(key=args.wandb_key)
    config = {
        "commit_id": args.commit_id,
        "num_experts": args.num_experts,
    }
    wandb.init(project="socialbit-ensemble", config=config)

    os.makedirs(args.output_dir, exist_ok=True)
    model, signature_matrix, device = initialize_model(args)
    ear_test_gen, lab_test_gen = get_data_loaders(args)
    results_ear = evaluate_and_save_results(
        model,
        ear_test_gen,
        signature_matrix,
        device,
        args.output_dir,
        "EAR",
    )
    results_lab = evaluate_and_save_results(
        model,
        lab_test_gen,
        signature_matrix,
        device,
        args.output_dir,
        "LAB",
    )

    # Save results to csv file
    results.to_csv(os.path.join(args.output_dir, "weights-output.csv"), index=False)

    # Visualize csv file in wandb
    wandb.save(os.path.join(args.output_dir, "weights-output.csv"))


def initialize_args(parser):
    parser.add_argument("--commit_id", type=str, help="Commit ID")
    parser.add_argument("--wandb_key", type=str, help="Wandb API key")

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )
    parser.add_argument(
        "--ear_ground_truth",
        required=True,
        help="Path to the directory containing test dataset",
    )
    parser.add_argument(
        "--lab_ground_truth",
        required=True,
        help="Path to the directory containing test dataset",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=2,
        help="Number of experts to use in the MasterModel",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
