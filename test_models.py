import os
import pandas as pd
from models import MasterModel
from data_loader import YAMNetFeaturesDatasetDavid
import torch
from torch.utils.data import DataLoader
from train import evaluate_and_save_results
import argparse
import wandb


def initialize_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MasterModel(
        num_experts=args.num_experts,
    ).to(device)
    signature_matrix = torch.load(args.signature_matrix, map_location=device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    return model, signature_matrix, device


def test_models(model, signature_matrix, device, args):
    # Load CSV file
    test_df = pd.read_csv(os.path.join(args.ground_truth, "test.csv"))

    results = evaluate_and_save_results(
        model,
        ext_test_gen,
        signature_matrix,
        device,
        os.path.join(args.output_dir, "ext_test_results.txt"),
        "ext_test_predictions.csv",
    )
    return results


def main(args):
    wandb.login(key=args.wandb_key)
    config = {
        "commit_id": args.commit_id,
        "num_experts": args.num_experts,
    }
    wandb.init(project="socialbit-ensemble", config=config)

    os.makedirs(args.output_dir, exist_ok=True)
    model, signature_matrix, device = initialize_model(args)
    results = test_models(model, signature_matrix, device, args)


def initialize_args(parser):
    parser.add_argument("--commit_id", type=str, help="Commit ID")
    parser.add_argument("--wandb_key", type=str, help="Wandb API key")

    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Path to the directory containing test dataset",
    )

    parser.add_argument(
        "--model_file",
        required=True,
        help="Path to the model file",
    )
    parser.add_argument(
        "--signature_matrix",
        required=True,
        help="Path to the signature matrix",
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
