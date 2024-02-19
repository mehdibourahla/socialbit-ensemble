import os
import pandas as pd
from models import MasterModel
from data_loader import YAMNetFeaturesDatasetDavid
import torch
from torch.utils.data import DataLoader
from train import evaluate_and_save_results
import argparse


def initialize_model(skip_connection, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MasterModel(
        num_experts=2,
        skip_connection=skip_connection,
    ).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    return model, device


def test_models(
    base_output_dir, ext_test_data_dir, skip_connection, num_folds=5, num_subfolds=5
):
    ext_test_df = pd.read_csv(os.path.join(ext_test_data_dir, "metadata.csv"))
    ext_test_gen = DataLoader(
        YAMNetFeaturesDatasetDavid(ext_test_df, ext_test_data_dir, domain=1),
        batch_size=32,
        shuffle=False,
    )
    for i_fold in range(num_folds):
        for j_subfold in range(num_subfolds):
            # Construct the path to the model.pth file
            model_file = os.path.join(
                base_output_dir,
                f"fold_{i_fold + 1}",
                f"subfold_{j_subfold + 1}",
                "model.pth",
            )

            current_output_dir = os.path.join(
                base_output_dir, f"fold_{i_fold + 1}", f"subfold_{j_subfold + 1}"
            )

            # Load the model and test it
            if os.path.exists(model_file):
                print(f"Testing model from fold {i_fold + 1}, subfold {j_subfold + 1}")
                # Load the model and test it
                model, device = initialize_model(skip_connection, model_file)
                evaluate_and_save_results(
                    model,
                    ext_test_gen,
                    device,
                    os.path.join(current_output_dir, "ext_test_results_domain_1.txt"),
                    "ext_test_predictions_domain_1.csv",
                )
            else:
                print(
                    f"Model from fold {i_fold + 1}, subfold {j_subfold + 1} not found"
                )


def main(args):
    test_models(
        args.output_dir, args.ext_test_data_dir, skip_connection=args.skip_connection
    )


def initialize_args(parser):
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )
    parser.add_argument(
        "--ext_test_data_dir",
        required=True,
        help="Path to the directory containing external test dataset",
    )

    parser.add_argument(
        "--skip_connection", action="store_true", help="Use skip connection"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
