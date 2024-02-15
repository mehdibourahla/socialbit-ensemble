import os
import argparse
from data_loader import YAMNetFeaturesDatasetEAR, YAMNetFeaturesDatasetDavid
from sklearn.utils.class_weight import compute_class_weight
from models import MasterModel, BiLSTMModel
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np


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
    plt.savefig(f"{output_path}")


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
            num_experts=2,
            class_weights_tensor=class_weights_tensor,
            skip_connection=args.skip_connection,
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
    model, train_gen, val_gen, test_gen, device, output_dir, current_output_dir
):
    train_losses, val_losses, train_accuracies, val_accuracies = model.train_model(
        train_gen, val_gen, device, output_dir, epochs=100
    )
    torch.save(model.state_dict(), os.path.join(current_output_dir, "model.pth"))
    evaluate_and_save_results(
        model,
        test_gen,
        device,
        os.path.join(current_output_dir, "test_results.txt"),
        "test_predictions.csv",
    )
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, current_output_dir
    )


def evaluate_and_save_results(
    model, data_loader, device, results_file_path, predictions_file_name
):
    test_loss, test_accuracy, sensitivity, specificity, predictions = (
        model.evaluate_model(data_loader, device)
    )
    with open(results_file_path, "w") as f:
        f.write(
            f"Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\n"
        )
    pd.DataFrame(predictions).to_csv(
        os.path.join(os.path.dirname(results_file_path), predictions_file_name),
        index=False,
    )


def main(args):
    setup_directories(args.output_dir)

    current_output_dir = os.path.join(
        args.output_dir, f"fold_{args.i_fold + 1}", f"subfold_{args.j_subfold + 1}"
    )
    setup_directories(current_output_dir)
    setup_logging(current_output_dir)

    training_fold, validation_fold, test_fold = load_and_prepare_data(
        args.data_dir, args.i_fold, args.j_subfold
    )
    class_weights = compute_weights(training_fold)
    model, device = initialize_model(args, class_weights)
    train_gen, val_gen, test_gen = get_data_loaders(
        training_fold, validation_fold, test_fold
    )
    train_and_evaluate_model(
        model, train_gen, val_gen, test_gen, device, args.output_dir, current_output_dir
    )

    # Evaluate on external test data
    ext_test_df = pd.read_csv(os.path.join(args.ext_test_data_dir, "metadata.csv"))
    ext_test_gen = DataLoader(
        YAMNetFeaturesDatasetDavid(ext_test_df, args.ext_test_data_dir, domain=0),
        batch_size=32,
        shuffle=False,
    )
    evaluate_and_save_results(
        model,
        ext_test_gen,
        device,
        os.path.join(current_output_dir, "ext_test_results.txt"),
        "ext_test_predictions.csv",
    )


def initialize_args(parser):
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
    parser.add_argument(
        "--skip_connection", action="store_true", help="Use skip connection"
    )
    parser.add_argument("--baseline", action="store_true", help="Use Baseline model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
