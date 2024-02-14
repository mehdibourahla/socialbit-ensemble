import os
import argparse
from data_loader import YAMNetFeaturesDatasetEAR
from sklearn.utils.class_weight import compute_class_weight
from model import MasterModel
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    i_fold = args.i_fold
    j_subfold = args.j_subfold
    current_output_dir = os.path.join(
        args.output_dir, f"fold_{i_fold + 1}", f"subfold_{j_subfold + 1}"
    )
    os.makedirs(current_output_dir, exist_ok=True)

    # Load Training, Validation and Test datasets
    train_data, val_data, test_data = load_csv_files(args.data_dir)
    data = pd.concat([train_data, val_data, test_data])

    num_folds = 5
    fold_size = len(data) // num_folds
    folds = []
    for i in range(num_folds):
        folds.append(data[i * fold_size : (i + 1) * fold_size])

    test_fold = folds[i_fold]
    validation_fold = folds[j_subfold]
    training_fold = pd.concat(
        [folds[i] for i in range(num_folds) if i != i_fold and i != j_subfold]
    )

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(training_fold["is_social"]),
        y=training_fold["is_social"],
    )

    # Handle class imbalance in the training set

    # Generate Data Loaders for Training, Validation and Test datasets
    train_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(training_fold), batch_size=32, shuffle=True
    )
    val_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(validation_fold), batch_size=32, shuffle=True
    )
    test_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(test_fold), batch_size=32, shuffle=True
    )

    # Configure the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    model = MasterModel(
        num_experts=2,
        class_weights_tensor=class_weights_tensor,
        num_classes=2,
        skip_connection=args.skip_connection,
    ).to(device)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = model.train_model(
        train_gen, val_gen, device, args.output_dir, epochs=100
    )

    # Save the model
    torch.save(model.state_dict(), os.path.join(current_output_dir, "model.pth"))

    # Evaluate the model on the test set
    test_loss, test_accuracy, sensitivity, specificity = model.evaluate_model(
        test_gen, device
    )

    # Save the test results
    print("Saving the test results...")
    with open(os.path.join(current_output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"Specificity: {specificity}\n")

    # Plot the training and validation loss and accuracy
    print("Plotting the training and validation loss and accuracy...")
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, current_output_dir
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

    parser.add_argument("--i_fold", type=int, help="Fold number")
    parser.add_argument("--j_subfold", type=int, help="Subfold number")
    parser.add_argument(
        "--skip_connection", action="store_true", help="Use skip connection"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
