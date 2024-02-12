import os
import argparse
from data_loader import YAMNetFeaturesDatasetEAR
from model import MasterModel
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import logging


def plot_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    output_dir,
):
    output_path = os.path.join(output_dir, "test_results.txt")
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

    # Load Training, Validation and Test datasets
    train_data, val_data, test_data = load_csv_files(args.data_dir)

    print(train_data.head())

    # Generate Data Loaders for Training, Validation and Test datasets
    train_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(train_data), batch_size=32, shuffle=True
    )
    val_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(val_data), batch_size=32, shuffle=True
    )
    test_gen = DataLoader(
        YAMNetFeaturesDatasetEAR(test_data), batch_size=32, shuffle=True
    )

    # Configure the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MasterModel(num_experts=2, num_classes=2).to(device)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = model.train_model(
        train_gen, val_gen, device, args.output_dir, epochs=100
    )

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pth"))

    # Evaluate the model on the test set
    test_loss, test_accuracy, sensitivity, specificity = model.evaluate_model(
        test_gen, device
    )

    # Save the test results
    print("Saving the test results...")
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Sensitivity: {sensitivity}\n")
        f.write(f"Specificity: {specificity}\n")

    # Plot the training and validation loss and accuracy
    print("Plotting the training and validation loss and accuracy...")
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, args.output_dir
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    print("Starting the training process...")
    main(parser.parse_args())
