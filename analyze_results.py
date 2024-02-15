import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def analyze_test_results(base_output_dir, num_folds=5, num_subfolds=5):
    results = []

    # Iterate over each fold and subfold
    for i_fold in range(num_folds):
        for j_subfold in range(num_subfolds):
            # Construct the path to the test_results.txt file
            result_file = os.path.join(
                base_output_dir,
                f"fold_{i_fold + 1}",
                f"subfold_{j_subfold + 1}",
                "test_results.txt",
            )

            # Read the results if the file exists
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    lines = f.readlines()
                    result = {
                        line.split(":")[0].strip(): float(line.split(":")[1].strip())
                        for line in lines
                    }
                    result["fold"] = i_fold + 1
                    result["subfold"] = j_subfold + 1
                    results.append(result)

    # Convert the results into a DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Calculate average metrics
    avg_test_loss = df["Test Loss"].mean()
    std_test_loss = df["Test Loss"].std()
    avg_test_accuracy = df["Test Accuracy"].mean()
    std_test_accuracy = df["Test Accuracy"].std()
    avg_sensitivity = df["Sensitivity"].mean()
    std_sensitivity = df["Sensitivity"].std()
    avg_specificity = df["Specificity"].mean()
    std_specificity = df["Specificity"].std()

    print("Average Test Loss:", avg_test_loss, "Std Dev:", std_test_loss)
    print("Average Test Accuracy:", avg_test_accuracy, "Std Dev:", std_test_accuracy)
    print("Average Sensitivity:", avg_sensitivity, "Std Dev:", std_sensitivity)
    print("Average Specificity:", avg_specificity, "Std Dev:", std_specificity)

    with open(os.path.join(base_output_dir, "avg_test_results.txt"), "w") as f:
        f.write(f"Test Loss: {avg_test_loss} ± {std_test_loss}\n")
        f.write(f"Test Accuracy: {avg_test_accuracy} ± {std_test_accuracy}\n")
        f.write(f"Sensitivity: {avg_sensitivity} ± {std_sensitivity}\n")
        f.write(f"Specificity: {avg_specificity} ± {std_specificity}\n")

    # Optionally, plot the metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(df["fold"], df["Test Loss"], "o-")
    axs[0, 0].set_title("Test Loss by Fold")
    axs[0, 1].plot(df["fold"], df["Test Accuracy"], "o-")
    axs[0, 1].set_title("Test Accuracy by Fold")
    axs[1, 0].plot(df["fold"], df["Sensitivity"], "o-")
    axs[1, 0].set_title("Sensitivity by Fold")
    axs[1, 1].plot(df["fold"], df["Specificity"], "o-")
    axs[1, 1].set_title("Specificity by Fold")

    for ax in axs.flat:
        ax.set(xlabel="Fold", ylabel="Metric Value")
        ax.label_outer()

    plt.tight_layout()
    output_path = os.path.join(base_output_dir, "results.png")
    plt.savefig(f"{output_path}")


def find_and_save_best_model(
    base_output_dir, num_folds=5, num_subfolds=5, metric="Test Accuracy"
):
    results = []
    best_metric_value = float("-inf") if metric != "Test Loss" else float("inf")
    best_model_info = None

    # Iterate over each fold and subfold
    for i_fold in range(num_folds):
        for j_subfold in range(num_subfolds):
            result_file = os.path.join(
                base_output_dir,
                f"fold_{i_fold + 1}",
                f"subfold_{j_subfold + 1}",
                "test_results.txt",
            )

            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    lines = f.readlines()
                    result = {
                        line.split(":")[0].strip(): float(line.split(":")[1].strip())
                        for line in lines
                    }
                    result["fold"] = i_fold + 1
                    result["subfold"] = j_subfold + 1
                    results.append(result)

                    # Check if this model is the best so far
                    current_metric_value = result[metric]
                    if (
                        metric != "Test Loss"
                        and current_metric_value > best_metric_value
                    ) or (
                        metric == "Test Loss"
                        and current_metric_value < best_metric_value
                    ):
                        best_metric_value = current_metric_value
                        best_model_info = (i_fold + 1, j_subfold + 1)

    if best_model_info:
        best_model_fold, best_model_subfold = best_model_info
        best_model_path = os.path.join(
            base_output_dir,
            f"fold_{best_model_fold}",
            f"subfold_{best_model_subfold}",
            "model.pth",
        )
        print(
            f"The best model is from fold {best_model_fold}, subfold {best_model_subfold} with {metric}: {best_metric_value}"
        )
        print(f"Best model path: {best_model_path}")

        # Optionally, copy the best model to a common directory
        import shutil

        destination_path = os.path.join(base_output_dir, "best_model.pth")
        shutil.copy(best_model_path, destination_path)
        print(f"Best model copied to: {destination_path}")


def main(args):
    analyze_test_results(args.output_dir)
    find_and_save_best_model(args.output_dir, metric="Test Accuracy")


def initialize_args(parser):
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())