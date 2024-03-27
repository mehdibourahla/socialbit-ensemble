import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(ground_truth):

    train_data, test_data = train_test_split(
        ground_truth, test_size=0.3, stratify=ground_truth["is_social"]
    )
    validation_data, test_data = train_test_split(
        test_data, test_size=0.5, stratify=test_data["is_social"]
    )

    return train_data, validation_data, test_data


def analyze_data(dataset, dataset_name):
    print(f"Analyzing {dataset_name}")
    class_distribution = dataset["is_social"].value_counts()
    print(f"Class distribution in {dataset_name}:\n{class_distribution}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    root_dir = args.root_dir
    merged_train_data = pd.DataFrame()
    merged_validation_data = pd.DataFrame()
    merged_test_data = pd.DataFrame()
    for node in range(1, 9):
        if node == 5:
            continue
        node_dir = f"Node{node}"
        node_path = f"{root_dir}/{node_dir}/{node_dir}_new_annotations.csv"
        feature_path = f"{root_dir}/{node_dir}/YAMNet_v2"

        print(f"Loading ground truth for {node_dir} from {node_path}")

        ground_truth = pd.read_csv(node_path)
        ground_truth.rename(columns={"audio_file": "filename"}, inplace=True)
        ground_truth["filename"] = ground_truth["filename"].apply(
            lambda x: f"{feature_path}/{x.split('.')[0]}.mat"
        )
        ground_truth["dataset"] = node_dir
        ground_truth["is_social"] = ground_truth["activity"].apply(
            lambda x: 1 if x in ["visit", "calling"] else 0
        )
        analyze_data(ground_truth, f"Ground Truth for {node_dir}")

        train_data, validation_data, test_data = split_dataset(ground_truth)
        merged_train_data = pd.concat([merged_train_data, train_data])
        merged_validation_data = pd.concat([merged_validation_data, validation_data])
        merged_test_data = pd.concat([merged_test_data, test_data])

        analyze_data(train_data, f"Train Data for {node_dir}")
        analyze_data(validation_data, f"Validation Data for {node_dir}")
        analyze_data(test_data, f"Test Data for {node_dir}")

    merged_train_data.to_csv(f"{args.output_dir}/train_data.csv", index=False)
    merged_validation_data.to_csv(f"{args.output_dir}/validation_data.csv", index=False)
    merged_test_data.to_csv(f"{args.output_dir}/test_data.csv", index=False)


def initialize_args(parser):
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Path to the root directory containing the annotation files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    args = parser.parse_args()
    main(args)
