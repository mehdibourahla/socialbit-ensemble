# This script aims to create the Training, Validation and Test datasets from EAR dataset.
# The dataset is divided into 3 parts: Training (70%), Validation (15%) and Test (15%).
# Each dataset is first preprocessed to remove unwanted records.

import pandas as pd
import argparse
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_features(dataframe):
    filenames = dataframe["filename"].tolist()
    features = []
    for filename in filenames:
        data = loadmat(filename)["yamnet_top"]
        data = data[:, :30]
        if data.shape[1] < 30:
            padding = np.zeros((data.shape[0], 30 - data.shape[1]))
            data = np.concatenate((data, padding), axis=1)
        features.append(data)
    return filenames, np.stack(features)


def create_domains(dataframe, output_dir):
    filenames, data = load_features(dataframe)
    data_reshaped = data.reshape(-1, 1024 * 30)

    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(data_reshaped)

    # Apply PCA to reduce the dimensionality to 50
    pca = PCA(n_components=50, random_state=42)
    data_pca = pca.fit_transform(normalized_embeddings)

    # Apply KMeans to cluster the data into 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(data_pca)

    # Apply TSNE to reduce the dimensionality to 2 for visualization
    tsne_results = TSNE(
        n_components=2, random_state=42, learning_rate="auto", init="random"
    ).fit_transform(data_pca)

    # Visualize the clusters using a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=clusters, s=5, cmap="rainbow"
    )
    plt.title("t-SNE visualization of YAMNet embeddings after KMeans clustering")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter)  # Show color scale
    plt.savefig(os.path.join(output_dir, "tsne_clusters.png"))

    filename_cluster_map = {
        filename: cluster for filename, cluster in zip(filenames, clusters)
    }

    return filename_cluster_map


def normalize_columns(data, columns):
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
        data[column] = data[column].astype(int)
    return data


def process_dataset(df, data_dir, dataset_name, output_dir):
    df.columns = map(str.lower, df.columns)

    # Remove records that do not have a corresponding file in the data directory
    files_in_dir = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    df["filename"] = df["filename"].str.replace(".wav", ".mat")
    df = df[df["filename"].isin(files_in_dir)]

    # Add the full path to the file
    df["filename"] = df["filename"].apply(lambda x: os.path.join(data_dir, x))

    # Normalize the columns
    df = df.rename(columns={"id": "participant_id"})
    if "socializing/ entertaining" in df.columns:
        df = df.rename(columns={"socializing/ entertaining": "socent"})
    if "talk " in df.columns:
        df = df.rename(columns={"talk ": "talk"})
    if "w/ one person" in df.columns:
        df = df.rename(columns={"w/ one person": "withone"})
    if "w/ multiple people" in df.columns:
        df = df.rename(columns={"w/ multiple people": "withgroup"})
    df = normalize_columns(
        df, ["socent", "problems", "talk", "withone", "withgroup", "phone", "self"]
    )

    # Remove duplicates and records with problems
    df = df.drop_duplicates(subset="filename", keep="first")
    df = df[df["problems"] == 0]

    # Add label for social interaction "is_social"
    # Records with 'talk' == 1 and 'self' == 0 and ('w/ one person' == 1 or 'w/ multiple people' == 1 or phone == 1)
    df["is_social"] = (
        (df["talk"] == 1)
        & (df["self"] == 0)
        & ((df["withone"] == 1) | (df["withgroup"] == 1) | (df["phone"] == 1))
    ).astype(int)

    filename_cluster_map = create_domains(df, output_dir)
    # Add a new column to the dataset to indicate the dataset name
    df["dataset"] = df["filename"].map(filename_cluster_map)
    df["dataset"] = dataset_name + "_" + df["dataset"].astype(str)

    # Update the participant_id with the dataset name as prefix
    df["participant_id"] = dataset_name + "_" + df["participant_id"].astype(str)

    return df


def merge_datasets(dfs):
    dfs = pd.concat(dfs)
    dfs.set_index("filename", inplace=True)
    return dfs


def analyze_dataset(df):
    # Analyze the dataset
    print("Number of records:", len(df))
    print("Number of unique participants:", df["participant_id"].nunique())
    print("Number of records with social interaction:", df["is_social"].sum())
    print(
        "Number of records without social interaction:", len(df) - df["is_social"].sum()
    )
    print(
        "Number of records from each dataset:"
        + df["dataset"].value_counts().to_string()
    )


def split_dataset(df, test_size=0.3, random_state=42):
    # Split participants
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_val_idx = next(gss.split(df, groups=df["participant_id"]))

    train_data = df.iloc[train_idx]
    test_val_data = df.iloc[test_val_idx]

    # Further split test_val_data into validation and test sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(
        gss.split(test_val_data, groups=test_val_data["participant_id"])
    )

    validation_data = test_val_data.iloc[val_idx]
    test_data = test_val_data.iloc[test_idx]

    return train_data, validation_data, test_data


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # Load the ground truth dataset
    gt_dse = pd.read_csv(args.gt_dse)
    gt_aging = pd.read_csv(args.gt_aging)

    # Process the dataset
    df_dse = process_dataset(gt_dse, args.features_dse, "dse", args.output_dir)
    df_aging = process_dataset(gt_aging, args.features_aging, "aging", args.output_dir)

    # Merge the datasets
    df = merge_datasets([df_dse, df_aging])

    # Analyze the dataset
    analyze_dataset(df)

    # Split the dataset
    train_data, val_data, test_data = split_dataset(df)

    # Analyze each split
    print("Training dataset:")
    analyze_dataset(train_data)
    print("\nValidation dataset:")
    analyze_dataset(val_data)
    print("\nTest dataset:")
    analyze_dataset(test_data)

    # Save the datasets
    train_data.to_csv(os.path.join(args.output_dir, "train.csv"))
    val_data.to_csv(os.path.join(args.output_dir, "val.csv"))
    test_data.to_csv(os.path.join(args.output_dir, "test.csv"))


def initialize_args(parser):
    # Input paths
    parser.add_argument(
        "--features_dse",
        required=True,
        help="Path to the directory containing the data",
    )
    parser.add_argument(
        "--features_aging",
        required=True,
        help="Path to the directory containing the data",
    )
    parser.add_argument(
        "--gt_dse", required=True, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--gt_aging", default=None, help="Path to the ground truth CSV file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to Output the results"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    initialize_args(parser)
    main(parser.parse_args())
