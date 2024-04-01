import pandas as pd
import argparse
import os


def normalize_columns(data, columns):
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
        data[column] = data[column].astype(int)
    return data


def process_dataset(df, args, dataset_name):
    data_dir = args.features_dse if dataset_name == "dse" else args.features_aging
    num_clusters = args.num_clusters

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
    df["is_social"] = ((df["talk"] == 1)).astype(int)

    # Add the dataset name
    df["dataset"] = dataset_name

    # Split the dataset into num_clusters
    samples_per_participant = df.groupby("participant_id").size()
    participants_sorted = samples_per_participant.sort_values(ascending=False)
    splits = {i: [] for i in range(num_clusters)}  # Track participants per split
    samples_count_per_split = {
        i: 0 for i in range(num_clusters)
    }  # Track samples per split

    for participant_id, sample_count in participants_sorted.items():
        # Find the split with the minimum number of samples
        min_samples_split = min(
            samples_count_per_split, key=samples_count_per_split.get
        )

        # Allocate this participant to that split
        splits[min_samples_split].append(participant_id)
        samples_count_per_split[min_samples_split] += sample_count

    split_datasets = {}
    for i in range(num_clusters):
        split_datasets[i] = df[df["participant_id"].isin(splits[i])]

    final_datasets = {}
    for i in range(num_clusters):
        final_datasets[i] = split_datasets[i]
        final_datasets[i]["dataset"] = final_datasets[i]["dataset"] + "_" + str(i)

    # Merge the splits
    final_dataset = merge_datasets(list(final_datasets.values()))

    # Update the participant_id with the dataset name as prefix
    final_dataset["participant_id"] = (
        final_dataset["dataset"] + "_" + final_dataset["participant_id"].astype(str)
    )

    return final_dataset


def merge_datasets(dfs):
    dfs = pd.concat(dfs)
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


def main(args):
    args.output_dir = f"{args.output_dir}_{args.num_clusters}"
    os.makedirs(args.output_dir, exist_ok=True)
    # Load the ground truth dataset
    gt_dse = pd.read_csv(args.gt_dse)
    gt_aging = pd.read_csv(args.gt_aging)

    # Process the dataset
    df_dse = process_dataset(gt_dse, args, "dse")
    df_aging = process_dataset(gt_aging, args, "aging")

    # Merge the datasets
    df = merge_datasets([df_dse, df_aging])
    df.set_index("filename", inplace=True)

    # Analyze the dataset
    analyze_dataset(df)

    # Save the dataset
    df.to_csv(os.path.join(args.output_dir, "dse_aging.csv"))


def initialize_args(parser):
    parser.add_argument(
        "--num_clusters",
        default=2,
        type=int,
        help="Number of clusters to split the data into",
    )
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
