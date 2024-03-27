import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.nn.functional import one_hot
import os


class YAMNetFeaturesSINS(Dataset):
    # Dataset for the SINS dataset
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.filenames = dataframe.filename.tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        fpath = self.filenames[idx]
        label = self.dataframe.iloc[idx]["is_social"]
        domain = self.dataframe.iloc[idx]["dataset"]

        try:
            data = loadmat(fpath)["yamnet_top"]
            data = data[:, :60]
            if data.shape[1] < 60:
                padding = np.zeros((data.shape[0], 60 - data.shape[1]))
                data = np.concatenate((data, padding), axis=1)
            data_tensor = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            data_tensor = torch.tensor([], dtype=torch.float32)

        label_tensor = torch.tensor([label])
        domain_tensor = torch.tensor(domain, dtype=torch.int)

        label_one_hot = one_hot(label_tensor, num_classes=2).squeeze()

        return fpath, data_tensor, label_one_hot, domain_tensor


class YAMNetFeaturesDatasetDavid(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: str,
        seq_len: int = 30,
    ):
        """
        Initializes the dataset.

        :param dataframe: A pandas DataFrame indexed by filenames with columns ['is_social', 'dataset'].
        """
        self.dataframe = dataframe
        self.data_dir = data_dir
        # Ensure the DataFrame index is a list for easy access
        self.filenames = dataframe.filename.tolist()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve the filename using the DataFrame's index
        fpath = os.path.join(self.data_dir, self.filenames[idx])
        # Retrieve the label and domain directly from the DataFrame
        label = self.dataframe.iloc[idx]["is_social"]
        domain = self.dataframe.iloc[idx]["dataset"]

        try:
            # Load the feature data from the file
            data = loadmat(fpath)["yamnet_top"]
            # Limit the number of features to 30
            data = data[:, : self.seq_len]
            if data.shape[1] < self.seq_len:
                # If the segment is shorter than 30 seconds, pad the rest with zeros
                padding = np.zeros((data.shape[0], self.seq_len - data.shape[1]))
                data = np.concatenate((data, padding), axis=1)
            # Convert the loaded data to a tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            # In case of an error, return None or consider an alternative handling method
            data_tensor = torch.tensor(
                [], dtype=torch.float32
            )  # Or use an empty tensor

        # Convert label and domain to tensors
        label_tensor = torch.tensor([label])
        domain_tensor = torch.tensor(domain, dtype=torch.int)

        label_one_hot = one_hot(label_tensor, num_classes=2).squeeze()

        return self.filenames[idx], data_tensor, label_one_hot, domain_tensor


class YAMNetFeaturesDatasetEAR(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the dataset.

        :param dataframe: A pandas DataFrame indexed by filenames with columns ['is_social', 'dataset'].
        """
        self.dataframe = dataframe
        # Ensure the DataFrame index is a list for easy access
        self.filenames = dataframe.filename.tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve the filename using the DataFrame's index
        fpath = self.filenames[idx]
        # Retrieve the label and domain directly from the DataFrame
        label = self.dataframe.iloc[idx]["is_social"]
        domain = self.dataframe.iloc[idx]["dataset"]

        try:
            # Load the feature data from the file
            data = loadmat(fpath)["yamnet_top"]
            # Limit the number of features to 30
            data = data[:, :30]
            if data.shape[1] < 30:
                # If the segment is shorter than 30 seconds, pad the rest with zeros
                padding = np.zeros((data.shape[0], 30 - data.shape[1]))
                data = np.concatenate((data, padding), axis=1)
            # Convert the loaded data to a tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            # In case of an error, return None or consider an alternative handling method
            data_tensor = torch.tensor(
                [], dtype=torch.float32
            )  # Or use an empty tensor

        # Convert label and domain to tensors
        label_tensor = torch.tensor([label])
        domain_tensor = torch.tensor(domain, dtype=torch.int)

        label_one_hot = one_hot(label_tensor, num_classes=2).squeeze()

        return fpath, data_tensor, label_one_hot, domain_tensor
