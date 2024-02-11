import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch.nn.functional import one_hot


# class YAMNetFeaturesDataset(Dataset):
#     def __init__(self, features_dir, labels, domains, seconds_per_chunk=30):
#         """
#         :param features_dir: Directory where feature files are stored.
#         :param labels: A dictionary mapping filenames to their labels (0 or 1).
#         :param domains: A dictionary mapping filenames to their domain IDs (e.g., speaker IDs).
#         :param seconds_per_chunk: Number of seconds per chunk for segmentation.
#         """
#         self.features_dir = features_dir
#         self.labels = labels
#         self.domains = domains
#         self.seconds_per_chunk = seconds_per_chunk
#         self.segments = self._prepare_segments()

#     def _prepare_segments(self):
#         segments = []
#         for fname in os.listdir(self.features_dir):
#             fpath = os.path.join(self.features_dir, fname)
#             label = self.labels[fname]
#             domain = self.domains[fname]  # Domain ID for the current file
#             data = loadmat(fpath)["yamnet_top"]
#             num_seconds = data.shape[1]
#             num_segments = np.ceil(num_seconds / self.seconds_per_chunk).astype(int)
#             for segment_idx in range(num_segments):
#                 segments.append((fname, segment_idx, label, domain))
#         return segments

#     def strings_to_categorical_tensor(self, strings):
#         categorical = pd.Categorical(strings)
#         codes = categorical.codes
#         tensor_labels = torch.tensor(codes, dtype=torch.int)

#         return tensor_labels

#     def __len__(self):
#         return len(self.segments)

#     def __getitem__(self, idx):
#         fname, segment_idx, label, domain = self.segments[idx]
#         fpath = os.path.join(self.features_dir, fname)
#         data = loadmat(fpath)["yamnet_top"]
#         start_second = segment_idx * self.seconds_per_chunk
#         end_second = min(data.shape[1], start_second + self.seconds_per_chunk)
#         segment = data[:, start_second:end_second]

#         # If the segment is shorter than 30 seconds, pad the rest with zeros
#         if segment.shape[1] < self.seconds_per_chunk:
#             padding = np.zeros(
#                 (data.shape[0], self.seconds_per_chunk - segment.shape[1])
#             )
#             segment = np.concatenate((segment, padding), axis=1)

#         label_one_hot = one_hot(torch.tensor(label), num_classes=2).squeeze()

#         return torch.tensor(segment, dtype=torch.float32), label_one_hot, domain


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

        # domain_tensor = self.strings_to_categorical_tensor(domain)

        # Convert label and domain to tensors
        label_tensor = torch.tensor([label])
        domain_tensor = torch.tensor(domain, dtype=torch.int)

        label_one_hot = one_hot(label_tensor, num_classes=2).squeeze()

        return data_tensor, label_one_hot, domain_tensor
