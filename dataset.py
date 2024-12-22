import logging
import torch
import h5py
from utils import pad_features
from config import Config
from logger import setup_logger

# Initialize logger for this script
logger = setup_logger(name=__name__, level=logging.INFO)


class HDF5Dataset:
    """
    Dataset class to handle HDF5 file loading and processing.
    """

    def __init__(self, h5_file_path, adjusted_features, pad_function, sequence_length):
        """
        Initialize the HDF5 dataset loader.

        Args:
            h5_file_path (str): Path to the HDF5 file.
            adjusted_features (int): Target feature size after padding.
            pad_function (function): Function to pad features to the target size.
            sequence_length (int): Length of each sweep in the dataset.
        """
        try:
            self.h5_file = h5py.File(h5_file_path, "r")

            # Validate required datasets
            required_datasets = ["Features", "Chemical", "Concentration", "Experiment_Number", "Frequency"]
            for dataset in required_datasets:
                if dataset not in self.h5_file:
                    raise KeyError(f"Required dataset '{dataset}' not found in HDF5 file.")

            self.features = self.h5_file["Features"]
            self.chemical = self.h5_file["Chemical"]
            self.concentration = self.h5_file["Concentration"]
            self.experiment_number = self.h5_file["Experiment_Number"]
            self.frequency = self.h5_file["Frequency"]

            self.adjusted_features = adjusted_features
            self.pad_function = pad_function
            self.sequence_length = sequence_length

            # Validate that data length is divisible by sequence length
            total_length = self.features.shape[0]
            if total_length % self.sequence_length != 0:
                raise ValueError(
                    f"Dataset length ({total_length}) is not evenly divisible by sequence length ({sequence_length})."
                )
            self.num_sweeps = total_length // self.sequence_length

            logger.info(f"Dataset loaded with {self.num_sweeps} sweeps and sequence length {self.sequence_length}.")

        except Exception as e:
            logger.error(f"Error initializing HDF5Dataset: {e}", exc_info=True)
            raise

    def __len__(self):
        """
        Get the total number of sweeps in the dataset.
        """
        return self.num_sweeps

    def __getitem__(self, idx):
        """
        Retrieve a single sweep from the dataset and pad its features.

        Args:
            idx (int): Index of the sweep to retrieve.

        Returns:
            dict: A dictionary containing the padded features and other attributes.
        """
        try:
            logger.debug(f"Retrieving sweep at index {idx}")

            # Compute start and end indices for the sweep
            start_idx = idx * self.sequence_length
            end_idx = start_idx + self.sequence_length

            # Retrieve original features for the sweep
            original_features = torch.tensor(
                self.features[start_idx:end_idx], dtype=Config.TENSOR_FLOAT_TYPE
            )

            # Pad the features if necessary
            padded_features = self.pad_function(original_features, self.adjusted_features)

            # Retrieve corresponding metadata
            chemical = torch.tensor(self.chemical[idx], dtype=Config.TENSOR_LONG_TYPE)
            concentration = torch.tensor(self.concentration[idx], dtype=Config.TENSOR_FLOAT_TYPE)
            experiment_number = torch.tensor(self.experiment_number[idx], dtype=Config.TENSOR_LONG_TYPE)
            frequency = torch.tensor(self.frequency[start_idx:end_idx], dtype=Config.TENSOR_FLOAT_TYPE)

            logger.debug(f"Successfully retrieved sweep at index {idx}")
            return {
                "features": padded_features,
                "chemical": chemical,
                "concentration": concentration,
                "experiment_number": experiment_number,
                "frequency": frequency,
            }

        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {e}", exc_info=True)
            raise

    def close(self):
        """
        Close the HDF5 file.
        """
        try:
            if hasattr(self, "h5_file") and self.h5_file:
                self.h5_file.close()
                logger.info("HDF5 file closed.")
        except Exception as e:
            logger.error(f"Error closing HDF5 file: {e}", exc_info=True)
