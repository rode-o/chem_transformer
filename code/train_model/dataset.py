import logging
import torch
import h5py
from utils import pad_features
from logger import setup_logger

# Initialize logger for this script
logger = setup_logger(name=__name__, level=logging.DEBUG)


class HDF5Dataset:
    """
    Dataset class to handle HDF5 file loading and processing.
    """

    def __init__(self, h5_file_path, adjusted_features, pad_function, sequence_length, num_chemicals):
        """
        Initialize the HDF5 dataset loader.

        Args:
            h5_file_path (str): Path to the HDF5 file.
            adjusted_features (int): Target feature size after padding.
            pad_function (function): Function to pad features to the target size.
            sequence_length (int): Length of each sweep in the dataset.
            num_chemicals (int): Number of unique chemicals in the dataset.
        """
        try:
            logger.info(f"Initializing HDF5Dataset with file: {h5_file_path}")
            self.h5_file = h5py.File(h5_file_path, "r")

            # Validate required datasets
            required_datasets = ["Features", "Chemical", "Concentration", "Experiment_Number", "Frequency"]
            for dataset in required_datasets:
                if dataset not in self.h5_file:
                    logger.error(f"Missing dataset: {dataset}")
                    raise KeyError(f"Required dataset '{dataset}' not found in HDF5 file.")

            self.features = self.h5_file["Features"]
            self.chemical = self.h5_file["Chemical"]
            self.concentration = self.h5_file["Concentration"]
            self.experiment_number = self.h5_file["Experiment_Number"]
            self.frequency = self.h5_file["Frequency"]

            self.adjusted_features = adjusted_features
            self.pad_function = pad_function
            self.sequence_length = sequence_length
            self.num_chemicals = num_chemicals

            # Validate dataset dimensions
            total_length = self.features.shape[0]
            if total_length % self.sequence_length != 0:
                logger.error(
                    f"Dataset length {total_length} not divisible by sequence length {sequence_length}."
                )
                raise ValueError(
                    f"Dataset length ({total_length}) is not evenly divisible by sequence length ({sequence_length})."
                )
            self.num_sweeps = total_length // self.sequence_length

            logger.info(
                f"Dataset initialized: {self.num_sweeps} sweeps, sequence length {self.sequence_length}, "
                f"num chemicals {self.num_chemicals}."
            )

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
            logger.debug(f"Retrieving data for index {idx}...")

            # Compute indices for slicing
            start_idx = idx * self.sequence_length
            end_idx = start_idx + self.sequence_length
            logger.debug(f"Start index: {start_idx}, End index: {end_idx}")

            # Retrieve features
            features = torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32)
            logger.debug(f"Features shape: {features.shape}")

            # Pad features if necessary
            padded_features = self.pad_function(features, self.adjusted_features)
            logger.debug(f"Padded features shape: {padded_features.shape}")

            # Retrieve chemical indices and convert to multi-hot
            chemical_indices = torch.tensor(self.chemical[idx], dtype=torch.int64)
            logger.debug(f"Chemical indices: {chemical_indices} Shape: {chemical_indices.shape}")

            # Perform one-hot encoding dynamically
            if chemical_indices.dim() == 0:
                chemical_indices = chemical_indices.unsqueeze(0)

            chemical = torch.nn.functional.one_hot(chemical_indices, num_classes=self.num_chemicals).sum(dim=0).float()
            logger.debug(f"Chemical (multi-hot): {chemical} Shape: {chemical.shape}")

            # Retrieve concentration
            concentration = torch.tensor(self.concentration[idx], dtype=torch.float32)
            logger.debug(f"Concentration: {concentration} Shape: {concentration.shape}")

            # Retrieve experiment number
            experiment_number = torch.tensor(self.experiment_number[idx], dtype=torch.int64)
            logger.debug(f"Experiment number: {experiment_number}")

            # Retrieve frequency data
            frequency = torch.tensor(self.frequency[start_idx:end_idx], dtype=torch.float32)
            logger.debug(f"Frequency shape: {frequency.shape}")

            # Validate tensor shapes
            assert padded_features.shape[1] == self.adjusted_features, (
                f"Padded features shape mismatch: {padded_features.shape}"
            )
            assert chemical.shape[0] == self.num_chemicals, (
                f"Chemical multi-hot vector shape mismatch: {chemical.shape}"
            )
            assert len(concentration.shape) == 1, (
                f"Concentration vector should be 1D but got: {concentration.shape}"
            )

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
            if self.h5_file:
                self.h5_file.close()
                logger.info("HDF5 file closed successfully.")
        except Exception as e:
            logger.error(f"Error closing HDF5 file: {e}", exc_info=True)
