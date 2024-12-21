from logger import setup_logger
import logging
import torch
import h5py
from utils import pad_features
from config import Config

# Initialize logger for this script
logger = setup_logger(name=__name__, level=logging.INFO)

class HDF5Dataset:
    """
    Dataset class to handle HDF5 file loading and processing.
    """

    def __init__(self, h5_file_path, adjusted_features, pad_function):
        """
        Initialize the HDF5 dataset loader.
        Args:
            h5_file_path (str): Path to the HDF5 file.
            adjusted_features (int): Target feature size after padding.
            pad_function (function): Function to pad features to the target size.
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

            logger.info(f"Dataset loaded with {len(self.features)} samples.")

        except Exception as e:
            logger.error(f"Error initializing HDF5Dataset: {e}", exc_info=True)
            raise

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return self.features.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset and pad its features.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the padded features and other attributes.
        """
        try:
            logger.debug(f"Retrieving sample at index {idx}")

            # Retrieve original features
            original_features = torch.tensor(self.features[idx], dtype=Config.TENSOR_FLOAT_TYPE)

            # Ensure features have at least 2 dimensions
            if original_features.ndimension() == 1:
                logger.debug(f"Original features at index {idx} have only one dimension, adding sequence dimension.")
                original_features = original_features.unsqueeze(0)  # Add a sequence dimension

            # Pad the features
            padded_features = self.pad_function(original_features, self.adjusted_features)

            # Return the padded features and other attributes
            logger.debug(f"Successfully retrieved and padded sample at index {idx}")
            return {
                "features": padded_features,
                "chemical": torch.tensor(self.chemical[idx], dtype=Config.TENSOR_LONG_TYPE),
                "concentration": torch.tensor(self.concentration[idx], dtype=Config.TENSOR_FLOAT_TYPE),
                "experiment_number": torch.tensor(self.experiment_number[idx], dtype=Config.TENSOR_LONG_TYPE),
                "frequency": torch.tensor(self.frequency[idx], dtype=Config.TENSOR_FLOAT_TYPE),
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
