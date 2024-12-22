import logging
import psutil
import torch
import h5py
import os
from logger import setup_logger
from config import Config  # Import centralized configuration

# Initialize logger for this script
logger = setup_logger(name="utils", level=logging.INFO)


def ensure_folders_exist(folders):
    """
    Ensure that the required folders exist, creating them if necessary.

    Args:
        folders (list or tuple): List of folder paths to check and create.
    """
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Created missing folder: {folder}")
        else:
            logger.info(f"Folder exists: {folder}")


def calculate_padding(num_features, attention_resolution=Config.ATTENTION_RESOLUTION):
    """
    Calculate padding needed to make num_features divisible by attention resolution.
    """
    padding_needed = (attention_resolution - (num_features % attention_resolution)) % attention_resolution
    adjusted_features = num_features + padding_needed
    logger.info(f"Adjusted features: {adjusted_features} (with {padding_needed} padding)")
    return adjusted_features, padding_needed


def pad_features(features, adjusted_features):
    """
    Pad feature tensors to match the adjusted features size.
    """
    logger.debug(f"Features shape before padding: {features.shape}")
    if len(features.shape) < 2:
        raise ValueError(f"Features tensor is malformed: {features.shape}")
    if features.shape[1] < adjusted_features:
        padding_size = adjusted_features - features.shape[1]
        padded_features = torch.nn.functional.pad(features, (0, padding_size), mode="constant", value=0)
        logger.debug(f"Features shape after padding: {padded_features.shape}")
        return padded_features
    logger.debug("No padding needed.")
    return features


def detect_sweep_lengths(frequency_dataset):
    """
    Detect sweep lengths by identifying transitions from high to low frequencies.

    Args:
        frequency_dataset (numpy.ndarray): 1D array of concatenated frequencies.

    Returns:
        tuple: (list of sweep lengths, longest sweep length, boundaries of sweeps)
    """
    logger.info("Detecting sweep lengths...")
    try:
        boundaries = [0]  # Start with the first index
        for i in range(1, len(frequency_dataset)):
            if frequency_dataset[i] < frequency_dataset[i - 1]:  # Low-to-high transition detected (sweep reset)
                boundaries.append(i)

        # Append the end of the dataset as the last boundary
        boundaries.append(len(frequency_dataset))

        # Calculate sweep lengths
        sweep_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

        if len(sweep_lengths) == 0:
            raise ValueError("No sweeps detected. Check the Frequency dataset.")

        logger.info(f"Detected {len(sweep_lengths)} sweeps.")
        logger.info(f"Longest sweep length: {max(sweep_lengths)}")

        return sweep_lengths, max(sweep_lengths), boundaries

    except Exception as e:
        logger.error(f"Error detecting sweep lengths: {e}", exc_info=True)
        raise



def analyze_data(h5_file_path, attention_resolution=Config.ATTENTION_RESOLUTION, reserved_memory_gb=Config.RESERVED_MEMORY_GB):
    """
    Analyze the shape of the dataset, determine input parameters, and calculate batch size.
    """
    try:
        logger.info(f"Analyzing dataset at: {h5_file_path}")

        with h5py.File(h5_file_path, "r") as h5_file:
            # Analyze features
            num_features = h5_file["Features"].shape[1]

            # Determine unique chemical values
            unique_chemicals = len(set(h5_file["Chemical"][:]))
            logger.info(f"Number of unique chemicals (output_dim_chemical): {unique_chemicals}")

            # Detect sweep lengths
            frequency_dataset = h5_file["Frequency"][:]
            if len(frequency_dataset.shape) != 1:
                raise ValueError("Expected 'Frequency' dataset to be a 1D array of concatenated sweeps.")

            sweep_lengths, sequence_length, boundaries = detect_sweep_lengths(frequency_dataset)

            if len(set(sweep_lengths)) > 1:
                logger.warning(f"Variable sweep lengths detected: {set(sweep_lengths)}")

            # Calculate padding
            adjusted_features, padding_needed = calculate_padding(num_features, attention_resolution)

            # Estimate batch size based on memory
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / 1e9 - reserved_memory_gb
            feature_size = h5_file["Features"].dtype.itemsize
            sample_size = adjusted_features * feature_size
            batch_size = int((available_memory_gb * 1e9) // sample_size)

            # Log analysis details
            logger.info(f"Dataset Analysis:")
            logger.info(f"  Number of features: {num_features}")
            logger.info(f"  Adjusted features: {adjusted_features} (with {padding_needed} padding)")
            logger.info(f"  Sequence length: {sequence_length}")
            logger.info(f"  Total Memory: {memory.total / 1e9:.2f} GB")
            logger.info(f"  Available Memory: {available_memory_gb:.2f} GB")
            logger.info(f"  Estimated batch size: {batch_size}")

            return adjusted_features, padding_needed, num_features, sequence_length, batch_size, unique_chemicals

    except Exception as e:
        logger.error(f"Error analyzing data: {e}", exc_info=True)
        raise
