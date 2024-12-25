import logging
import torch
import numpy as np

# Initialize logger for preprocessing
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)


def one_hot_encode(chemical_indices, num_chemicals):
    """
    Perform one-hot encoding for chemical indices.

    Args:
        chemical_indices (torch.Tensor): Indices of chemicals.
        num_chemicals (int): Number of unique chemicals.

    Returns:
        torch.Tensor: Multi-hot encoded vector.
    """
    if isinstance(chemical_indices, (list, np.ndarray)):
        chemical_indices = torch.tensor(chemical_indices, dtype=torch.int64)
    if chemical_indices.dim() == 0:
        chemical_indices = chemical_indices.unsqueeze(0)
    encoded = torch.nn.functional.one_hot(chemical_indices, num_classes=num_chemicals).sum(dim=0).float()
    logger.info(f"One-hot encoding complete. Shape: {encoded.shape}, Values: {encoded.tolist()}")
    return encoded


def min_max_scale(data, min_val=None, max_val=None):
    """
    Apply min-max scaling to normalize data between 0 and 1.

    Args:
        data (torch.Tensor or np.ndarray): Input data.
        min_val (float): Minimum value for scaling. If None, inferred from data.
        max_val (float): Maximum value for scaling. If None, inferred from data.

    Returns:
        torch.Tensor: Min-max scaled data.
    """
    data = torch.tensor(data, dtype=torch.float32)
    if min_val is None:
        min_val = data.min().item()
    if max_val is None:
        max_val = data.max().item()
    scaled = (data - min_val) / (max_val - min_val)
    logger.info(f"Min-max scaling complete. Min: {min_val}, Max: {max_val}, Shape: {scaled.shape}")
    return scaled


def log_scale(data, epsilon=1e-6):
    """
    Apply log scaling to compress data range.

    Args:
        data (torch.Tensor or np.ndarray): Input data.
        epsilon (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: Log-scaled data.
    """
    data = torch.tensor(data, dtype=torch.float32)
    scaled = torch.log1p(data + epsilon)
    logger.info(f"Log scaling complete. Shape: {scaled.shape}, Sample Values: {scaled[:5].tolist()}")
    return scaled


def preprocess_data(raw_data, num_chemicals):
    """
    Preprocess the raw data by applying encoding and scaling.

    Args:
        raw_data (dict): Raw data with keys 'features', 'chemical', 'concentration', 'frequency'.
        num_chemicals (int): Number of unique chemicals.

    Returns:
        dict: Preprocessed data.
    """
    logger.info("Starting preprocessing...")

    # One-hot encode chemical data
    chemical = one_hot_encode(raw_data["chemical"], num_chemicals)

    # Scale concentration data (Min-Max scaling)
    concentration = min_max_scale(raw_data["concentration"])
    logger.info(f"Concentration scaling complete. Shape: {concentration.shape}")

    # Log-scale frequency data
    frequency = log_scale(raw_data["frequency"])
    logger.info(f"Frequency log scaling complete. Shape: {frequency.shape}")

    # Features remain as-is or can be normalized/scaled further
    features = raw_data["features"]
    logger.info(f"Features retained. Shape: {features.shape}")

    logger.info("Preprocessing complete.")
    return {
        "features": features,
        "chemical": chemical,
        "concentration": concentration,
        "frequency": frequency,
    }
