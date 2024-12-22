import h5py
import numpy as np
from logger import setup_logger
import os

# Initialize logger for this script
logger = setup_logger(name="inspect_frequency", level="INFO")

def inspect_frequency(h5_file_path):
    """
    Inspect and print details about the Frequency feature in the HDF5 dataset.

    Args:
        h5_file_path (str): Path to the HDF5 file.
    """
    if not os.path.exists(h5_file_path):
        logger.error(f"HDF5 file not found at: {h5_file_path}")
        return

    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            # Validate the existence of the Frequency dataset
            if "Frequency" not in h5_file:
                logger.error("'Frequency' dataset not found in the HDF5 file.")
                return

            frequency_dataset = h5_file["Frequency"][:]
            logger.info(f"Frequency dataset loaded with shape: {frequency_dataset.shape}")
            logger.info(f"Frequency dataset (first 300 values): {frequency_dataset[:300]}")

            # Identify sweep boundaries based on resets
            boundaries = [0]  # Start with the first index
            for i in range(1, len(frequency_dataset)):
                if frequency_dataset[i] < frequency_dataset[i - 1]:  # Reset detected
                    boundaries.append(i)

            # Append the end of the dataset as the last boundary
            boundaries.append(len(frequency_dataset))
            logger.info(f"Detected sweep boundaries: {boundaries[:20]} (showing first 20)")

            # Calculate sweep lengths
            sweep_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
            unique_lengths = set(sweep_lengths)

            logger.info(f"Detected sweep lengths: {sweep_lengths[:20]} (showing first 20)")
            logger.info(f"Unique sweep lengths: {unique_lengths}")
            logger.info(f"Longest sweep length: {max(sweep_lengths)}")
            logger.info(f"Shortest sweep length: {min(sweep_lengths)}")
            logger.info(f"Total number of sweeps: {len(sweep_lengths)}")

    except Exception as e:
        logger.error(f"Error inspecting frequency feature: {e}", exc_info=True)

if __name__ == "__main__":
    # Replace this path with the path to your HDF5 file
    h5_file_path = r"C:\Users\RodePeters\Desktop\chem_transformer\synth_data_h5\synthetic_dataset.h5"
    inspect_frequency(h5_file_path)
