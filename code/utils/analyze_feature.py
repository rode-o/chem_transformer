import h5py
import numpy as np

def analyze_chemical_dataset(h5_file_path, dataset_name="Chemical", sample_size=10):
    """
    Analyze the Chemical dataset in an HDF5 file.

    Args:
        h5_file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to analyze (default is "Chemical").
        sample_size (int): Number of sample entries to print.

    Returns:
        None
    """
    try:
        # Open the HDF5 file
        print(f"Opening HDF5 file: {h5_file_path}")
        with h5py.File(h5_file_path, "r") as h5_file:
            # Check if the dataset exists
            if dataset_name not in h5_file:
                print(f"Dataset '{dataset_name}' not found in the HDF5 file.")
                return
            
            # Access the dataset
            dataset = h5_file[dataset_name]
            print(f"Dataset '{dataset_name}' loaded successfully.")

            # Dataset properties
            print(f"\nDataset Properties:")
            print(f"  Shape: {dataset.shape}")
            print(f"  Data Type: {dataset.dtype}")

            # Example data
            print(f"\nSample Data ({sample_size} entries):")
            for i in range(min(sample_size, len(dataset))):
                print(f"  Entry {i}: {dataset[i]}")

            # Unique values and statistics
            print(f"\nAnalyzing Unique Values and Statistics...")
            unique_values = np.unique(dataset)
            print(f"  Unique Values: {unique_values}")
            print(f"  Total Unique Values: {len(unique_values)}")

            # Check for any unexpected shapes or empty entries
            if len(dataset.shape) > 1:
                print(f"\nChecking entries for inconsistent shapes:")
                for i, entry in enumerate(dataset):
                    if isinstance(entry, np.ndarray) and entry.size == 0:
                        print(f"  Empty entry found at index {i}.")
                    elif isinstance(entry, (list, np.ndarray)) and len(entry) != dataset.shape[1]:
                        print(f"  Mismatched shape at index {i}: {entry}")

    except Exception as e:
        print(f"An error occurred while analyzing the dataset: {e}")

# Path to your HDF5 file
h5_file_path = r"C:\Users\RodePeters\Desktop\chem_transformer\synth_data_h5\synthetic_dataset.h5"

# Call the analysis function
analyze_chemical_dataset(h5_file_path)
