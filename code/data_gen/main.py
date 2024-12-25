from data_gen import generate_chemical_centers, generate_synthetic_data, save_to_h5
from ..config.config import Config
from visualize import visualize_data  # Import the visualization function
import numpy as np


def main():
    # Generate chemical feature centers
    chemical_centers = generate_chemical_centers()

    # Define concentration levels and frequencies
    concentration_levels = np.linspace(
        Config.MIN_CONCENTRATION, Config.MAX_CONCENTRATION, Config.NUM_CONCENTRATIONS
    )
    frequencies = np.linspace(Config.START_FREQ, Config.STOP_FREQ, Config.NUM_FREQ_POINTS)
    chemical_names = [f"Chemical_{i}" for i in range(Config.NUM_CHEMICALS)]

    # Generate synthetic data
    data_chemicals, data_concentrations, data_experiments, data_frequencies, data_features = generate_synthetic_data(
        chemical_centers, concentration_levels, frequencies
    )

    # Save the data to HDF5
    save_to_h5(data_chemicals, data_concentrations, data_experiments, data_frequencies, data_features, chemical_names)

    # Visualize the data
    visualize_data(Config.H5_PATH)  # Pass the HDF5 file path to the visualization function


if __name__ == "__main__":
    main()
