import numpy as np
import h5py
from ..config.config import Config

def generate_chemical_centers():
    """Generate random chemical feature centers."""
    return {
        i: np.random.normal(
            loc=0, scale=Config.CLUSTER_CENTER_SCALE, size=Config.NUM_FEATURES
        ).astype(Config.NUMERIC_TYPE)
        for i in range(Config.NUM_CHEMICALS)
    }


def generate_synthetic_data(chemical_centers, concentration_levels, frequencies):
    """Generate synthetic dataset."""
    data_chemicals, data_concentrations = [], []
    data_experiments, data_frequencies = [], []
    data_features = []

    for chem_id in range(Config.NUM_CHEMICALS):
        center = chemical_centers[chem_id]
        for concentration in concentration_levels:
            radius_target = (
                (Config.MAX_CONCENTRATION - concentration)
                / (Config.MAX_CONCENTRATION - Config.MIN_CONCENTRATION)
            ) * Config.MAX_RADIUS
            for experiment_num in range(Config.NUM_EXPERIMENTS_PER_CONCENTRATION):
                direction = np.random.normal(size=Config.NUM_FEATURES).astype(Config.NUMERIC_TYPE)
                direction /= np.linalg.norm(direction)
                radius = np.abs(
                    np.random.normal(
                        loc=radius_target, scale=Config.RADIUS_JITTER_SCALE
                    )
                )
                experiment_vector = center + direction * radius

                for freq in frequencies:
                    features = experiment_vector + np.random.normal(
                        loc=0, scale=Config.FEATURE_NOISE_SCALE, size=Config.NUM_FEATURES
                    )
                    data_chemicals.append(chem_id)
                    data_concentrations.append(concentration)
                    data_experiments.append(experiment_num)
                    data_frequencies.append(freq)
                    data_features.append(features.astype(Config.NUMERIC_TYPE))

    return data_chemicals, data_concentrations, data_experiments, data_frequencies, data_features


def save_to_h5(data_chemicals, data_concentrations, data_experiments, data_frequencies, data_features, chemical_names):
    """Save the synthetic data to an HDF5 file."""
    with h5py.File(Config.H5_PATH, "w") as f:
        f.create_dataset("Chemical", data=np.array(data_chemicals, dtype=Config.INT_TYPE))
        f.create_dataset("Concentration", data=np.array(data_concentrations, dtype=Config.NUMERIC_TYPE))
        f.create_dataset("Experiment_Number", data=np.array(data_experiments, dtype=Config.INT_TYPE))
        f.create_dataset("Frequency", data=np.array(data_frequencies, dtype=Config.NUMERIC_TYPE))
        f.create_dataset("Features", data=np.array(data_features, dtype=Config.NUMERIC_TYPE))
        max_len = max(len(name) for name in chemical_names)
        dt = h5py.string_dtype(encoding="utf-8", length=max_len)
        f.create_dataset("All_Chemical_Names", (len(chemical_names),), dtype=dt, data=chemical_names)
    print(f"Synthetic dataset (HDF5) saved to {Config.H5_PATH}")
