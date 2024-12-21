import os
import numpy as np
import h5py

# =========================================
# Adjustable Parameters
# =========================================
num_files = 5  # How many inference data files to generate

# Set the number of chemicals by this single fixed variable
num_chemicals = 3  # e.g. Chemical_0, Chemical_1, Chemical_2

concentration_min = 10
concentration_max = 100

num_features = 75
num_freq_points = 100
start_freq = 0.0
stop_freq = 6e9
random_seed = 42

max_radius = 5.0
radius_jitter_scale = 0.5
feature_noise_scale = 0.05
cluster_center_scale = 1.0
experiment_number = 0

np.random.seed(random_seed)

# Set up directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
infer_data_dir = os.path.join(project_root, "infer_data")
os.makedirs(infer_data_dir, exist_ok=True)

frequencies = np.linspace(start_freq, stop_freq, num_freq_points)

# Generate a list of possible chemical names from the single `num_chemicals` variable
chemical_names_all = [f"Chemical_{i}" for i in range(num_chemicals)]

for _ in range(num_files):
    # Randomly select a chemical from the available chemicals
    chemical_id = np.random.randint(0, num_chemicals)
    chemical_name = chemical_names_all[chemical_id]

    # Random concentration
    concentration_value = np.random.uniform(concentration_min, concentration_max)

    # Create a filename based on chemical and concentration
    conc_int = int(concentration_value)  # to keep filenames neat
    file_name = f"{chemical_name}_Concentration_{conc_int}.h5"
    h5_path = os.path.join(infer_data_dir, file_name)

    # For the chosen chemical, generate one cluster center
    chemical_centers = {
        chemical_id: np.random.normal(loc=0, scale=cluster_center_scale, size=num_features)
    }
    center = chemical_centers[chemical_id]

    # Compute radius target
    radius_target = ((concentration_max - concentration_value) / 
                     (concentration_max - concentration_min)) * max_radius

    direction = np.random.normal(size=num_features)
    direction /= np.linalg.norm(direction)
    radius = np.abs(np.random.normal(loc=radius_target, scale=radius_jitter_scale))
    experiment_vector = center + direction * radius

    data_chemicals = []
    data_concentrations = []
    data_experiments = []
    data_frequencies = []
    data_features = []

    for freq in frequencies:
        features = experiment_vector + np.random.normal(loc=0, scale=feature_noise_scale, size=num_features)
        data_chemicals.append(chemical_id)
        data_concentrations.append(concentration_value)
        data_experiments.append(experiment_number)
        data_frequencies.append(freq)
        data_features.append(features)

    data_chemicals = np.array(data_chemicals, dtype=np.int32)
    data_concentrations = np.array(data_concentrations, dtype=np.float32)
    data_experiments = np.array(data_experiments, dtype=np.int32)
    data_frequencies = np.array(data_frequencies, dtype=np.float64)
    data_features = np.array(data_features, dtype=np.float32)

    # Store only this single chemical name used in this file
    chemical_names = [chemical_name]
    max_len = max(len(name) for name in chemical_names)
    dt = h5py.string_dtype(encoding='utf-8', length=max_len)

    # Save to HDF5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Chemical", data=data_chemicals)
        f.create_dataset("Concentration", data=data_concentrations)
        f.create_dataset("Experiment_Number", data=data_experiments)
        f.create_dataset("Frequency", data=data_frequencies)
        f.create_dataset("Features", data=data_features)
        f.create_dataset("Chemical_Names", (len(chemical_names),), dtype=dt, data=chemical_names)

    print("Single experiment inference data created at:", h5_path)
