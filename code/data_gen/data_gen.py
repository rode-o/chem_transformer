import os
import numpy as np
import pandas as pd
import h5py

# =========================================
# Parameters
# =========================================
num_chemicals = 3
num_concentrations = 10
num_experiments_per_concentration = 10
num_features = 75
num_freq_points = 100
start_freq = 0.0
stop_freq = 6e9
random_seed = 42

min_concentration = 10
max_concentration = 100
max_radius = 5.0
radius_jitter_scale = 0.5
feature_noise_scale = 0.05
cluster_center_scale = 1.0

np.random.seed(random_seed)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
synth_data_dir = os.path.join(project_root, "synth_data")
synth_data_h5_dir = os.path.join(project_root, "synth_data_h5")
os.makedirs(synth_data_dir, exist_ok=True)
os.makedirs(synth_data_h5_dir, exist_ok=True)

csv_path = os.path.join(synth_data_dir, "synthetic_dataset.csv")
h5_path = os.path.join(synth_data_h5_dir, "synthetic_dataset.h5")

concentration_levels = np.linspace(min_concentration, max_concentration, num_concentrations)
frequencies = np.linspace(start_freq, stop_freq, num_freq_points)

chemical_names = [f"Chemical_{i}" for i in range(num_chemicals)]

chemical_centers = {
    i: np.random.normal(loc=0, scale=cluster_center_scale, size=num_features)
    for i in range(num_chemicals)
}

data_rows = []
data_chemicals = []
data_concentrations = []
data_experiments = []
data_frequencies = []
data_features = []

for chem_id in range(num_chemicals):
    center = chemical_centers[chem_id]
    chem_name = chemical_names[chem_id]
    for concentration in concentration_levels:
        radius_target = ((max_concentration - concentration) / (max_concentration - min_concentration)) * max_radius
        for experiment_num in range(num_experiments_per_concentration):
            direction = np.random.normal(size=num_features)
            direction /= np.linalg.norm(direction)
            radius = np.abs(np.random.normal(loc=radius_target, scale=radius_jitter_scale))
            experiment_vector = center + direction * radius

            for freq in frequencies:
                features = experiment_vector + np.random.normal(loc=0, scale=feature_noise_scale, size=num_features)
                row = {
                    "Chemical": chem_name,
                    "Concentration": concentration,
                    "Experiment_Number": experiment_num,
                    "Frequency": freq
                }
                for i in range(num_features):
                    row[f"Feature_{i}"] = features[i]
                data_rows.append(row)

                data_chemicals.append(chem_id)
                data_concentrations.append(concentration)
                data_experiments.append(experiment_num)
                data_frequencies.append(freq)
                data_features.append(features)

df = pd.DataFrame(data_rows)
df.to_csv(csv_path, index=False)
print(f"Synthetic dataset (CSV) saved to {csv_path}")

data_chemicals = np.array(data_chemicals, dtype=np.int32)
data_concentrations = np.array(data_concentrations, dtype=np.float32)
data_experiments = np.array(data_experiments, dtype=np.int32)
data_frequencies = np.array(data_frequencies, dtype=np.float64)
data_features = np.array(data_features, dtype=np.float32)

with h5py.File(h5_path, "w") as f:
    f.create_dataset("Chemical", data=data_chemicals)
    f.create_dataset("Concentration", data=data_concentrations)
    f.create_dataset("Experiment_Number", data=data_experiments)
    f.create_dataset("Frequency", data=data_frequencies)
    f.create_dataset("Features", data=data_features)
    max_len = max(len(name) for name in chemical_names)
    dt = h5py.string_dtype(encoding='utf-8', length=max_len)
    f.create_dataset("All_Chemical_Names", (len(chemical_names),), dtype=dt, data=chemical_names)

print("Data generation complete. CSV and HDF5 created.")
