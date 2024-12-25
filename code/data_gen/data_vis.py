import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Paths
h5_path = r"C:\Users\RodePeters\Desktop\chem_transformer\synth_data_h5\synthetic_dataset.h5"

# Load data from HDF5
def load_h5_data(h5_path):
    with h5py.File(h5_path, "r") as h5file:
        # Load metadata columns
        chemicals = np.array(h5file["Chemical"][:])
        concentrations = np.array(h5file["Concentration"][:])
        experiment_numbers = np.array(h5file["Experiment_Number"][:])
        frequencies = np.array(h5file["Frequency"][:])  # Not used in this visualization
        
        # Load features from the single "Features" dataset
        features = np.array(h5file["Features"][:])

    # Combine metadata into a single DataFrame
    df = pd.DataFrame({
        "Chemical": chemicals,
        "Concentration": concentrations,
        "Experiment_Number": experiment_numbers,
    })
    return df, features


# Load the dataset
metadata, X = load_h5_data(h5_path)

# Aggregate features for each experiment
metadata["Group_Key"] = metadata.apply(lambda row: (row["Chemical"], row["Concentration"], row["Experiment_Number"]), axis=1)

# Group by unique experiment combinations and compute the mean for features
unique_keys = metadata["Group_Key"].drop_duplicates()
aggregated_features = np.array([
    X[metadata["Group_Key"] == key].mean(axis=0)
    for key in unique_keys
])
aggregated_metadata = metadata.loc[unique_keys.index].reset_index(drop=True)

# Standardize features before PCA
scaler = StandardScaler()
aggregated_features_scaled = scaler.fit_transform(aggregated_features)

# Dimensionality Reduction to 3D (Using PCA)
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(aggregated_features_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
total_variance_explained = np.sum(explained_variance)

# Print variance outputs
print("Explained Variance Ratio for Each Component:")
print(f"PC1: {explained_variance[0]:.4f}")
print(f"PC2: {explained_variance[1]:.4f}")
print(f"PC3: {explained_variance[2]:.4f}")
print(f"Total Variance Explained by First 3 Components: {total_variance_explained:.4f}")

# Normalize Concentration for Gradient
concentration_min = aggregated_metadata["Concentration"].min()
concentration_max = aggregated_metadata["Concentration"].max()
norm = Normalize(vmin=concentration_min, vmax=concentration_max)

# Dynamically generate colormaps for all unique chemicals
unique_chemicals = aggregated_metadata["Chemical"].unique()
base_colormap_names = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]
chemical_colormaps = {
    chemical: plt.colormaps.get_cmap(base_colormap_names[i % len(base_colormap_names)])
    for i, chemical in enumerate(unique_chemicals)
}

# Map Colors Based on Chemical and Concentration (Ensure no completely white points)
def get_color(row):
    normalized_concentration = norm(row["Concentration"])
    clamped_concentration = max(0.2, normalized_concentration)  # Increase minimum saturation
    return chemical_colormaps[row["Chemical"]](clamped_concentration)

aggregated_metadata["Color"] = aggregated_metadata.apply(get_color, axis=1)

# Apply Dark Mode Theme
plt.style.use('dark_background')

# Create a 3D scatter plot with unified background color
fig = plt.figure(figsize=(12, 10), facecolor="#1a1a1a")  # Set figure background color
ax = fig.add_subplot(111, projection='3d', facecolor="#1a1a1a")  # Set axes background color

# Plot each chemical with its gradient-based color
for chemical in unique_chemicals:
    chem_mask = aggregated_metadata["Chemical"] == chemical
    ax.scatter(
        X_reduced[chem_mask, 0],
        X_reduced[chem_mask, 1],
        X_reduced[chem_mask, 2],
        c=aggregated_metadata.loc[chem_mask, "Color"],
        s=80,  # Fixed marker size
        alpha=0.8
    )

# Add fully saturated colors to the legend for chemicals
for i, chemical in enumerate(unique_chemicals):
    base_color = chemical_colormaps[chemical](1.0)  # Use the fully saturated base color
    ax.scatter([], [], color=base_color, label=chemical, s=80)  # Legend entry for chemical

# Customize Axes for Dark Mode
ax.grid(color="#444444")     # Dim grid lines
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.zaxis.label.set_color("white")
ax.tick_params(colors="white")  # White tick labels
ax.title.set_color("white")

# Add labels, title, and legend
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Visualization with Chemical Colors and Concentration Gradient')
ax.legend(title="Chemical", loc="upper left", labelcolor="white", facecolor="#2a2a2a", edgecolor="white")

# Show the plot
plt.show()
