import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ..config.config import Config

def load_h5_data(h5_path):
    """
    Load data from the HDF5 file.

    Args:
        h5_path (str or Path): Path to the HDF5 file.

    Returns:
        tuple: Metadata DataFrame and features array.
    """
    with h5py.File(h5_path, "r") as h5file:
        chemicals = np.array(h5file["Chemical"][:], dtype=Config.INT_TYPE)
        concentrations = np.array(h5file["Concentration"][:], dtype=Config.NUMERIC_TYPE)
        experiment_numbers = np.array(h5file["Experiment_Number"][:], dtype=Config.INT_TYPE)
        features = np.array(h5file["Features"][:], dtype=Config.NUMERIC_TYPE)

    metadata = {
        "Chemical": chemicals,
        "Concentration": concentrations,
        "Experiment_Number": experiment_numbers,
    }
    return metadata, features


def aggregate_features(metadata, features):
    """
    Aggregate features by unique experiment combinations.

    Args:
        metadata (dict): Metadata dictionary.
        features (np.ndarray): Features array.

    Returns:
        tuple: Aggregated metadata and features.
    """
    metadata["Group_Key"] = [
        (chem, conc, exp)
        for chem, conc, exp in zip(metadata["Chemical"], metadata["Concentration"], metadata["Experiment_Number"])
    ]
    unique_keys = list(set(metadata["Group_Key"]))
    aggregated_features = np.array([
        features[np.array(metadata["Group_Key"]) == key].mean(axis=0)
        for key in unique_keys
    ])
    aggregated_metadata = {
        "Chemical": [key[0] for key in unique_keys],
        "Concentration": [key[1] for key in unique_keys],
        "Experiment_Number": [key[2] for key in unique_keys],
    }
    return aggregated_metadata, aggregated_features


def perform_pca(features):
    """
    Perform PCA on the features.

    Args:
        features (np.ndarray): Aggregated features array.

    Returns:
        tuple: PCA-transformed features and explained variance ratios.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_
    return features_pca, explained_variance


def visualize_data(h5_path):
    """
    Visualize the data stored in the HDF5 file.

    Args:
        h5_path (str or Path): Path to the HDF5 file.
    """
    metadata, features = load_h5_data(h5_path)
    aggregated_metadata, aggregated_features = aggregate_features(metadata, features)
    features_pca, explained_variance = perform_pca(aggregated_features)

    print("Explained Variance Ratio for Each Component:")
    print(f"PC1: {explained_variance[0]:.4f}")
    print(f"PC2: {explained_variance[1]:.4f}")
    print(f"PC3: {explained_variance[2]:.4f}")
    print(f"Total Variance Explained by First 3 Components: {sum(explained_variance):.4f}")

    unique_chemicals = list(set(aggregated_metadata["Chemical"]))
    norm = Normalize(
        vmin=min(aggregated_metadata["Concentration"]),
        vmax=max(aggregated_metadata["Concentration"])
    )
    colormap_names = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]
    chemical_colormaps = {
        chemical: plt.colormaps.get_cmap(colormap_names[i % len(colormap_names)])
        for i, chemical in enumerate(unique_chemicals)
    }

    def get_color(row):
        normalized_concentration = norm(row["Concentration"])
        clamped_concentration = max(0.2, normalized_concentration)
        return chemical_colormaps[row["Chemical"]](clamped_concentration)

    aggregated_metadata["Color"] = [
        get_color({"Chemical": chem, "Concentration": conc})
        for chem, conc in zip(aggregated_metadata["Chemical"], aggregated_metadata["Concentration"])
    ]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 10), facecolor="#1a1a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

    for chemical in unique_chemicals:
        chem_mask = [chem == chemical for chem in aggregated_metadata["Chemical"]]
        ax.scatter(
            features_pca[chem_mask, 0],
            features_pca[chem_mask, 1],
            features_pca[chem_mask, 2],
            c=[aggregated_metadata["Color"][i] for i in range(len(chem_mask)) if chem_mask[i]],
            s=80,
            alpha=0.8
        )

    for i, chemical in enumerate(unique_chemicals):
        ax.scatter([], [], color=chemical_colormaps[chemical](1.0), label=f"Chemical {chemical}", s=80)

    ax.grid(color="#444444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.tick_params(colors="white")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Visualization")
    ax.legend(title="Chemicals", loc="upper left", labelcolor="white", facecolor="#2a2a2a", edgecolor="white")
    plt.show()
