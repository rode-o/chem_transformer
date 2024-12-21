import os
import pandas as pd
import h5py

# Full path to the dataset
csv_path = "/home/rop/Documents/analyzer_v2/ctrls/mach_model/synth_data/synthetic_dataset.csv"

# New HDF5 directory and file path
base_dir = os.path.dirname(os.path.dirname(csv_path))  # Parent directory of "synth_data"
h5_dir = os.path.join(base_dir, "synth_data_h5")
os.makedirs(h5_dir, exist_ok=True)
h5_path = os.path.join(h5_dir, "synthetic_dataset.h5")

# Convert CSV to HDF5
def convert_csv_to_h5(csv_path, h5_path):
    # Read the CSV file into a DataFrame
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Save to HDF5 format
    print(f"Saving data to HDF5 at {h5_path}...")
    with h5py.File(h5_path, "w") as h5file:
        for column in df.columns:
            h5file.create_dataset(column, data=df[column].values, compression="gzip")

    print("Conversion complete!")

# Execute the conversion
if __name__ == "__main__":
    if os.path.exists(csv_path):
        convert_csv_to_h5(csv_path, h5_path)
    else:
        print(f"CSV file not found at {csv_path}. Ensure the dataset exists before running this script.")
