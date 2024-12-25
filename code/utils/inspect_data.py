import h5py

def inspect_h5_file(h5_path):
    with h5py.File(h5_path, "r") as h5file:
        print("Datasets in the HDF5 file:")
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f" - {name}")
        h5file.visititems(print_datasets)

# Inspect the file structure
inspect_h5_file(r"C:\Users\RodePeters\Desktop\chem_transformer\synth_data_h5\synthetic_dataset.h5")