import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
import psutil  # For memory checks
import shutil  # For disk I/O checks


# Dataset Class
class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.chemical = self.h5_file["Chemical"]
        self.concentration = self.h5_file["Concentration"]
        self.experiment_number = self.h5_file["Experiment_Number"]
        self.frequency = self.h5_file["Frequency"]
        self.features = self.h5_file["Features"]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "chemical": torch.tensor(self.chemical[idx], dtype=torch.long),
            "concentration": torch.tensor(self.concentration[idx], dtype=torch.float32),
            "experiment_number": torch.tensor(self.experiment_number[idx], dtype=torch.long),
            "frequency": torch.tensor(self.frequency[idx], dtype=torch.float32),
        }

    def close(self):
        self.h5_file.close()


# Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True  # Using batch_first to optimize inference
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        print(f"Input to Transformer (batch, sequence, features): {x.shape}")
        encoded = self.encoder(x)
        print(f"Output from Transformer Encoder: {encoded.shape}")
        output = self.fc(encoded[:, 0, :])  # Using the first token's representation
        print(f"Output after Fully Connected Layer: {output.shape}")
        return output


# Check System Resources
def check_system_resources():
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    print(f"Total Memory: {memory.total / 1e9:.2f} GB")
    print(f"Available Memory: {memory.available / 1e9:.2f} GB")
    print(f"Disk Total: {disk.total / 1e9:.2f} GB")
    print(f"Disk Free: {disk.free / 1e9:.2f} GB")

    # Estimate batch size based on available memory
    reserved_memory_gb = 1  # Reserve 1 GB for system processes
    available_memory_gb = memory.available / 1e9 - reserved_memory_gb
    batch_size = int((available_memory_gb * 1e9) // (75 * 4 * 64))  # Assuming 75 features, float32 (4 bytes), sequence=64
    print(f"Estimated Optimal Batch Size: {batch_size}")
    return batch_size


# Main Function
def main():
    try:
        print("Script started...")

        # Construct the path to the HDF5 file using pathlib
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent  # Go up one directory
        h5_file_path = project_root / "synth_data_h5" / "synthetic_dataset.h5"

        print(f"Loading data from: {h5_file_path}")

        # Check if the HDF5 file exists
        if not h5_file_path.exists():
            print(f"Error: File not found at {h5_file_path}")
            return

        # Check system resources
        batch_size = check_system_resources()

        # Load the dataset
        dataset = HDF5Dataset(h5_file_path)
        print(f"Dataset length: {len(dataset)}")

        if len(dataset) == 0:
            print("Error: Dataset is empty.")
            return

        # Use DataLoader for batching
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("DataLoader created successfully.")

        # Define Transformer model
        input_dim = 75  # Number of features per sample
        num_heads = 5   # Ensure input_dim is divisible by num_heads
        num_layers = 2
        output_dim = 10  # Example output classes

        # Check divisibility
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        # Check for CUDA and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        model = SimpleTransformer(input_dim, num_heads, num_layers, output_dim).to(device)
        print("Model initialized.")

        # Track processed samples
        total_samples = 0
        all_outputs = []

        # Iterate through all batches
        for idx, batch in enumerate(dataloader):
            print(f"Processing batch {idx + 1}...")
            features = batch["features"].to(device)  # Move features to GPU
            print(f"Loaded Features Shape: {features.shape}")
            print(f"Tensor device: {features.device}")

            # Add sequence length dimension for the Transformer
            features = features.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, input_dim)
            print(f"Features reshaped for Transformer: {features.shape}")

            # Pass through the model
            output = model(features)
            print(f"Batch {idx + 1} Model Output: {output.shape}")

            # Accumulate processed samples
            total_samples += features.size(0)
            all_outputs.append(output)

        # Verify total processed samples
        print(f"Total samples processed: {total_samples}")
        assert total_samples == len(dataset), "Not all samples were processed!"

        print("All samples successfully processed!")

        # Close the dataset
        dataset.close()
        print("Script finished successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
