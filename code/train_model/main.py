from logger import setup_logger
import logging
from pathlib import Path
import torch
from utils import analyze_data, pad_features, ensure_folders_exist
from dataset import HDF5Dataset
from model import MultiChemicalTransformer  # Correct model name
from train import train_model
from config import Config

# Initialize logger for the main script
logger = setup_logger(name="__main__", level=logging.INFO)

def main():
    try:
        logger.info("Script started...")

        # Path to dataset
        h5_file_path = Config.PROJECT_ROOT / "synth_data_h5" / "synthetic_dataset.h5"
        logger.info(f"Attempting to load dataset from: {h5_file_path}")

        if not h5_file_path.exists():
            logger.error(f"File not found at: {h5_file_path}")
            raise FileNotFoundError(f"Dataset file does not exist at: {h5_file_path}")

        # Ensure output folders exist
        ensure_folders_exist([Config.OUTPUT_DIR, Config.MODEL_SAVE_DIR, Config.LOG_DIR])

        # Analyze dataset and get parameters
        logger.info("Analyzing dataset...")
        (
            adjusted_features,
            padding_needed,
            num_features,
            sequence_length,
            batch_size,  # Use this batch size in training
            num_chemicals,  # Number of unique chemicals
        ) = analyze_data(
            h5_file_path,
            attention_resolution=Config.ATTENTION_RESOLUTION,
            reserved_memory_gb=Config.RESERVED_MEMORY_GB,
        )
        logger.info(
            "Dataset analysis complete:\n"
            f"  Adjusted Features: {adjusted_features}\n"
            f"  Padding Needed: {padding_needed}\n"
            f"  Number of Features: {num_features}\n"
            f"  Sequence Length: {sequence_length}\n"
            f"  Batch Size: {batch_size}\n"
            f"  Number of Chemicals: {num_chemicals}"
        )

        # Load dataset
        dataset = HDF5Dataset(
            h5_file_path,
            adjusted_features,
            pad_features,
            sequence_length=sequence_length,
            num_chemicals=num_chemicals,  # Correctly pass number of chemicals
        )

        logger.info(f"Dataset loaded with {len(dataset)} sweeps and sequence length {sequence_length}.")

        # Initialize the model
        num_heads = adjusted_features // Config.ATTENTION_RESOLUTION
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MultiChemicalTransformer(  # Correct model name
            input_dim=adjusted_features,
            num_heads=num_heads,
            num_layers=Config.NUM_LAYERS,
            num_chemicals=num_chemicals,  # Pass number of chemicals
        ).to(device)

        logger.info(
            "Model initialized:\n"
            f"  Adjusted Features: {adjusted_features}\n"
            f"  Num Heads: {num_heads}\n"
            f"  Num Layers: {Config.NUM_LAYERS}\n"
            f"  Sequence Length: {sequence_length}\n"
            f"  Number of Chemicals: {num_chemicals}"
        )

        # Start training
        train_model(
            model=model,
            dataset=dataset,  # Pass the dataset directly
            device=device,
            num_epochs=Config.NUM_EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            batch_size=batch_size,  # Pass calculated batch size
            early_stopping=Config.EARLY_STOPPING,
            early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
