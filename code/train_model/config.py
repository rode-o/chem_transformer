from pathlib import Path
from datetime import datetime
import torch

class Config:
    # General Settings
    ATTENTION_RESOLUTION = 8
    NUM_LAYERS = 2
    OUTPUT_DIM_CHEMICAL = None  # Dynamically determined by the dataset
    OUTPUT_DIM_CONCENTRATION = 1  # Regression output dimensions (single value)
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    RESERVED_MEMORY_GB = 0  # Memory isn't a concern; setting to zero

    # Data Type for Tensor Conversion
    TENSOR_FLOAT_TYPE = torch.float32  # Increased precision
    TENSOR_LONG_TYPE = torch.long  # Default tensor long type for integer targets

    # Validation Settings
    VALIDATION_SPLIT = 0.2  # 20% of the dataset reserved for validation
    VALIDATION_METRICS = ["accuracy", "r2"]  # Metrics for evaluation during validation

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODEL_SAVE_DIR = OUTPUT_DIR / "models"
    LOG_DIR = OUTPUT_DIR / "logs"

    # Logging
    LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # Unique log file for each session

    # Optional Training Settings
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 100

    # Ensure all required directories exist
    @staticmethod
    def ensure_directories():
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Ensure directories are created when the config is loaded
Config.ensure_directories()
