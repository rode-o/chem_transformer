from pathlib import Path
from datetime import datetime
import torch

class Config:
    #---# Training Parameters  #----------------------------------------------------------------#
    ATTENTION_RESOLUTION = 8
    NUM_LAYERS = 2
    OUTPUT_DIM_CHEMICAL = None  # Dynamically determined by the dataset
    OUTPUT_DIM_CONCENTRATION = 1  # Regression output dimensions (single value)
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    RESERVED_MEMORY_GB = 0  # Memory isn't a concern; setting to zero

    #---# Data Type Settings #------------------------------------------------------------------#
    USE_64_BIT = True  # Toggle between 64-bit and 32-bit
    if USE_64_BIT:
        NUMERIC_TYPE = np.float64
        INT_TYPE = np.int64
        TENSOR_FLOAT_TYPE = torch.float64
        TENSOR_LONG_TYPE = torch.int64
    else:
        NUMERIC_TYPE = np.float32
        INT_TYPE = np.int32
        TENSOR_FLOAT_TYPE = torch.float32
        TENSOR_LONG_TYPE = torch.int32

    #---# Validation Settings #----------------------------------------------------------------#
    VALIDATION_SPLIT = 0.2  # 20% of the dataset reserved for validation
    VALIDATION_METRICS = ["accuracy", "r2"]  # Metrics for evaluation during validation

    #---# Synthetic Data Generation Parameters #-----------------------------------------------#
    NUM_CHEMICALS = 3
    NUM_CONCENTRATIONS = 10
    NUM_EXPERIMENTS_PER_CONCENTRATION = 10
    NUM_FEATURES = 75
    NUM_FREQ_POINTS = 100
    START_FREQ = 0.0
    STOP_FREQ = 6e9
    RANDOM_SEED = 42

    MIN_CONCENTRATION = 10
    MAX_CONCENTRATION = 100
    MAX_RADIUS = 5.0
    RADIUS_JITTER_SCALE = 0.5
    FEATURE_NOISE_SCALE = 0.05
    CLUSTER_CENTER_SCALE = 1.0

    #---# Paths #------------------------------------------------------------------------------#
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODEL_SAVE_DIR = OUTPUT_DIR / "models"
    LOG_DIR = OUTPUT_DIR / "logs"
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
    SYNTH_DATA_DIR = DATA_DIR / "synth_data"
    SYNTH_DATA_H5_DIR = DATA_DIR / "synth_data_h5"
    CSV_PATH = SYNTH_DATA_DIR / "synthetic_dataset.csv"  # Path to synthetic dataset CSV
    H5_PATH = SYNTH_DATA_H5_DIR / "synthetic_dataset.h5"  # Path to synthetic dataset HDF5

    #---# Logging #-----------------------------------------------------------------------------#
    LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # Unique log file for each session

    #---# Optional Training Settings #---------------------------------------------------------#
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 100

    #---# Ensure All Required Directories Exist #----------------------------------------------#
    @staticmethod
    def ensure_directories():
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.SYNTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.SYNTH_DATA_H5_DIR.mkdir(parents=True, exist_ok=True)

    #---# Seed Initialization #----------------------------------------------------------------#
    @staticmethod
    def initialize_seed():
        np.random.seed(Config.RANDOM_SEED)

# Ensure directories are created and seed is initialized when the config is loaded
Config.ensure_directories()
Config.initialize_seed()
