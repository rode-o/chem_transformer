import logging
import torch.nn as nn
import torch
from logger import setup_logger

# Initialize logger for this script
logger = setup_logger(name="model", level=logging.DEBUG)


class MultiChemicalTransformer(nn.Module):
    """
    A multitask Transformer model for simultaneous multi-chemical classification and concentration regression.
    """

    def __init__(self, input_dim, num_heads, num_layers, num_chemicals, dim_feedforward=512):
        """
        Initialize the Transformer model.

        Args:
            input_dim (int): Dimension of input features.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            num_chemicals (int): Number of unique chemicals in the dataset.
            dim_feedforward (int): Hidden dimension size for the feedforward network in Transformer layers.
        """
        super(MultiChemicalTransformer, self).__init__()

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Multi-label classification head for chemical presence
        self.fc_presence = nn.Linear(input_dim, num_chemicals)

        # Regression head for chemical concentrations
        self.fc_concentration = nn.Linear(input_dim, num_chemicals)

        # Activation functions
        self.sigmoid = nn.Sigmoid()  # For multi-label classification (presence prediction)
        self.softmax = nn.Softmax(dim=1)  # For normalized concentration predictions

        logger.info(
            f"MultiChemicalTransformer initialized:\n"
            f"  Input Dim: {input_dim}\n"
            f"  Num Heads: {num_heads}\n"
            f"  Num Layers: {num_layers}\n"
            f"  Dim Feedforward: {dim_feedforward}\n"
            f"  Number of Chemicals: {num_chemicals}"
        )

    def forward(self, x):
        """
        Forward pass for the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            tuple: (chemical_presence, chemical_concentrations)
        """
        # Log the input shape
        logger.debug(f"Initial input tensor shape: {x.shape} (batch_size, sequence_length, input_dim expected)")

        # Validate the input tensor shape
        if len(x.shape) != 3:
            logger.error(f"Input tensor shape mismatch. Got {x.shape}, expected (batch_size, sequence_length, input_dim).")
            raise ValueError(
                f"Expected input tensor of shape (batch_size, sequence_length, input_dim), but got shape {x.shape}"
            )

        # Log the tensor dtype
        logger.debug(f"Input tensor dtype: {x.dtype}")

        # Transformer encoding
        encoded = self.encoder(x)
        logger.debug(f"After Transformer encoder: shape={encoded.shape}, dtype={encoded.dtype}")

        # Pooling: Mean over the sequence dimension
        pooled = encoded.mean(dim=1)
        logger.debug(f"After mean pooling: shape={pooled.shape}, dtype={pooled.dtype}")

        # Multi-label classification for chemical presence
        logger.debug("Generating chemical presence predictions...")
        chemical_presence = self.sigmoid(self.fc_presence(pooled))  # Sigmoid for presence probabilities
        logger.debug(f"Chemical presence output shape: {chemical_presence.shape}, dtype={chemical_presence.dtype}")

        # Normalized regression for chemical concentrations
        logger.debug("Generating normalized chemical concentration predictions...")
        chemical_concentrations = self.softmax(self.fc_concentration(pooled))  # Softmax for normalized concentrations
        logger.debug(f"Chemical concentrations output shape: {chemical_concentrations.shape}, dtype={chemical_concentrations.dtype}")

        # Final log before returning
        logger.debug("Returning chemical presence and concentrations outputs.")

        return chemical_presence, chemical_concentrations
