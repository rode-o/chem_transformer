import logging
import torch.nn as nn
from logger import setup_logger

# Initialize logger for this script
logger = setup_logger(name="model", level=logging.INFO)


class SimpleTransformer(nn.Module):
    """
    A multitask Transformer model for simultaneous chemical classification and concentration regression.
    """

    def __init__(self, input_dim, num_heads, num_layers, output_dim_chemical, output_dim_concentration, dim_feedforward=512):
        """
        Initialize the Transformer model.

        Args:
            input_dim (int): Dimension of input features.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            output_dim_chemical (int): Output dimension for the chemical classification task.
            output_dim_concentration (int): Output dimension for the concentration regression task.
            dim_feedforward (int): Hidden dimension size for the feedforward network in Transformer layers.
        """
        super(SimpleTransformer, self).__init__()

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Separate heads for multitask learning
        self.fc_chemical = nn.Linear(input_dim, output_dim_chemical)  # Classification head
        self.fc_concentration = nn.Linear(input_dim, output_dim_concentration)  # Regression head

        # Activation functions
        self.softmax = nn.Softmax(dim=1)  # For classification

        logger.info(
            f"SimpleTransformer initialized:\n"
            f"  Input Dim: {input_dim}\n"
            f"  Num Heads: {num_heads}\n"
            f"  Num Layers: {num_layers}\n"
            f"  Dim Feedforward: {dim_feedforward}\n"
            f"  Output Dim (Chemical): {output_dim_chemical}\n"
            f"  Output Dim (Concentration): {output_dim_concentration}"
        )

    def forward(self, x):
        """
        Forward pass for the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            tuple: (output_chemical, output_concentration)
        """
        encoded = self.encoder(x)  # Shape: (batch_size, sequence_length, input_dim)

        # Use the representation of the first token (CLS token equivalent)
        output_chemical = self.softmax(self.fc_chemical(encoded[:, 0, :]))  # Classification output
        output_concentration = self.fc_concentration(encoded[:, 0, :])  # Regression output

        logger.debug(f"Forward pass completed. Outputs:\n"
                     f"  Chemical Shape: {output_chemical.shape}\n"
                     f"  Concentration Shape: {output_concentration.shape}")
        return output_chemical, output_concentration
