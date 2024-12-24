import logging
import torch.nn as nn
from logger import setup_logger

# Initialize logger for this script
logger = setup_logger(name="model", level=logging.DEBUG)


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

        # Log the initial state of the encoder
        logger.debug("Passing the tensor through the Transformer encoder...")

        # Transformer encoding
        encoded = self.encoder(x)
        logger.debug(f"After Transformer encoder: shape={encoded.shape}, dtype={encoded.dtype}")

        # Check if encoded tensor shape matches expectations
        if encoded.shape[2] != x.shape[2]:
            logger.warning(
                f"Mismatch detected after Transformer encoding. Expected last dimension {x.shape[2]}, got {encoded.shape[2]}."
            )

        # Pooling: Mean over the sequence dimension
        logger.debug("Applying mean pooling over the sequence dimension...")
        pooled = encoded.mean(dim=1)
        logger.debug(f"After mean pooling: shape={pooled.shape}, dtype={pooled.dtype}")

        # Classification output
        logger.debug("Generating classification output using the classification head...")
        output_chemical = self.softmax(self.fc_chemical(pooled))
        logger.debug(f"Classification output shape: {output_chemical.shape}, dtype={output_chemical.dtype}")

        # Regression output
        logger.debug("Generating regression output using the regression head...")
        output_concentration = self.fc_concentration(pooled)
        logger.debug(f"Regression output shape: {output_concentration.shape}, dtype={output_concentration.dtype}")

        # Final log before returning
        logger.debug("Returning classification and regression outputs.")
        return output_chemical, output_concentration
