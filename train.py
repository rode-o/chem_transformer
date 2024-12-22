import torch
import torch.nn as nn
import torch.optim as optim
import logging
from logger import setup_logger
from config import Config  # Import Config for paths

# Initialize logger for this script
logger = setup_logger(name="train", level=logging.INFO)


def train_model(model, dataloader, device, num_epochs, learning_rate, early_stopping=False, early_stopping_patience=5):
    """
    Train the model with the given dataloader and configuration.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to train on (CPU or CUDA).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        early_stopping (bool): Whether to use early stopping.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    try:
        logger.info("Starting training loop...")

        # Define loss functions and optimizer
        classification_criterion = nn.CrossEntropyLoss()  # For chemical classification
        regression_criterion = nn.MSELoss()  # For concentration regression
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs} started...")
            model.train()  # Set model to training mode

            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                features = batch["features"].to(device)
                target_chemical = batch["chemical"].to(device)  # Classification target
                target_concentration = batch["concentration"].to(device)  # Regression target

                optimizer.zero_grad()

                # Forward pass
                output_chemical, output_concentration = model(features)

                # Calculate losses
                classification_loss = classification_criterion(output_chemical, target_chemical)
                regression_loss = regression_criterion(output_concentration.view(-1), target_concentration)
                loss = classification_loss + regression_loss
                epoch_loss += loss.item()

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                logger.debug(
                    f"Batch {batch_idx + 1}, Classification Loss: {classification_loss.item():.4f}, "
                    f"Regression Loss: {regression_loss.item():.4f}, Total Loss: {loss.item():.4f}"
                )

            average_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}")

            # Early stopping logic
            if early_stopping:
                if average_loss < best_loss:
                    best_loss = average_loss
                    patience_counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "best_model.pth")
                    logger.info(f"New best model saved with loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered. Training terminated.")
                        break

        # Save the final model
        torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "final_model.pth")
        logger.info(f"Final model saved at {Config.MODEL_SAVE_DIR / 'final_model.pth'}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise
