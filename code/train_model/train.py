import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
from logger import setup_logger
import logging
from config import Config  # Import Config for paths

# Initialize logger for this script
logger = setup_logger(name="train", level=logging.INFO)


def calculate_metrics(output_chemical, target_chemical, output_concentration, target_concentration):
    """
    Calculate validation metrics for multi-chemical classification and regression tasks.

    Args:
        output_chemical (torch.Tensor): Model predictions for chemical classification (multi-hot).
        target_chemical (torch.Tensor): Ground truth for chemical classification.
        output_concentration (torch.Tensor): Model predictions for regression.
        target_concentration (torch.Tensor): Ground truth for regression.

    Returns:
        dict: Calculated metrics (accuracy and R²).
    """
    # Accuracy for multi-hot classification (thresholded at 0.5)
    predicted_classes = (torch.sigmoid(output_chemical) > 0.5).float()
    accuracy = (predicted_classes == target_chemical).float().mean().item()

    # R² for regression
    r2 = r2_score(
        target_concentration.detach().cpu().numpy(),
        output_concentration.detach().cpu().numpy()
    )

    return {"accuracy": accuracy, "r2": r2}


def train_model(
    model,
    dataset,
    device,
    num_epochs,
    learning_rate,
    batch_size,  # Use dynamic batch size
    early_stopping=False,
    early_stopping_patience=5,
):
    """
    Train the model with the given dataset and configuration.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataset (Dataset): The dataset for training and validation.
        device (torch.device): Device to train on (CPU or CUDA).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training and validation loaders.
        early_stopping (bool): Whether to use early stopping.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    try:
        logger.info("Starting training loop...")

        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        val_size = int(dataset_size * Config.VALIDATION_SPLIT)
        train_size = dataset_size - val_size

        if train_size <= 0 or val_size <= 0:
            raise ValueError(
                f"Invalid dataset split sizes. Ensure VALIDATION_SPLIT ({Config.VALIDATION_SPLIT}) results in valid train/val sizes."
            )

        logger.info(f"Splitting dataset: {train_size} training samples, {val_size} validation samples.")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss functions and optimizer
        classification_criterion = nn.BCEWithLogitsLoss()  # Multi-hot classification
        regression_criterion = nn.MSELoss()  # Regression
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs} started...")

            # Training phase
            model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                features = batch["features"].to(device)
                target_chemical = batch["chemical"].to(device)
                target_concentration = batch["concentration"].to(device)

                optimizer.zero_grad()

                # Forward pass
                output_chemical, output_concentration = model(features)

                # Calculate losses
                classification_loss = classification_criterion(output_chemical, target_chemical)
                regression_loss = regression_criterion(output_concentration, target_concentration)
                loss = classification_loss + regression_loss
                total_train_loss += loss.item()

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

            # Validation phase
            model.eval()
            total_val_loss = 0.0
            all_metrics = {metric: 0.0 for metric in Config.VALIDATION_METRICS}
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(device)
                    target_chemical = batch["chemical"].to(device)
                    target_concentration = batch["concentration"].to(device)

                    # Forward pass
                    output_chemical, output_concentration = model(features)

                    # Calculate losses
                    classification_loss = classification_criterion(output_chemical, target_chemical)
                    regression_loss = regression_criterion(output_concentration, target_concentration)
                    total_val_loss += classification_loss.item() + regression_loss.item()

                    # Calculate metrics
                    metrics = calculate_metrics(output_chemical, target_chemical, output_concentration, target_concentration)
                    for metric, value in metrics.items():
                        all_metrics[metric] += value

            # Average validation metrics
            avg_val_loss = total_val_loss / len(val_loader)
            for metric in all_metrics:
                all_metrics[metric] /= len(val_loader)

            logger.info(
                f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}, "
                + ", ".join([f"{metric.capitalize()} = {value:.4f}" for metric, value in all_metrics.items()])
            )

            # Early stopping logic
            if early_stopping:
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "best_model.pth")
                    logger.info(f"New best model saved with loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered. Training terminated.")
                        break

        torch.save(model.state_dict(), Config.MODEL_SAVE_DIR / "final_model.pth")
        logger.info(f"Final model saved at {Config.MODEL_SAVE_DIR / 'final_model.pth'}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise
