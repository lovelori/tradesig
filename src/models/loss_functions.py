import torch
import torch.nn as nn

class DeltaBasedLoss(nn.Module):
    """
    Custom loss function that compares model predictions with price delta
    """
    def __init__(self):
        super(DeltaBasedLoss, self).__init__()

    def forward(self, predictions, delta):
        """
        Calculate loss based on model predictions and actual price delta
        
        Args:
            predictions (torch.Tensor): Model output predictions
            delta (torch.Tensor): Actual price changes from the dataset
            
        Returns:
            torch.Tensor: Calculated loss value
        """
        # Ensure tensors are the same shape
        predictions = predictions.squeeze()
        if predictions.shape != delta.shape:
            raise ValueError(f"Predictions shape {predictions.shape} must match delta shape {delta.shape}")
            
        # Calculate loss as the product of predictions and delta
        
        loss = - torch.dot(predictions, delta) 
        return loss

# Example usage in training script

def train_model(model, train_loader, optimizer, num_epochs):
    criterion = DeltaBasedLoss()
    
    for epoch in range(num_epochs):
        for batch_idx, (data, delta) in enumerate(train_loader):
            # Forward pass
            predictions = model(data)
            
            # Calculate loss
            loss = criterion(predictions, delta)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log training progress
            
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')