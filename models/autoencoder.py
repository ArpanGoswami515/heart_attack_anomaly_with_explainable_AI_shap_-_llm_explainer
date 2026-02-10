"""
Autoencoder anomaly detection model using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List
from .base_model import BaseAnomalyModel


class AutoencoderNetwork(nn.Module):
    """Neural network architecture for autoencoder."""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: List[int]):
        """
        Initialize autoencoder network.
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(AutoencoderNetwork, self).__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded representation
        """
        return self.encoder(x)


class Autoencoder(BaseAnomalyModel):
    """Autoencoder-based anomaly detection model."""
    
    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_dims: List[int] = [32, 16],
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Autoencoder model.
        
        Args:
            encoding_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            random_state: Random seed
        """
        super().__init__(name="Autoencoder")
        
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        self.model: nn.Module = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_errors_train: np.ndarray = None
        self.threshold_percentile = 95
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def fit(self, X: np.ndarray) -> "Autoencoder":
        """
        Train the autoencoder on normal samples only.
        
        Args:
            X: Training data (normal samples)
            
        Returns:
            self: Fitted model
        """
        input_dim = X.shape[1]
        
        # Initialize model
        self.model = AutoencoderNetwork(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_target in dataloader:
                # Forward pass
                reconstructed = self.model(batch_X)
                loss = criterion(reconstructed, batch_target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                break
        
        # Compute reconstruction errors on training data for threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructed) ** 2, 
                dim=1
            ).cpu().numpy()
            self.reconstruction_errors_train = reconstruction_errors
        
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on reconstruction error.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Normalized anomaly scores in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            
            # Compute reconstruction error per sample
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructed) ** 2,
                dim=1
            ).cpu().numpy()
        
        # Normalize scores to [0, 1]
        normalized_scores = self._normalize_scores(reconstruction_errors, reverse=False)
        
        return normalized_scores
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Get per-feature reconstruction errors.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Per-feature reconstruction errors (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            
            # Per-feature reconstruction error
            errors = ((X_tensor - reconstructed) ** 2).cpu().numpy()
        
        return errors
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input samples.
        
        Args:
            X: Input samples
            
        Returns:
            np.ndarray: Reconstructed samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor).cpu().numpy()
        
        return reconstructed
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dict containing model metadata
        """
        info = super().get_model_info()
        info.update({
            "encoding_dim": self.encoding_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": str(self.device)
        })
        
        return info
