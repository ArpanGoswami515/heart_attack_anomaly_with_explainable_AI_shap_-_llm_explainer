"""
Data preprocessing utilities for heart attack anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """Preprocess medical data for anomaly detection models."""
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: Scaling method ("standard" or "minmax")
        """
        self.scaling_method = scaling_method
        self.scaler: Optional[object] = None
        self.feature_names: Optional[list] = None
        
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
    
    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            
        Returns:
            self: Fitted preprocessor
        """
        self.feature_names = X.columns.tolist()
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            np.ndarray: Scaled features
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            np.ndarray: Scaled features
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X: Scaled features
            
        Returns:
            np.ndarray: Original scale features
        """
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        return self.scaler.inverse_transform(X)
    
    def handle_missing_values(
        self, 
        X: pd.DataFrame, 
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            X: Features with potential missing values
            strategy: Strategy to handle missing values ("mean", "median", "drop")
            
        Returns:
            pd.DataFrame: Features with handled missing values
        """
        if strategy == "drop":
            return X.dropna()
        elif strategy == "mean":
            return X.fillna(X.mean())
        elif strategy == "median":
            return X.fillna(X.median())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def remove_outliers(
        self, 
        X: pd.DataFrame, 
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Remove extreme outliers using z-score method.
        
        Args:
            X: Features
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Tuple of (cleaned features, outlier mask)
        """
        z_scores = np.abs((X - X.mean()) / X.std())
        outlier_mask = (z_scores < threshold).all(axis=1)
        
        return X[outlier_mask], outlier_mask
    
    def get_feature_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basic statistics for each feature.
        
        Args:
            X: Features
            
        Returns:
            pd.DataFrame: Statistics summary
        """
        stats = pd.DataFrame({
            "mean": X.mean(),
            "std": X.std(),
            "min": X.min(),
            "max": X.max(),
            "median": X.median(),
            "missing": X.isnull().sum()
        })
        
        return stats
    
    def validate_data(self, X: pd.DataFrame) -> bool:
        """
        Validate input data for common issues.
        
        Args:
            X: Features to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data has issues
        """
        # Check for empty dataset
        if X.empty:
            raise ValueError("Dataset is empty")
        
        # Check for all-NaN columns
        nan_cols = X.columns[X.isnull().all()].tolist()
        if nan_cols:
            raise ValueError(f"Columns with all NaN values: {nan_cols}")
        
        # Check for constant columns
        constant_cols = X.columns[X.nunique() <= 1].tolist()
        if constant_cols:
            raise ValueError(f"Constant columns detected: {constant_cols}")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Dataset contains infinite values")
        
        return True
    
    def get_feature_names(self) -> list:
        """
        Get feature names.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set. Fit preprocessor first.")
        return self.feature_names
