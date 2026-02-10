"""
Data loading utilities for heart attack dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class HeartAttackDataLoader:
    """Data loader for heart attack medical dataset."""
    
    def __init__(self, file_path: str, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            file_path: Path to the dataset CSV file
            random_state: Random seed for reproducibility
        """
        self.file_path = file_path
        self.random_state = random_state
        self.feature_names: Optional[list] = None
        self.data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If dataset is empty
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        if self.data.empty:
            raise ValueError("Dataset is empty")
        
        # Handle "Heart Disease" column if present - convert to numeric target
        if "Heart Disease" in self.data.columns:
            # Convert Presence/Absence to 1/0
            self.data["target"] = self.data["Heart Disease"].map({
                "Presence": 1, 
                "Absence": 0
            })
            # Drop original column
            self.data = self.data.drop(columns=["Heart Disease"])
        
        return self.data
    
    def get_features_and_labels(
        self, 
        target_column: str = "target"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Split dataset into features and labels.
        
        Args:
            target_column: Name of the target column (if exists)
            
        Returns:
            Tuple of (features, labels). Labels is None if target column doesn't exist.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if target_column in self.data.columns:
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
        else:
            X = self.data
            y = None
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def split_normal_anomaly(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        normal_label: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into normal and anomaly samples.
        
        Args:
            X: Feature matrix
            y: Labels (0 = normal, 1 = anomaly/heart attack)
            normal_label: Label value for normal samples
            
        Returns:
            Tuple of (normal_samples, anomaly_samples)
        """
        normal_mask = y == normal_label
        X_normal = X[normal_mask]
        X_anomaly = X[~normal_mask]
        
        return X_normal, X_anomaly
    
    def train_test_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            test_size: Proportion of test set
            stratify: Whether to stratify split by labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_by = y if (stratify and y is not None) else None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_by
            )
        else:
            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=self.random_state
            )
            y_train, y_test = None, None
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not set. Load data first.")
        return self.feature_names
    
    @staticmethod
    def generate_sample_dataset(
        n_samples: int = 1000,
        n_features: int = 13,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate a synthetic heart attack dataset for testing.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            contamination: Proportion of anomalies
            random_state: Random seed
            
        Returns:
            pd.DataFrame: Generated dataset with target column
        """
        np.random.seed(random_state)
        
        # Feature names typical for heart disease datasets
        feature_names = [
            "age", "sex", "chest_pain_type", "resting_bp", 
            "cholesterol", "fasting_blood_sugar", "resting_ecg",
            "max_heart_rate", "exercise_angina", "st_depression",
            "st_slope", "num_major_vessels", "thalassemia"
        ]
        
        # Adjust if different number of features requested
        if n_features != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Generate normal samples
        n_normal = int(n_samples * (1 - contamination))
        n_anomaly = n_samples - n_normal
        
        # Normal samples: centered around 0
        X_normal = np.random.randn(n_normal, n_features)
        y_normal = np.zeros(n_normal)
        
        # Anomaly samples: shifted distribution
        X_anomaly = np.random.randn(n_anomaly, n_features) * 1.5 + 2
        y_anomaly = np.ones(n_anomaly)
        
        # Combine
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([y_normal, y_anomaly])
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y.astype(int)
        
        return df
