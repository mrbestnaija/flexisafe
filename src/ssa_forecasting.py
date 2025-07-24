"""
SSA (Singular Spectrum Analysis) Forecasting Implementation

This module provides a clean, object-oriented interface for SSA forecasting
following the MIT block-based methodology.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .utils import create_page_matrix, select_rank_by_energy


class SSAForecaster:
    """
    Singular Spectrum Analysis forecaster for univariate time series.
    
    Implements block-based page matrix construction with SVD decomposition
    and energy-based rank selection for time series forecasting.
    """
    
    def __init__(self, window_size: int = 132, energy_threshold: float = 0.99):
        """
        Initialize SSA forecaster.
        
        Args:
            window_size: Window length L for page matrix construction
            energy_threshold: Energy concentration threshold for rank selection
        """
        self.window_size = window_size
        self.energy_threshold = energy_threshold
        self.is_fitted = False
        
        # Model parameters (set during fitting)
        self.betas = None
        self.rank = None
        self.rho_hat = None
        self.metadata = {}
    
    def fit(self, train_data: pd.DataFrame, target_column: str = 'MT_370') -> 'SSAForecaster':
        """
        Fit SSA model to training data.
        
        Args:
            train_data: Training dataset
            target_column: Name of target column
            
        Returns:
            Self for method chaining
        """
        if target_column not in train_data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Handle missing values
        null_idx = train_data[target_column].isna()
        self.rho_hat = 1 - (null_idx.sum() / len(train_data))
        ts = train_data[target_column].fillna(0).to_numpy()
        
        # Step 1: Create page matrix
        page_matrix = create_page_matrix(ts, self.window_size)
        
        # Step 2: SVD and rank selection
        U, s, Vh = np.linalg.svd(page_matrix, full_matrices=False)
        self.rank = select_rank_by_energy(s, self.energy_threshold)
        
        # Step 3: Denoise matrix
        U_r = U[:, :self.rank]
        s_r = s[:self.rank]
        Vh_r = Vh[:self.rank, :]
        denoised_matrix = (U_r @ np.diag(s_r) @ Vh_r) / self.rho_hat
        
        # Step 4: Learn prediction coefficients
        Phi = denoised_matrix[:-1, :].T
        Y = page_matrix[-1, :]
        self.betas = np.linalg.pinv(Phi) @ Y
        
        # Store metadata
        self.metadata = {
            'method': 'SSA',
            'window_size': self.window_size,
            'rank': self.rank,
            'rho_hat': self.rho_hat,
            'energy_threshold': self.energy_threshold,
            'page_matrix_shape': page_matrix.shape,
            'training_samples': len(train_data)
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, full_data: pd.DataFrame, target_column: str = 'MT_370',
                forecast_horizon: int = 24, test_index: Optional[pd.Index] = None) -> pd.Series:
        """
        Generate forecasts using fitted SSA model.
        
        Args:
            full_data: Full dataset (including training portion)
            target_column: Name of target column
            forecast_horizon: Number of steps to forecast
            test_index: Index for forecast series
            
        Returns:
            Forecast series
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        full_ts = full_data[target_column].fillna(0).to_numpy()
        train_length = len(full_data) - forecast_horizon
        
        forecasts = []
        for t in range(train_length, train_length + forecast_horizon):
            if t >= self.window_size - 1:
                # Use L-1 previous values for prediction
                forecast_window = full_ts[t - self.window_size + 1:t]
                if len(forecast_window) == len(self.betas):
                    forecast = np.dot(self.betas, forecast_window)
                else:
                    forecast = np.nan
                forecasts.append(forecast)
            else:
                forecasts.append(np.nan)
        
        # Create index if not provided
        if test_index is None:
            if isinstance(full_data.index, pd.DatetimeIndex):
                start_idx = len(full_data) - forecast_horizon
                test_index = full_data.index[start_idx:]
            else:
                test_index = range(len(forecasts))
        
        return pd.Series(forecasts, index=test_index)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata and parameters.
        
        Returns:
            Dictionary of model metadata
        """
        if not self.is_fitted:
            return {}
        return self.metadata.copy()
    
    def summary(self) -> None:
        """Print model summary."""
        if not self.is_fitted:
            print("Model not fitted yet.")
            return
        
        print("SSA Forecaster Summary")
        print("=" * 30)
        print(f"Window Size (L): {self.window_size}")
        print(f"Selected Rank: {self.rank}")
        print(f"Energy Threshold: {self.energy_threshold}")
        print(f"Missing Data Ratio: {1 - self.rho_hat:.4f}")
        print(f"Page Matrix Shape: {self.metadata['page_matrix_shape']}")
        print(f"Training Samples: {self.metadata['training_samples']}")
        print(f"Beta Coefficients: {len(self.betas)}")