"""
Utility functions for time series forecasting

Contains common functions used across all forecasting methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings


def load_data(file_path: str, target_column: str, 
              train_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split time series data.
    
    Args:
        file_path: Path to CSV file
        target_column: Name of target column
        train_size: Number of training samples
        test_size: Number of test samples
        
    Returns:
        Tuple of (full_data, train_data, test_data)
    """
    try:
        full_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if target_column not in full_df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        train_data = full_df.iloc[:train_size, :]
        test_data = full_df.iloc[train_size:train_size + test_size, :]
        
        return full_df, train_data, test_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file '{file_path}' not found")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def create_page_matrix(ts: np.ndarray, L: int) -> np.ndarray:
    """
    Create page matrix from time series using block-based construction.
    
    Args:
        ts: Time series array
        L: Window length
        
    Returns:
        Page matrix of shape (L, T//L)
    """
    T = int(len(ts))
    if L > T:
        raise ValueError(f"Window size L ({L}) > time series length ({T})")
    
    cols = int(T // L)
    page_matrix = np.zeros((L, cols))
    
    for t in range(min(T, L * cols)):
        i = t % L
        j = t // L
        if j < cols:
            page_matrix[i, j] = ts[t]
    
    return page_matrix


def select_rank_by_energy(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """
    Select rank based on energy concentration criterion.
    
    Args:
        singular_values: SVD singular values
        threshold: Energy threshold (default 0.99)
        
    Returns:
        Selected rank
    """
    cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    r = np.argmax(cumulative_energy >= threshold) + 1
    return min(r, len(singular_values))


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    valid_mask = ~(actual.isna() | predicted.isna())
    actual_clean = actual[valid_mask]
    predicted_clean = predicted[valid_mask]
    
    if len(actual_clean) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
    
    # Calculate metrics
    mse = np.mean((actual_clean - predicted_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    
    # MAPE with zero protection
    mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100 \
           if actual_clean.min() > 0 else np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'valid_predictions': len(actual_clean)
    }


def plot_results(actual: pd.Series, forecasts_dict: Dict[str, pd.Series], 
                title: str = "Forecasting Results") -> None:
    """
    Create visualization of forecasting results.
    
    Args:
        actual: Actual time series values
        forecasts_dict: Dictionary of forecasts {method: series}
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.subplot(2, 1, 1)
    plt.plot(actual.index, actual.values, 'k-', linewidth=2, label='Actual')
    
    # Plot forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (method, forecast) in enumerate(forecasts_dict.items()):
        plt.plot(forecast.index, forecast.values, 
                color=colors[i % len(colors)], linestyle='--', 
                linewidth=2, label=f'{method} Forecast')
    
    plt.title(f'{title} - Time Series Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    for i, (method, forecast) in enumerate(forecasts_dict.items()):
        residuals = actual - forecast
        plt.plot(residuals.index, residuals.values,
                color=colors[i % len(colors)], alpha=0.7, label=f'{method} Residuals')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title('Forecast Residuals')
    plt.xlabel('Time') 
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_performance_summary(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print formatted performance summary table.
    
    Args:
        results_dict: Dictionary of {method: metrics_dict}
    """
    print(f"\n{'Method':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-" * 65)
    
    for method, metrics in results_dict.items():
        mse = metrics['mse']
        rmse = metrics['rmse'] 
        mae = metrics['mae']
        mape = metrics['mape']
        
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
        print(f"{method:<10} {mse:<12.2f} {rmse:<12.2f} {mae:<12.2f} {mape_str:<12}")
    
    # Find best method
    valid_methods = {k: v for k, v in results_dict.items() if not np.isnan(v['mse'])}
    if valid_methods:
        best_method = min(valid_methods.keys(), key=lambda k: valid_methods[k]['mse'])
        best_mse = valid_methods[best_method]['mse']
        print(f"\nBest performing method: {best_method} (MSE: {best_mse:.2f})")


class Config:
    """Configuration class for forecasting parameters."""
    
    # Data parameters
    DATA_FILE = 'data/recitations_data_electricity_demand_timeseries.csv'
    TARGET_COLUMN = 'MT_370'
    TRAIN_SIZE = 132**2  # 17,424
    TEST_SIZE = 24
    
    # Method parameters
    SSA_WINDOW_SIZE = 132
    MSSA_WINDOW_SIZE = 132 * 12  # 1,584
    TSSA_WINDOW_SIZE = 132
    ENERGY_THRESHOLD = 0.99
    TSSA_RANK = 20
    MAX_SERIES = 50
    
    # Random seed for reproducibility
    RANDOM_SEED = 42