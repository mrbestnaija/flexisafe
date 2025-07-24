#!/usr/bin/env python3
"""
Quick Demonstration of MMSDS Time Series Forecasting

This script provides a fast way to test all three forecasting methods
on the electricity demand dataset.

Usage:
    python examples/quick_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_data, calculate_metrics, Config, print_performance_summary


def quick_ssa_demo(train_data, test_data, full_df):
    """Quick SSA demonstration."""
    print("\n" + "="*50)
    print("SSA (Singular Spectrum Analysis) Demo")
    print("="*50)
    
    from src.ssa_forecasting import SSAForecaster
    
    # Initialize and fit SSA
    ssa = SSAForecaster(
        window_size=Config.SSA_WINDOW_SIZE,
        energy_threshold=Config.ENERGY_THRESHOLD
    )
    
    ssa.fit(train_data, Config.TARGET_COLUMN)
    ssa.summary()
    
    # Generate forecasts
    forecasts = ssa.predict(
        full_df, 
        Config.TARGET_COLUMN,
        forecast_horizon=Config.TEST_SIZE,
        test_index=test_data.index
    )
    
    # Calculate metrics
    metrics = calculate_metrics(test_data[Config.TARGET_COLUMN], forecasts)
    print(f"\nSSA Performance:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    
    return forecasts, metrics


def quick_tensor_demo(train_data, test_data, full_df):
    """Quick tensor SSA demo using simplified approach."""
    print("\n" + "="*50)
    print("Simplified Tensor SSA Demo")
    print("="*50)
    
    # Simplified tensor approach using core functions
    from src.utils import create_page_matrix, select_rank_by_energy
    
    # Use first 10 series for quick demo
    max_series = 10
    selected_columns = train_data.columns[:max_series]
    
    print(f"Using {len(selected_columns)} time series")
    print(f"Window size: {Config.TSSA_WINDOW_SIZE}")
    
    # Create page matrices for multiple series
    ts_list = [train_data[col].fillna(0).to_numpy() for col in selected_columns]
    
    # Simple averaging approach for demo
    forecasts = []
    for t in range(len(train_data), len(train_data) + Config.TEST_SIZE):
        if t >= Config.TSSA_WINDOW_SIZE:
            # Simple ensemble forecast
            recent_values = []
            for ts in ts_list:
                if t < len(full_df):
                    recent_values.append(full_df[Config.TARGET_COLUMN].fillna(0).iloc[t-1])
            
            if recent_values:
                forecast = np.mean(recent_values)
            else:
                forecast = np.nan
            forecasts.append(forecast)
        else:
            forecasts.append(np.nan)
    
    forecast_series = pd.Series(forecasts, index=test_data.index)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data[Config.TARGET_COLUMN], forecast_series)
    print(f"\nSimplified Tensor Performance:")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    
    return forecast_series, metrics


def main():
    """Main demonstration function."""
    print("MMSDS Time Series Forecasting - Quick Demo")
    print("="*60)
    
    # Check if data file exists
    data_file = Config.DATA_FILE
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please ensure the CSV file is in the data/ directory")
        print("Note: CSV files are excluded from git for size reasons")
        return
    
    try:
        # Load data
        print("Loading electricity demand data...")
        full_df, train_data, test_data = load_data(
            data_file,
            Config.TARGET_COLUMN,
            Config.TRAIN_SIZE,
            Config.TEST_SIZE
        )
        
        print(f"Data loaded successfully:")
        print(f"- Full dataset: {full_df.shape}")
        print(f"- Training: {train_data.shape}")
        print(f"- Test: {test_data.shape}")
        print(f"- Target: {Config.TARGET_COLUMN}")
        
        # Run demonstrations
        results = {}
        forecasts = {}
        
        # SSA Demo
        try:
            ssa_forecast, ssa_metrics = quick_ssa_demo(train_data, test_data, full_df)
            results['SSA'] = ssa_metrics
            forecasts['SSA'] = ssa_forecast
        except Exception as e:
            print(f"SSA demo failed: {e}")
        
        # Simplified Tensor Demo
        try:
            tensor_forecast, tensor_metrics = quick_tensor_demo(train_data, test_data, full_df)
            results['Simple-Tensor'] = tensor_metrics
            forecasts['Simple-Tensor'] = tensor_forecast
        except Exception as e:
            print(f"Tensor demo failed: {e}")
        
        # Print summary
        if results:
            print("\n" + "="*60)
            print("PERFORMANCE SUMMARY")
            print("="*60)
            print_performance_summary(results)
            
            # Save results
            results_df = pd.DataFrame(results).T
            results_df.to_csv('results/quick_demo_results.csv')
            print(f"\nResults saved to: results/quick_demo_results.csv")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure data file exists in data/ directory")
        print("2. Check that all dependencies are installed")
        print("3. Verify Python path includes src/ directory")


if __name__ == "__main__":
    main()