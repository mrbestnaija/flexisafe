"""
MMSDS Time Series Forecasting Package

This package provides implementations of advanced time series forecasting methods:
- SSA (Singular Spectrum Analysis)
- mSSA (Multivariate SSA) 
- tSSA (Tensor SSA)

Author: mrbestnaija
"""

__version__ = "1.0.0"
__author__ = "mrbestnaija"

from .ssa_forecasting import SSAForecaster
from .mssa_forecasting import MSSAForecaster  
from .tssa_forecasting import TSSAForecaster
from .utils import load_data, calculate_metrics, plot_results

__all__ = [
    'SSAForecaster',
    'MSSAForecaster', 
    'TSSAForecaster',
    'load_data',
    'calculate_metrics',
    'plot_results'
]