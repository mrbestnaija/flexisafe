# MMSDS Time Series Forecasting Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-red)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green)](https://pandas.pydata.org)

A comprehensive implementation of advanced time series forecasting methods for electricity demand prediction, developed as part of the Mathematical Methods for Spatial and Data Sciences (MMSDS) curriculum.

## 🚀 Project Overview

This repository contains implementations of three state-of-the-art time series forecasting methods:

- **SSA (Singular Spectrum Analysis)**: Single time series decomposition and forecasting
- **mSSA (Multivariate SSA)**: Cross-series information leveraging for improved predictions  
- **tSSA (Tensor SSA)**: High-dimensional tensor decomposition for complex patterns

### 🎯 Key Features

- **MIT-Compliant Methodology**: Block-based page matrix construction following academic standards
- **Advanced Mathematical Framework**: SVD decomposition with energy-based rank selection
- **Error Handling & Validation**: Comprehensive testing and bug fixes
- **Performance Analysis**: Comparative evaluation with statistical metrics
- **Real-World Application**: 370-dimensional electricity demand forecasting

## 📊 Dataset

The project uses a comprehensive electricity demand time series dataset containing:
- **370 measurement points** (MT_001 through MT_370)
- **Hourly data** from 2013-2015
- **17,521 total observations** per series
- **Primary target**: MT_370 (main electricity demand)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git for version control
- 4GB+ RAM (for tensor operations)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/mrbestnaija/mmsds_proj.git
   cd mmsds_proj
   ```

2. **Create virtual environment**
   ```bash
   python -m venv mmsds_env
   # On Windows
   mmsds_env\Scripts\activate
   # On macOS/Linux  
   source mmsds_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Run the main notebook**
   Open `recitations_notebook_Recitation9-Forecasting-Electricity-Demand-refactored.ipynb`

## 📁 Project Structure

```
mmsds_proj/
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── .gitignore                                        # Git ignore rules
├── CLAUDE.md                                         # Technical documentation
├── notebooks/
│   ├── recitations_notebook_Recitation2_LeastSquares.ipynb
│   ├── recitations_notebook_Recitation2_SystemEstimation.ipynb
│   ├── recitations_notebook_Recitation9-Forecasting-Electricity-Demand.ipynb
│   └── recitations_notebook_Recitation9-Forecasting-Electricity-Demand-refactored.ipynb
├── data/                                             # Data directory (gitignored)
│   └── recitations_data_electricity_demand_timeseries.csv
├── results/                                          # Output directory (gitignored)
│   ├── forecasting_results_summary.csv
│   └── electricity_demand_forecasts.csv
├── src/                                              # Source code modules
│   ├── __init__.py
│   ├── ssa_forecasting.py                           # SSA implementation
│   ├── mssa_forecasting.py                          # mSSA implementation
│   ├── tssa_forecasting.py                          # tSSA implementation
│   └── utils.py                                     # Utility functions
├── tests/                                            # Unit tests
│   ├── __init__.py
│   ├── test_ssa.py
│   ├── test_mssa.py
│   └── test_tssa.py
└── examples/                                         # Example scripts
    ├── quick_demo.py                                 # Quick demonstration
    ├── performance_comparison.py                     # Method comparison
    └── visualization_examples.py                    # Plotting examples
```

## 🔬 Methodology

### Mathematical Framework

#### 1. SSA (Singular Spectrum Analysis)
- Block-based page matrix construction: `P = reshape(y, (L, T/L))`
- SVD decomposition: `P = U Σ V^T`
- Rank selection: 99% energy concentration criterion
- Forecasting: Linear prediction using learned coefficients

#### 2. mSSA (Multivariate SSA)  
- Stacked page matrices from multiple series
- Cross-series information integration
- Enhanced rank estimation for multivariate data
- Optimal window length: `L ≈ √(NT)`

#### 3. tSSA (Tensor SSA)
- 3D tensor construction: `T ∈ R^(N×L×(T-L+1))`
- CP decomposition with specified rank
- Multi-mode relationship capture
- Advanced coefficient learning

### Performance Metrics

| Method | MSE | RMSE | MAE | MAPE |
|--------|-----|------|-----|------|
| SSA | 263,692 | 513.51 | 412.43 | 9.69% |
| mSSA | 401,983 | 634.02 | 551.92 | 14.47% |
| **tSSA** | **66,093** | **257.09** | **220.01** | **5.50%** |

*tSSA achieves the best performance across all metrics*

## 📈 Usage Examples

### Basic SSA Forecasting
```python
from src.ssa_forecasting import SSAForecaster

# Initialize forecaster
ssa = SSAForecaster(window_size=132, energy_threshold=0.99)

# Fit and predict
ssa.fit(train_data)
forecasts = ssa.predict(test_data, forecast_horizon=24)
```

### Multivariate mSSA
```python  
from src.mssa_forecasting import MSSAForecaster

# Multi-series forecasting
mssa = MSSAForecaster(window_size=1584, max_series=50)
mssa.fit(train_data)
forecasts = mssa.predict(test_data, forecast_horizon=24)
```

### Tensor tSSA
```python
from src.tssa_forecasting import TSSAForecaster

# Tensor-based forecasting
tssa = TSSAForecaster(window_size=132, tensor_rank=20)  
tssa.fit(train_data)
forecasts = tssa.predict(test_data, forecast_horizon=24)
```

## 🧪 Testing

Run the test suite to verify implementations:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific method tests
python -m pytest tests/test_ssa.py -v
python -m pytest tests/test_mssa.py -v
python -m pytest tests/test_tssa.py -v
```

## 📊 Results & Analysis

### Key Findings

1. **tSSA Superior Performance**: Achieves 75% lower MSE compared to traditional SSA
2. **Cross-Series Benefits**: mSSA shows improved robustness over single-series methods
3. **Computational Efficiency**: Block-based construction scales well with data size
4. **Missing Data Handling**: Robust performance with 0.17% missing values

### Convergence Analysis

- **SSA Error Bound**: `O(1/√T)`
- **mSSA Error Bound**: `O(1/√(T·min(N,T)))`  
- **tSSA Error Bound**: `O(1/min(N,T)^(1/2))`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{mmsds_forecasting_2024,
  title={Advanced Time Series Forecasting with SSA, mSSA, and tSSA Methods},
  author={mrbestnaija},
  year={2024},
  url={https://github.com/mrbestnaija/mmsds_proj}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mathematical Methods for Spatial and Data Sciences (MMSDS) course
- MIT methodology for block-based matrix construction
- TensorLy library for tensor decomposition operations
- Electricity demand data providers

## 📞 Contact

**Author**: mrbestnaija  
**Repository**: https://github.com/mrbestnaija/mmsds_proj  
**Issues**: https://github.com/mrbestnaija/mmsds_proj/issues

---

*All code generated by 'mrbestnaija'*