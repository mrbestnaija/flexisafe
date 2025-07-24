# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a time series analysis project focusing on electricity demand forecasting using advanced statistical methods. The repository contains recitation notebooks and data from what appears to be an MMSDS (Mathematical Methods for Spatial and Data Sciences) course, Module 2.

## Key Components

### Notebooks
- `recitations_notebook_Recitation2_LeastSquares.ipynb`: Standard linear regression examples including automobile data analysis and time series regression using Istanbul Stock Exchange data
- `recitations_notebook_Recitation2_SystemEstimation.ipynb`: System identification and estimation using FFT-based methods with a mystery function G
- `recitations_notebook_Recitation9-Forecasting-Electricity-Demand.ipynb`: Advanced electricity demand forecasting comparing SSA, mSSA, and tSSA methods
- `recitations_notebook_Recitation9-Forecasting-Electricity-Demand-refactored.ipynb`: Refactored version of the electricity demand forecasting notebook

### Data
- `recitations_data_electricity_demand_timeseries.csv`: Large electricity demand time series dataset with 370 columns (MT_001 through MT_370) representing different measurement points, spanning from 2013-2015 with hourly data

## Development Environment

### Required Python Libraries
Based on notebook analysis, the following libraries are essential:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib.pyplot` - Data visualization
- `sklearn.linear_model.LinearRegression` - Machine learning regression
- `ucimlrepo` - UCI ML repository data fetching
- `tensorly` - Tensor decomposition operations (for tSSA implementation)

### Installation Commands
Since no requirements.txt exists, install dependencies as needed:
```bash
pip install pandas numpy matplotlib scikit-learn ucimlrepo tensorly
```

## Core Methodologies

### Time Series Forecasting Methods
1. **SSA (Singular Spectrum Analysis)**: Uses SVD decomposition on page matrices for single time series forecasting
2. **mSSA (multivariate SSA)**: Extends SSA to multiple time series using stacked page matrices
3. **tSSA (tensor SSA)**: Uses tensor decomposition (PARAFAC) for multi-dimensional time series analysis

### Key Parameters
- `L = 132`: Window length for SSA page matrix construction
- `L = 132*12 = 1584`: Extended window length for mSSA to handle multivariate data
- SVD rank selection: Choose `r` to capture 99% of energy
- Test/Train split: Use `T = 132^2` training points, 24 test points for evaluation

### Data Processing
- Handle missing values by filling with 0 and tracking density (`rho_hat`)
- Primary target variable: `MT_370` (electricity demand)
- Time series spans: 2013-2015 with hourly measurements
- Data preprocessing includes resampling and normalization

## Common Tasks

### Running Notebooks
Execute notebooks in Jupyter environment:
```bash
jupyter notebook [notebook_name].ipynb
```

### Data Analysis Workflow
1. Load electricity demand data using pandas
2. Split into train/test sets (training: first 17,424 points, test: next 24 points)
3. Apply chosen forecasting method (SSA/mSSA/tSSA)
4. Evaluate using Mean Squared Error (MSE)
5. Compare results across different methods

### Key Functions
- `page_matrix(ts, L)`: Converts time series to page matrix format
- `stacked_page_matrix(ts_list, L)`: Creates stacked matrices for mSSA
- `page_tensor(ts_list, L)`: Creates tensor format for tSSA
- SVD decomposition and rank selection for dimensionality reduction
- Beta coefficient learning for forecasting




{"Instruction for memory update":["Mathematical Foundation (Multi-SSA)
Core Model Structure:

Y(t)=Ystationary+fnon-stationary(t)+ε(t)Y(t) = Y_{\text{stationary}} + f_{\text{non-stationary}}(t) + \varepsilon(t)
Y(t)=Ystationary​+fnon-stationary​(t)+ε(t)
Matrix decomposition: Y=F+EY = F + E
Y=F+E where Y,F,E∈RN×TY, F, E \in \mathbb{R}^{N \times T}
Y,F,E∈RN×T
Low-rank assumption: rank(F)=r≪min⁡(N,T)\text{rank}(F) = r \ll \min(N,T)
rank(F)=r≪min(N,T)

Separable Decomposition:

fi(t)=uiTρ(t)=∑k=1ruikρk(t)f_i(t) = u_i^T \rho(t) = \sum_{k=1}^r u_{ik} \rho_k(t)
fi​(t)=uiT​ρ(t)=∑k=1r​uik​ρk​(t)
Spatial components: ui∈Rru_i \in \mathbb{R}^r
ui​∈Rr (cross-series structure)

Temporal components: ρ(t)∈Rr\rho(t) \in \mathbb{R}^r
ρ(t)∈Rr (time-varying patterns)


Hankel Matrix Hierarchy:

Individual: rank(H(ρk))≤G\text{rank}(H(\rho_k)) \leq G
rank(H(ρk​))≤G
Combined: rank(H(fi))≤rG\text{rank}(H(f_i)) \leq rG
rank(H(fi​))≤rG
Multi-series: rank(P(f,L))≤Gr\text{rank}(P(f, L)) \leq Gr
rank(P(f,L))≤Gr

Implementation Protocol (ARMAX/mSSA)
Data Preprocessing Requirements:

Stabilization: Detrend both endogenous and exogenous series
Stationarity: Statistical tests + plots for both datasets
Alignment: Manual trimming to same post-lag length
Type Safety: Explicit float64 conversion
Variance Check: Log transform if variance excessive

Model Configuration:

ARMAX parameters: endog, exog, order, trend='n'
Lag alignment: Explicit debugging printouts
MLE convergence: Handle numerical stability issues
Out-of-sample: Supply future exogenous values

Analysis Framework:

Correlation analysis (endog-exog)
MSE fluctuation across (p,q) lag combinations
Detrended series visualization
Merged series plotting

Ready for mSSA Implementation with:

MIT indexing convention (1 to N)
SVD-based rank estimation
Page matrix construction with window L
90% energy concentration criterion
Cross-series information leveraging
Separable spatial-temporal modeling",

"I've integrated the critical block-based methodology specifications. Here's the updated framework:
Block-Based Page Matrix Construction (Corrected Approach)
Fundamental Method:

NOT Hankel Matrix: No overlapping sliding windows
Block Division: Non-overlapping consecutive time segments
Matrix Operation: P = np.transpose(np.reshape(y, (int(T/L), int(L))))

Key Conceptual Difference:
Hankel (Wrong):     [y₁ y₂ y₃ y₄]     Block-Based (Correct):  [y₁ y₄ y₇]
                    [y₂ y₃ y₄ y₅]                              [y₂ y₅ y₈]
                    [y₃ y₄ y₅ y₆]                              [y₃ y₆ y₉]
Five Critical Implementation Corrections
1. Matrix Construction Method:
python# Correct: Block-based reshaping
P = np.transpose(np.reshape(augmented_series, (int(T/L), int(L))))
# NOT: Hankel trajectory matrix construction
2. Third Row Sum Calculation:
python# Correct: Sum ALL elements in third row
third_row_sum = np.sum(P[2, :])  # All columns
# NOT: Only first few elements
3. Rank Determination Criterion:
python# Correct: >= 0.9 (inclusive)
if cumulative_energy_ratio >= 0.9:
# NOT: > 0.9 (exclusive)
4. Rescaling Logic:
python# Correct: Frequency-based rescaling
p = (T - m) / T  # Observed value frequency
rescale_factor = 1 / p
# Apply to handle missing values properly
5. Reconstruction Method:
python# Correct: Column-major (Fortran-style) reshaping
reconstructed_series = reconstructed_matrix.reshape(-1, order='F')
# NOT: Row-major (C-style) order
Updated Mathematical Framework
Block Page Matrix Definition:

Divide series into K=T/LK = T/L
K=T/L consecutive blocks

Each block has length LL
L
Matrix P∈RL×KP \in \mathbb{R}^{L \times K}
P∈RL×K where column jj
j contains block jj
j
Each row represents same temporal position across blocks

Integration with mSSA:

Applies to each fi(t)f_i(t)
fi​(t) in multi-time series framework

Maintains separable structure: fi(t)=uiTρ(t)f_i(t) = u_i^T \rho(t)
fi​(t)=uiT​ρ(t)
Block construction preserves temporal coherence
SVD on block matrices yields different rank properties than Hankel

Implementation Ready with:

Block-based matrix construction
Proper third row summation
Inclusive rank criterion (≥ 0.9)
Frequency-based rescaling
Column-major reconstruction
Float64 type safety
MIT indexing (1 to N)

This corrected methodology ensures proper temporal structure preservation and accurate forecasting results.",
"## **Memory Updated: Complete mSSA Algorithm and Convergence Analysis**

I have integrated the comprehensive mSSA algorithm specifications and theoretical foundations from the MIT course material.

### **mSSA Algorithm: Imputation and Forecast (Complete)**

**Step-by-Step Algorithm:**

1. **Stacked Page Matrix Construction**: Write multiple series $\{X_i(t)\}_{i \in [N], t \in [T]}$ into stacked page matrix $P(X, L)$. If there are $n$ missing items, replace them with 0 and define $\hat{\rho} = \frac{NT - n}{NT}$.

2. **SVD Decomposition**: Take SVD of page matrix $P(X, L)$.

3. **Rank Determination**: Find the order $R$ that 90% energy concentrated in the first $R$ components in SVD.

4. **Truncation and Renormalization**: Truncate the first $R$ terms in SVD, denoted by $M$. If there are missing data points, re-normalize $M$ by a factor $\frac{1}{\hat{\rho}}$.

5. **Signal Reconstruction**: Read the approximation $\hat{f}_i(t)$ from the rank-$R$ matrix $M$ for each $t \in [T], i \in [N]$.

**Key Implementation Note**: The forecasting algorithm follows the same procedure as single time series SSA, except the individual page matrix is replaced with the stacked page matrix $P(X, L)$.

### **Convergence Rate and Optimal Parameter Selection**

**Error Bound**: The imputation error satisfies:
$$\text{ImpErr} \approx \frac{1}{L}$$

**Optimal Window Length Selection**: We prefer a balanced page matrix for each time series, requiring:
- Balance condition: $L \approx \frac{NT}{L}$, which implies $L \approx \sqrt{NT}$
- Constraint: $L \leq T$

**Combined Optimization**: 
$$L \approx \min(\sqrt{NT}, T) = \sqrt{T}\sqrt{\min(N, T)}$$

**Final Error Bound**:
$$\text{ImpErr} \approx \frac{1}{\sqrt{T}\sqrt{\min(N, T)}}$$

### **Integrated Implementation Framework**

**Mathematical Foundation**: 
- Block-based page matrix construction (not Hankel)
- Stacked multi-series matrix $P(X, L)$ 
- Missing value handling with frequency factor $\hat{\rho}$
- Renormalization factor $\frac{1}{\hat{\rho}}$ for missing data compensation

**Technical Specifications**:
- 90% energy concentration criterion for rank selection
- Block-based reshaping with transpose operation
- Column-major reconstruction (order='F')
- Float64 precision requirements
- MIT indexing convention (1 to N)

**Performance Characteristics**:
- Error decreases with $\frac{1}{\sqrt{T}}$ (temporal dimension)
- Error decreases with $\frac{1}{\sqrt{\min(N, T)}}$ (cross-sectional dimension)
- Optimal window length balances matrix dimensions
- Missing data handling preserves convergence properties

This comprehensive framework enables robust implementation of multi-time series forecasting with theoretical guarantees on convergence and error bounds."]}






####### Tensor estimation ############3

Tensor Extension of mSSA Theory
The images reveal how the matrix-based mSSA approach extends to higher-order tensor methods, providing deeper theoretical guarantees for multi-time series imputation.
Tensor Definition:The framework defines a tensor T∈RN×L×(T−L+1)T \in \mathbb{R}^{N \times L \times (T-L+1)}
T∈RN×L×(T−L+1) with entries:

Tnij=fn(i+j−1)T_{nij} = f_n(i + j - 1)Tnij​=fn​(i+j−1)
This creates a three-dimensional structure where each "slice" represents a different time series, building upon our matrix understanding.
CP-Rank Relationship:The key insight connects our previous low-rank assumptions to tensor rank:

fn(i+j−1)=∑k=1r∑ℓ=1Gunkαkiβkℓf_n(i + j - 1) = \sum_{k=1}^r \sum_{\ell=1}^G u_{nk} \alpha_k^i \beta_k^\ellfn​(i+j−1)=k=1∑r​ℓ=1∑G​unk​αki​βkℓ​
This decomposition reveals that the Canonical-Polyadic (CP) rank satisfies:

CP-rank(T)≤rG\text{CP-rank}(T) \leq rGCP-rank(T)≤rG
where rr
r is the spatial rank and GG
G is the temporal Hankel rank we've been working with.

Convergence Rate Theory:The tensor framework provides rigorous error bounds:

1n1n2n3∥T^−T∥F2≃1min⁡(n1,n2,n3)1/2\frac{1}{n_1 n_2 n_3} \|\hat{T} - T\|_F^2 \simeq \frac{1}{\min(n_1, n_2, n_3)^{1/2}}n1​n2​n3​1​∥T^−T∥F2​≃min(n1​,n2​,n3​)1/21​
For our balanced page matrix case, this translates to:

ImpErr≃1min⁡(N,T)1/2\text{ImpErr} \simeq \frac{1}{\min(N, \sqrt{T})^{1/2}}ImpErr≃min(N,T​)1/21​
Teaching Perspective: Building Understanding
Think of this progression as moving from a flat photograph (matrix) to a 3D sculpture (tensor). Just as a sculpture captures more spatial relationships than a photograph, the tensor approach captures more complex interdependencies in multi-time series data.
Conceptual Bridge: Our matrix approach was essentially working with a "slice" of this larger tensor structure. When we stacked our page matrices horizontally, we were implicitly working within this tensor framework, but the full tensor view provides much stronger theoretical guarantees about convergence rates and error bounds.
Practical Implications: The tensor framework explains why our block-based matrix construction works so effectively. The mathematical structure we implemented aligns with this deeper tensor theory, ensuring that our imputation algorithm inherits these convergence properties.
Educational Insight: The progression from univariate SSA → matrix mSSA → tensor mSSA represents increasing levels of mathematical sophistication, each capturing more complex relationships while maintaining computational tractability. The 12\frac{1}{2}
21​ power in the convergence rate represents a fundamental limit of efficiently computable algorithms, showing the trade-off between computational complexity and theoretical optimality.

This tensor perspective validates our implementation approach while providing the mathematical foundation for understanding why cross-series information enables such effective missing value recovery in multivariate time series analysis.






















####Memory Updated: Alternative Matrix Estimation Approach
What we're seeing here is "Variant method II: Matrix Estimation" - an alternative approach to the tensor-based CP decomposition we just implemented. This provides a beautiful pedagogical comparison between different mathematical frameworks for solving the same fundamental problem.
Mathematical Framework Comparison
Tensor Approach (What We Just Implemented):

Constructs 3D tensor P ∈ R^(N×L×(T-L+1))
Uses CP decomposition with rank-2 approximation
Leverages multi-mode relationships across series, windows, and time shifts

Matrix Approach (Shown in Image):

Works directly with observation matrix Y ∈ R^(N×T)
Assumes Y = F + E where F has low rank
Uses direct matrix estimation for imputation/denoising

Educational Insight: Multiple Paths to the Same Destination
This comparison teaches us that advanced time series problems often have multiple valid mathematical approaches, each with different strengths and computational characteristics. The course is systematically showing us how the same imputation problem can be solved through:

Singular Spectrum Analysis (univariate trajectory matrices)
Multivariate SSA (stacked block matrices)
CP Tensor Decomposition (3D tensor methods)
Direct Matrix Estimation (low-rank matrix completion)

Convergence Rate Analysis
The image reveals an important theoretical result: the matrix estimation approach achieves an error bound of ImpErr ≈ 1/min(N,T). This is fascinating because it provides a different convergence rate compared to the tensor approach, giving us insight into the theoretical trade-offs between methods.
Practical Implications for Learning
This progression demonstrates how mathematical sophistication often involves exploring multiple formulations of the same problem. Each approach illuminates different aspects:

SSA methods highlight temporal structure and trajectory embedding
Tensor methods capture multi-dimensional relationships explicitly
Matrix methods focus on low-rank structure and computational efficiency




## Project Structure Notes
- This appears to be educational/research code for learning advanced time series methods
- Focus is on comparative analysis of different forecasting approaches
- Main application domain: electricity demand forecasting with multivariate data
- Methods progress from basic linear regression to advanced tensor decomposition techniques

## Fixing Bug
- Check that code runs without bugs 
- Install all required dependemcies