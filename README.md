# Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs for Portfolio Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 Research Paper

**Title:** Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs for Portfolio Optimization

**Authors:** Kholood Alsager, Nada Almuairi

**Submission ID:** 5a92129b-ff87-4133-a92d-fb078edb5b24

**Journal:** Scientific Reports (Nature)

**Period of Study:** October 2024

**DOI:** [To be assigned upon publication]

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Data](#data)
- [Usage Examples](#usage-examples)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🎯 Abstract

This repository contains the complete implementation of our research on integrating Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs (M-PFFGs) for portfolio optimization. Our approach combines:

- **LSTM Neural Networks** for stock return prediction
- **M-Polar Fermatean Fuzzy Graphs** for uncertainty modeling
- **Genetic Algorithms** for portfolio weight optimization
- **Sentiment Analysis** from social media (X/Twitter)

The study focuses on five major tech stocks (AAPL, MSFT, AMZN, GOOGL, TSLA) during October 2024, demonstrating superior risk-adjusted returns compared to traditional methods.

## ✨ Key Features

- ✅ **Complete LSTM Implementation** - Predict stock returns with deep learning
- ✅ **Genetic Algorithm Optimizer** - Find optimal portfolio weights
- ✅ **Fuzzy Graph Construction** - Model uncertainty using M-PFFGs
- ✅ **Sentiment Analysis Pipeline** - Extract market sentiment from social media
- ✅ **Data Processing Scripts** - Clean and prepare financial data
- ✅ **Visualization Tools** - Generate publication-quality figures
- ✅ **Fully Reproducible** - All random seeds and configurations documented

## 📁 Repository Structure

```
m-pffg-portfolio-optimization/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore file
│
├── data/                              # Data directory
│   ├── raw/                          # Raw data files
│   │   ├── stock_prices_oct2024.csv
│   │   ├── trading_volumes_oct2024.csv
│   │   └── sentiment_scores.csv
│   ├── processed/                    # Processed data
│   │   ├── normalized_data.csv
│   │   └── fuzzy_values.csv
│   └── README.md                     # Data documentation
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── lstm_model.py                 # LSTM prediction model
│   ├── genetic_algorithm.py          # GA optimizer
│   ├── fuzzy_graph.py                # M-PFFG construction
│   ├── sentiment_analysis.py         # NLP pipeline
│   ├── data_processing.py            # Data preprocessing
│   └── utils.py                      # Utility functions
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_lstm_training.ipynb
│   ├── 03_fuzzy_graph_construction.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_results_analysis.ipynb
│
├── scripts/                           # Executable scripts
│   ├── train_lstm.py
│   ├── run_optimization.py
│   └── generate_figures.py
│
├── tests/                             # Unit tests
│   ├── test_lstm.py
│   ├── test_ga.py
│   └── test_fuzzy_graph.py
│
├── results/                           # Output directory
│   ├── tables/                       # Generated tables
│   ├── figures/                      # Generated figures
│   └── models/                       # Saved models
│
└── docs/                              # Documentation
    ├── methodology.md
    ├── api_reference.md
    └── mathematical_formulations.pdf
```

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster LSTM training

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/m-pffg-portfolio-optimization.git
cd m-pffg-portfolio-optimization
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (for sentiment analysis)

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## 🚀 Quick Start

### Train LSTM Model

```python
from src.lstm_model import StockLSTMPredictor, predict_october_returns

# Predict returns for all stocks
predictions = predict_october_returns(['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'])
print(predictions)
```

### Run Portfolio Optimization

```python
from src.genetic_algorithm import run_portfolio_optimization

# Optimize portfolio weights
optimal_weights, results = run_portfolio_optimization()
print(f"Optimal Weights: {optimal_weights}")
```

### Complete Pipeline

```bash
# Run the complete analysis pipeline
python scripts/run_optimization.py --stocks AAPL MSFT AMZN GOOGL TSLA
```

## 🧮 Methodology

### 1. Data Collection

- **Stock Data:** Daily closing prices and trading volumes from Yahoo Finance
- **Sentiment Data:** X/Twitter posts analyzed using VADER sentiment analysis
- **Period:** September 2024 (training), October 2024 (testing)

### 2. LSTM Prediction

**Model Architecture:**
- Input Layer: (lookback_period=20, features=4)
- LSTM Layer 1: 64 units, tanh activation
- Dropout: 0.2
- LSTM Layer 2: 32 units, tanh activation
- Dropout: 0.2
- Dense Output: 1 unit, linear activation

**Training Parameters:**
- Optimizer: Adam (learning_rate=0.001)
- Loss Function: Mean Squared Error
- Epochs: 100
- Batch Size: 32
- Validation Split: 20%

### 3. M-Polar Fermatean Fuzzy Graph

**Membership Degree Calculation:**

$$\phi_i = \frac{1}{\sqrt{\text{Volatility}}}$$

**Non-Membership Degree:**

$$\psi_i = \sqrt{1 - (\phi_i)^2}$$

**Edge Weight Formula:**

$$h_{ij} = \frac{1}{3} \sum_{t=1}^{3} \min\left(1, \sqrt[3]{\phi_i(\mathfrak{R}_t) \cdot \phi_j(\mathfrak{R}_t)} + \left[1 - \sqrt[3]{\psi_i(\mathfrak{R}_t) \cdot \psi_j(\mathfrak{R}_t)}\right]\right)$$

### 4. Genetic Algorithm Optimization

**Objective Function:**
Minimize portfolio variance adjusted by fuzzy risk factors

**Constraints:**
- Weights sum to 1
- Minimum return ≥ 0.92%
- Non-negative weights

**GA Parameters:**
- Population Size: 100
- Generations: 200
- Crossover Rate: 0.8
- Mutation Rate: 0.1
- Elite Size: 5
- Selection: Tournament (size=3)

## 📊 Data

### Data Availability Statement

All data generated or analyzed during this study are included in this repository:

- **Raw stock data:** `data/raw/stock_prices_oct2024.csv`
- **Sentiment scores:** `data/raw/sentiment_scores.csv`
- **Processed fuzzy values:** `data/processed/fuzzy_values.csv`

The raw data can be reproduced by running:

```bash
python scripts/collect_data.py --start-date 2024-09-01 --end-date 2024-10-31
```

### Data Format

**Stock Prices (Table 22):**
| Stock | Closing Price (Oct 31) | Avg. Return (%) | Avg. Volatility (%) | Volume (M) |
|-------|------------------------|-----------------|---------------------|------------|
| AAPL  | 222.56                 | 0.042           | 1.25                | 58.3       |
| MSFT  | 406.35                 | 0.035           | 1.18                | 2.1        |
| AMZN  | 186.40                 | 0.055           | 1.45                | 38.9       |
| GOOGL | 171.11                 | 0.032           | 1.10                | 25.6       |
| TSLA  | 43.75                  | 0.095           | 2.10                | 96.2       |

## 💻 Usage Examples

### Example 1: Train LSTM for Single Stock

```python
from src.lstm_model import StockLSTMPredictor
import pandas as pd

# Load data
data = pd.read_csv('data/raw/AAPL_sept2024.csv')

# Initialize predictor
predictor = StockLSTMPredictor(lookback_period=20, random_state=42)

# Prepare data
X, y = predictor.prepare_data(data)

# Build and train model
predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
predictor.train(X, y, epochs=100, batch_size=32)

# Make prediction
prediction = predictor.predict(X[-1:])
print(f"Predicted return: {prediction[0]:.2f}%")
```

### Example 2: Construct Fuzzy Graph

```python
from src.fuzzy_graph import MPolarFermateanFuzzyGraph

# Initialize graph
graph = MPolarFermateanFuzzyGraph(stocks=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'])

# Add vertices with fuzzy values
graph.add_vertex('AAPL', md=[0.60, 0.56, 0.61], nmd=[0.92, 0.93, 0.91])
# ... add other vertices

# Calculate edge weights
edge_weights = graph.calculate_edge_weights()
print(edge_weights)
```

### Example 3: Sensitivity Analysis

```python
from src.genetic_algorithm import GeneticAlgorithmPortfolio

# Test different minimum return constraints
min_returns = [0.80, 0.92, 1.10]
results = []

for min_return in min_returns:
    ga = GeneticAlgorithmPortfolio()
    weights, fitness = ga.optimize(
        returns=returns,
        cov_matrix=cov_matrix,
        fuzzy_risk_adjustments=fuzzy_risk,
        min_return=min_return
    )
    results.append({'min_return': min_return, 'weights': weights})
```

## 🔄 Reproducibility

### Random Seeds

All stochastic processes use fixed random seeds:

- **LSTM Training:** `random_seed=42`
- **Genetic Algorithm:** `random_seed=42`
- **Data Splitting:** `random_state=42`

### Hardware/Software Environment

**Development Environment:**
- OS: Ubuntu 22.04 LTS
- Python: 3.9.18
- TensorFlow: 2.13.0
- NumPy: 1.24.3
- CPU: Intel Core i7 / AMD Ryzen 7
- (Optional) GPU: NVIDIA RTX 3070 with CUDA 11.8

**Reproducibility Checklist:**
- ✅ All random seeds documented
- ✅ Package versions specified in requirements.txt
- ✅ Data preprocessing steps documented
- ✅ Model architectures fully described
- ✅ Hyperparameters explicitly stated

### Running Complete Reproduction

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download data
python scripts/collect_data.py

# 3. Train LSTM models
python scripts/train_lstm.py

# 4. Run optimization
python scripts/run_optimization.py

# 5. Generate all figures and tables
python scripts/generate_figures.py
```

## 📈 Results

### Portfolio Performance (Table 27)

| Method              | Return (%) | Volatility (%) | Sharpe Ratio | Max Drawdown (%) |
|---------------------|------------|----------------|--------------|------------------|
| Mean-Variance       | 0.92       | 1.40           | 0.66         | 2.10             |
| LSTM Only           | 1.15       | 1.35           | 0.85         | 1.95             |
| M-PFFG-AI (Original)| 1.38       | 1.30           | 1.08         | 1.80             |
| **M-PFFG-Risk**     | **0.98**   | **1.10**       | **0.89**     | **1.50**         |

### Optimal Portfolio Weights (Table 26)

| Stock | Predicted Return (%) | Volatility (%) | Fuzzy-Adjusted Risk | Weight |
|-------|---------------------|----------------|---------------------|--------|
| AAPL  | 0.85                | 1.25           | 0.12                | 0.17   |
| MSFT  | 0.72                | 1.18           | 0.11                | 0.13   |
| AMZN  | 1.10                | 1.45           | 0.14                | 0.50   |
| GOOGL | 0.65                | 1.10           | 0.10                | 0.20   |
| TSLA  | 1.95                | 2.10           | 0.19                | 0.00   |

### Key Findings

1. **Risk Reduction:** M-PFFG-Risk approach reduced portfolio volatility by 21.4% compared to Mean-Variance
2. **Stable Returns:** Achieved 0.98% return with lowest maximum drawdown (1.50%)
3. **Sentiment Integration:** Social media sentiment improved prediction accuracy by 15%
4. **AMZN Allocation:** Fuzzy analysis identified AMZN as optimal risk-adjusted investment (50% weight)

## 📚 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{alsager2024mpffg,
  title={Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs for Portfolio Optimization},
  author={Alsager, Kholood and Almuairi, Nada},
  journal={Scientific Reports},
  year={2024},
  publisher={Nature Publishing Group},
  note={Submission ID: 5a92129b-ff87-4133-a92d-fb078edb5b24}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Kholood Alsager**
- Conceptualization, Methodology, Investigation, Writing (Original Draft)
- Email: kholood.alsager@qu.edu.sa
- Affiliation: Qassim University

**Nada Almuairi**
- Writing (Review and Editing)
- Email: nada.almuairi@qu.edu.sa
- Affiliation: Qassim University

## 🙏 Acknowledgments

The researchers thank the Deanship of Graduate Studies and Scientific Research at Qassim University for financial support (QU-APC-2025).

## 📞 Contact

For questions, issues, or collaborations:

- **Email:** kholood.alsager@qu.edu.sa
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/m-pffg-portfolio-optimization/issues)

## 🔗 Links

- **Research Paper:** [Link to published paper]
- **Supplementary Materials:** [Link to supplementary files]
- **Dataset (Zenodo):** [DOI will be added]
- **Project Website:** [If applicable]

---

**Last Updated:** January 2026

**Version:** 1.0.0

**Status:** ✅ Code Review Complete | 🔄 Under Revision
