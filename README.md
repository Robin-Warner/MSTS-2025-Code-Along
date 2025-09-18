
# MSTS Momentum Portfolio Code-Along

This repository contains a hands-on Jupyter workshop series that builds and improves a **market- and beta-neutral long/short momentum strategy** using S&P 500 price data. The project uses only `pandas`, `numpy`, and `matplotlib` for all analysis and visualization.

## Project Overview

The code-along guides you through the full quantitative investing workflow, from raw price data to a robust portfolio implementation. The September 2025 update introduces advanced risk controls, sector and beta neutralization, and more realistic portfolio construction techniques.


## Notebooks

The project contains several Jupyter notebooks:

1. **June 2025 Code Along - Student.ipynb**
    - For learners to complete exercises during the session
    - Contains code placeholders and TODOs for guided implementation
    - Step-by-step progression through the quant workflow

2. **June 2025 Code Along - Answers.ipynb**
    - Complete solutions to all exercises
    - Reference implementation of the full strategy

3. **September 2025 Code Along - Answers.ipynb**
    - Expanded and improved version with advanced portfolio construction, sector and beta neutralization, and more realistic constraints
    - Includes new risk management and performance analysis techniques


## What you'll practice
- Pull & clean price data
- Build a 12-month (252-day) momentum signal
- Apply advanced risk controls (leverage cap, sector & beta hedges, volatility targeting)
- Construct market- and beta-neutral portfolios
- Analyze performance and risk metrics


## Workflow Steps

1. **Data Collection** – Load historical prices and market data
2. **Data Curation** – Clean, handle missing values, and transform data
3. **Alpha Modeling** – Create and test momentum signals
4. **Risk Management** – Implement neutralization, volatility targeting, and risk controls
5. **Portfolio Construction** – Convert signals to positions with realistic constraints
6. **Performance Analysis** – Measure strategy performance and risk attributes


## Key Concepts Covered

- Cross-sectional momentum as a market factor
- Data cleaning and transformation techniques
- Market, sector, and beta neutralization methods
- Volatility targeting and leverage management
- Portfolio position sizing and risk management
- Performance analysis and attribution
## Helper Modules

Reusable Python modules in the `helpers/` directory provide functions for:
- Data cleaning and transformation (`data_functions.py`)
- Signal construction and sizing (`signal_functions.py`)
- Portfolio construction and risk management (`position_functions.py`)
- Performance and risk analysis (`performance_functions.py`)
- Visualization and charting (`charting_functions.py`)

These modules are used throughout the notebooks to keep code organized and reusable.

---


## Environment Setup

### Setting up a Virtual Environment

1. **Create a virtual environment**:
    ```bash
    # Windows
    python -m venv venv
    
    # macOS/Linux
    python3 -m venv venv
    ```

2. **Activate the virtual environment**:
    ```bash
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3. **Install required packages from requirements.txt**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch Jupyter notebook**:
    ```bash
    jupyter notebook
    ```

5. **When finished, deactivate the virtual environment**:
    ```bash
    deactivate
    ```