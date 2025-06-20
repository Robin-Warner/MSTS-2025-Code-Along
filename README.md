# Momentum Portfolio Code-Along

A one-hour Jupyter workshop that turns S&P 500 price data into a **market- and beta-neutral long/short momentum strategy** using only `pandas`, `numpy`, and `matplotlib`.

## Project Overview

This code-along session guides you through building a cross-sectional momentum strategy from raw price data to a final portfolio implementation. You'll experience the entire quantitative investing workflow while learning fundamental concepts in quantitative finance.

## Notebooks

The project contains two Jupyter notebooks:

1. **June 2025 Code Along - Student.ipynb**
   - Designed for learners to complete exercises during the session
   - Contains code placeholders with TODO comments for guided implementation
   - Structured in a step-by-step progression through the quant workflow

2. **June 2025 Code Along - Answers.ipynb**
   - Contains complete solutions to all exercises
   - Reference implementation of the full strategy
   - Use this for checking your work or if you get stuck

## What you'll practice
- Pull & clean price data  
- Build a 12-month (252-day) momentum signal  
- Apply simple risk controls (leverage cap, sector & beta hedges)  
- Create target weights and view basic performance stats  

## Workflow Steps

1. **Data Collection** - Loading historical prices and market data
2. **Data Curation** - Cleaning, handling missing values, transforming data
3. **Alpha Modeling** - Creating and testing momentum signals
4. **Risk Management** - Implementing neutralization and risk controls
5. **Portfolio Construction** - Converting signals to positions with risk constraints
6. **Performance Analysis** - Measuring strategy performance and attributes

## Key Concepts Covered

- Cross-sectional momentum as a market factor
- Data cleaning and transformation techniques
- Market and beta neutralization methods
- Portfolio position sizing and risk management
- Performance analysis and attribution

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