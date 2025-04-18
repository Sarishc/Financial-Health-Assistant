# Financial Health Assistant

An ML-powered system that analyzes financial transactions, predicts future spending patterns, and provides personalized recommendations for saving money.

## Features

- Transaction categorization using NLP
- Spending pattern analysis
- Future spending forecasting
- Personalized saving recommendations

## Setup and Installation

```
pip install -r requirements.txt
```

## Usage

[Documentation coming soon]


# Financial Health Assistant

An ML-powered system that analyzes financial transactions, predicts future spending patterns, and provides personalized recommendations for saving money.

## Day 1 Progress Report

### Completed Tasks
- Set up project structure and development environment
- Implemented data schema definitions with Pydantic
- Created transaction processing pipeline for bank data
- Implemented basic rule-based transaction categorization
- Generated data visualizations for transaction analysis
- Successfully processed a large bank transaction dataset

### Data Insights
- Successfully processed transaction data with 116201 transactions
- Identified 6 different spending categories
- Top spending categories: other, food, income
- Largest expense category: other
- Date range: 2015-01-01 to 2019-03-05
- Total withdrawals (spending): 2,363,099.32
- Total deposits (income): 576,870.79
- Net cash flow: -1,786,228.53

### Challenges & Solutions
- Challenge: Initial dataset links were unavailable
  - Solution: Used bank statement data and transformed it to fit our needs
- Challenge: Transaction data had different column structure than expected
  - Solution: Created flexible processing pipeline that adapts to different data formats
- Challenge: Many transactions were difficult to automatically categorize
  - Solution: Implemented simple keyword-based categorization as a starting point

### Next Steps (Day 2)
- Improve transaction categorization with NLP model
- Implement time-series forecasting for spending prediction
- Develop recommendation engine logic
- Create API endpoints for transaction data

## Project Structure
- `app/`: Main application code
  - `data/`: Data processing and schema definitions
  - `models/`: ML model implementations (categorization, forecasting)
  - `api/`: API implementation
- `notebooks/`: Analysis scripts and visualizations
- `tests/`: Test scripts for components

## Setup Instructions

### Prerequisites
- Python 3.9+
- Pandas, NumPy, Matplotlib, Scikit-learn
- FastAPI (for API development in Day 2)

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

### Running the Analysis
- Process transaction data: `python notebooks/financial_data_exploration.py`
- Generate visualizations: `python notebooks/transaction_visualization.py`

# Financial Health Assistant
## Comprehensive Implementation Guide (Days 2-5)

This document provides detailed implementation information for the Financial Health Assistant project, covering Days 2 through 5 of the development process.

## Table of Contents

- [Overview](#overview)
- [Day 2: Data Preprocessing](#day-2-data-preprocessing)
- [Day 3: Transaction Categorization](#day-3-transaction-categorization)
- [Day 4: Time Series Preprocessing](#day-4-time-series-preprocessing)
- [Day 5: Spending Forecasting Models](#day-5-spending-forecasting-models)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)

## Day 2: Data Preprocessing

### Implementation Details

The data preprocessing stage transforms raw transaction data into clean, structured formats suitable for analysis.

#### Key Components:

1. **Transaction Data Cleaner** (`app/utils/processor.py`):
   - Handles missing values, duplicates, and outliers
   - Standardizes date and amount formats
   - Applies data type conversions
   - Removes unnecessary columns

2. **Data Schemas** (`app/schemas/`):
   - `schema.py`: Defines data schemas using Pydantic
   - Enforces data validation and consistency

3. **Preprocessing Pipeline** (`app/utils/processor.py`):
   - Executes multiple stages of data cleaning
   - Logs transformation steps
   - Provides data quality metrics

#### Implementation Approach:

The preprocessing pipeline follows these steps:
- Load raw transaction data from CSV/Excel files
- Normalize column names (handle spaces, capitalization)
- Convert date columns to datetime format
- Handle numeric fields (withdrawal, deposit amounts)
- Create unified amount column (positive for deposits, negative for withdrawals)
- Remove duplicates based on transaction details and timestamps
- Handle missing values through imputation or removal
- Save cleaned data to `app/data/processed/transactions_clean.csv`

## Day 3: Transaction Categorization

### Implementation Details

The transaction categorization system automatically classifies financial transactions into meaningful categories.

#### Key Components:

1. **Transaction Categorizer** (`app/models/categorization/nlp_categorizer.py`):
   - Uses NLP techniques to analyze transaction descriptions
   - Implements rule-based and machine learning approaches
   - Handles edge cases and ambiguous transactions

2. **Category Definitions** (`app/models/categorization/category_definitions.py`):
   - Defines hierarchical category structure
   - Maps keywords to categories
   - Provides category metadata

3. **Training Pipeline** (`app/models/categorization/train_categorizer.py`):
   - Trains categorization models using labeled data
   - Cross-validates model performance
   - Saves trained models for inference

#### Implementation Approach:

The categorization system works through these steps:
- Extract features from transaction descriptions (keywords, merchant names, etc.)
- Apply rule-based system for common transaction patterns
- Use machine learning models for more complex categorization
- Combine multiple approaches in an ensemble for better accuracy
- Provide confidence scores for each categorization
- Allow for manual overrides and feedback integration
- Continuously improve models with new data

#### Category Structure:

The system uses a two-level category hierarchy:
- Primary categories: Housing, Food, Transportation, Utilities, etc.
- Sub-categories: e.g., Food → Groceries, Restaurants, Takeout

## Day 4: Time Series Preprocessing

### Implementation Details

The time series preprocessing module transforms transaction data into time series format for analysis and forecasting.

#### Key Components:

1. **TimeSeriesProcessor** (`app/models/time_series/time_series_processor.py`):
   - Converts transaction data to time series format
   - Extracts temporal features
   - Creates lagged and rolling window features
   - Detects and handles outliers
   - Visualizes spending patterns

2. **Data Processing Scripts**:
   - `scripts/process_time_series.py`: Main script for time series processing
   - `scripts/day4_execution.py`: Script to execute all Day 4 tasks

#### Implementation Approach:

The time series preprocessing follows these steps:
- Aggregate transactions by day, week, and month
- Separate data by category for category-specific analysis
- Extract temporal features (day of week, month, year, etc.)
- Create lagged features to capture autocorrelation
- Implement rolling window statistics (mean, std, etc.)
- Add seasonal features (seasons, holidays, etc.)
- Detect and flag anomalous spending
- Visualize spending patterns by category and time period

#### Visualization Outputs:

The system creates several visualizations:
- Time series plots of spending by category
- Daily/weekly/monthly spending patterns
- Category breakdowns
- Seasonal spending patterns
- Outlier detection visualizations

## Day 5: Spending Forecasting Models

### Implementation Details

The spending forecasting module predicts future transaction patterns based on historical data.

#### Key Components:

1. **Base Forecaster** (`app/models/forecasting/base_forecaster.py`):
   - Abstract base class defining the forecaster interface
   - Implements common evaluation and visualization methods
   - Provides model persistence functionalities

2. **ARIMA Forecaster** (`app/models/forecasting/arima_forecaster.py`):
   - Implements AutoRegressive Integrated Moving Average models
   - Handles stationarity testing and differencing
   - Provides confidence intervals for forecasts

3. **Prophet Forecaster** (`app/models/forecasting/prophet_forecaster.py`):
   - Implements Facebook's Prophet forecasting model
   - Handles multiple seasonality patterns
   - Provides decomposition of trend, seasonality, and holidays

4. **Ensemble Forecaster** (`app/models/forecasting/ensemble_forecaster.py`):
   - Combines predictions from multiple forecasting models
   - Applies weighted averaging based on model performance
   - Improves forecast robustness

5. **Forecast Evaluator** (`app/models/forecasting/forecast_evaluator.py`):
   - Evaluates forecasting models using multiple metrics
   - Creates visualization comparisons
   - Generates evaluation reports

#### Implementation Approach:

The forecasting system follows these steps:
- Load preprocessed time series data
- Split data into training and testing sets
- Train multiple forecasting models (ARIMA, Prophet)
- Generate forecasts for specified time horizons
- Evaluate model performance using MAE, RMSE, MAPE
- Create ensemble forecasts by combining individual models
- Visualize forecasts with confidence intervals
- Save models and forecasts for later use
- Generate evaluation reports comparing model performance

#### Model Selection Process:

The system uses these criteria for selecting the best forecasting model:
- Performance metrics (RMSE, MAE, MAPE)
- Robustness to outliers
- Ability to capture seasonality patterns
- Computational efficiency
- Confidence interval accuracy

## Project Structure

```
financial-health-assistant/
│
├── app/
│   ├── api/
│   ├── data/
│   │   ├── processed/
│   │   │   ├── transactions_clean.csv
│   │   │   ├── time_series/
│   │   │   │   ├── daily_ts.csv
│   │   │   │   ├── weekly_ts.csv
│   │   │   │   └── monthly_ts.csv
│   │   │   └── forecasts/
│   │   └── raw/
│   ├── models/
│   │   ├── categorization/
│   │   │   ├── __init__.py
│   │   │   ├── nlp_categorizer.py
│   │   │   └── category_definitions.py
│   │   ├── time_series/
│   │   │   ├── __init__.py
│   │   │   └── time_series_processor.py
│   │   └── forecasting/
│   │       ├── __init__.py
│   │       ├── base_forecaster.py
│   │       ├── arima_forecaster.py
│   │       ├── prophet_forecaster.py
│   │       ├── ensemble_forecaster.py
│   │       └── forecast_evaluator.py
│   ├── schemas/
│   │   └── schema.py
│   └── utils/
│       ├── processor.py
│       └── auth.py
│
├── notebooks/
│   ├── financial_data_exploration.ipynb
│   ├── time_series_analysis.ipynb
│   └── visualizations/
│       ├── time_series/
│       └── forecasts/
│
├── scripts/
│   ├── day2_execution.py
│   ├── day3_execution.py
│   ├── day4_execution.py
│   ├── day5_execution.py
│   ├── process_time_series.py
│   ├── train_forecast_models.py
│   └── evaluate_forecasts.py
│
├── tests/
│   ├── test_processor.py
│   ├── test_categorizer.py
│   ├── test_time_series.py
│   └── test_forecasting.py
│
└── README.md
```

## Running the Project

### Prerequisites

- Python 3.6+
- Dependencies in requirements.txt

### Installation

```bash
# Clone the repository
git clone https://github.com/username/financial-health-assistant.git
cd financial-health-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

To run each day's tasks:

```bash
# Run Day 2: Data Preprocessing
python scripts/day2_execution.py

# Run Day 3: Transaction Categorization
python scripts/day3_execution.py

# Run Day 4: Time Series Preprocessing
python scripts/day4_execution.py

# Run Day 5: Spending Forecasting Models
python scripts/day5_execution.py
```

