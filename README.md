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
