# Financial Health Assistant API Documentation

## Overview

The Financial Health Assistant API provides a comprehensive set of endpoints for financial transaction analysis, spending forecasting, and personalized financial recommendations. This RESTful API is built using FastAPI and provides authentication, data validation, and comprehensive documentation.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication. To access protected endpoints, you need to:

1. Register a user account (`/register`)
2. Obtain an access token (`/token`)
3. Include the token in the Authorization header for all protected requests:

```
Authorization: Bearer {your_access_token}
```

## Forecasts

### Get Forecasts

```
GET /forecasts
```

Get spending forecasts for all categories or a specific category.

**Query Parameters:**
- `days`: Number of days to forecast (default: 30 days)
- `category`: Filter by specific category (optional)

**Response:**
```json
{
  "forecasts": [
    {
      "category": "food",
      "forecast_points": [
        {
          "date": "2025-04-20",
          "amount": 58.25,
          "lower_bound": 45.75,
          "upper_bound": 70.85
        },
        ...
      ],
      "total_forecast": 1750.50,
      "historical_average": 1680.75,
      "percent_change": 4.15,
      "confidence_level": 0.80,
      "forecast_start_date": "2025-04-20",
      "forecast_end_date": "2025-05-19",
      "forecast_period_days": 30
    },
    ...
  ],
  "total": 5
}
```

### Get Forecast Summary

```
GET /forecasts/summary
```

Get a summary of forecasts across all categories.

**Query Parameters:**
- `days`: Number of days to forecast (default: 30 days)

**Response:**
```json
{
  "total_forecast": 5240.75,
  "categories_forecast": {
    "food": 1750.50,
    "transport": 850.25,
    "utilities": 520.00,
    "entertainment": 420.00,
    "shopping": 1700.00
  },
  "forecast_period_days": 30,
  "forecast_start_date": "2025-04-20",
  "forecast_end_date": "2025-05-19",
  "generated_at": "2025-04-19T15:30:00"
}
```

### Get Forecast Status

```
GET /forecasts/status
```

Get status of forecast models.

**Response:**
```json
{
  "is_trained": true,
  "last_trained": "2025-04-15T10:30:00",
  "categories_trained": [
    "food",
    "transport",
    "utilities",
    "entertainment",
    "shopping"
  ],
  "models_available": [
    "food.joblib",
    "transport.joblib",
    "utilities.joblib",
    "entertainment.joblib",
    "shopping.joblib"
  ],
  "accuracy_metrics": {
    "average_mape": 12.5,
    "average_rmse": 18.75,
    "model_details": {}
  }
}
```

### Train Forecast Models

```
POST /forecasts/train
```

Train new forecast models using transaction data.

**Query Parameters:**
- `days`: Number of days of history to use for training (default: 90 days)

**Response:**
```json
{
  "message": "Successfully trained 5 forecasting models"
}
```

## User Management

### Get Current User

```
GET /users/me
```

Get current user information.

**Response:**
```json
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "disabled": false
}
```

### Update User

```
PUT /users/me/update
```

Update current user information.

**Request Body:**
```json
{
  "full_name": "John Smith"
}
```

**Response:**
Updated user object.

### Change Password

```
PUT /users/me/change-password
```

Change user password.

**Request Body:**
```json
{
  "current_password": "old_password",
  "new_password": "new_secure_password"
}
```

**Response:**
```json
{
  "message": "Password updated successfully"
}
```

### Disable Account

```
PUT /users/me/disable
```

Disable user account.

**Response:**
```json
{
  "message": "Account disabled successfully"
}
```

### Get User Preferences

```
GET /users/preferences
```

Get user preferences.

**Response:**
```json
{
  "currency": "USD",
  "dateFormat": "MM/DD/YYYY",
  "theme": "light",
  "notifications": {
    "email": true,
    "push": false
  },
  "dashboardWidgets": [
    "spending_summary",
    "category_breakdown",
    "top_recommendations"
  ]
}
```

### Update User Preferences

```
PUT /users/preferences
```

Update user preferences.

**Request Body:**
```json
{
  "currency": "EUR",
  "theme": "dark"
}
```

**Response:**
Updated preferences object.

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Request succeeded
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failed
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

Error responses include a detail message explaining the error:

```json
{
  "detail": "Transaction not found"
}
```

## Rate Limiting

The API implements rate limiting to protect against abuse:

- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

When the rate limit is exceeded, the API returns a `429 Too Many Requests` status code.

## Versioning

The API uses URL versioning with the format `/api/vX` where X is the version number. The current version is `/api/v1`.

## Additional Resources

- [SwaggerUI Documentation](http://localhost:8000/docs): Interactive API documentation
- [ReDoc Documentation](http://localhost:8000/redoc): Alternative API documentation
```

### Authentication Endpoints

#### Register

```
POST /register
```

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "message": "User registered successfully"
}
```

#### Login

```
POST /token
```

Authenticate and obtain an access token.

**Request Form Data:**
- `username`: Email address
- `password`: User password

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Transactions

### Get Transactions

```
GET /transactions
```

Get a list of transactions with optional filters.

**Query Parameters:**
- `skip`: Number of transactions to skip (pagination)
- `limit`: Maximum number of transactions to return
- `start_date`: Filter by start date (ISO format)
- `end_date`: Filter by end date (ISO format)
- `category`: Filter by category
- `min_amount`: Filter by minimum amount
- `max_amount`: Filter by maximum amount

**Response:**
```json
{
  "transactions": [
    {
      "id": "12345678-1234-5678-1234-567812345678",
      "description": "Grocery Store",
      "amount": -58.75,
      "transaction_date": "2025-04-15T10:30:00",
      "category": "food",
      "account_id": "main_checking",
      "notes": "Weekly groceries",
      "created_at": "2025-04-15T12:30:00",
      "updated_at": null
    },
    ...
  ],
  "total": 120
}
```

### Create Transaction

```
POST /transactions
```

Create a new transaction.

**Request Body:**
```json
{
  "description": "Grocery Store",
  "amount": -58.75,
  "transaction_date": "2025-04-15T10:30:00",
  "category": "food",
  "account_id": "main_checking",
  "notes": "Weekly groceries"
}
```

**Response:**
Transaction object with generated ID and timestamps.

### Get Transaction

```
GET /transactions/{transaction_id}
```

Get a single transaction by ID.

**Response:**
Transaction object.

### Update Transaction

```
PUT /transactions/{transaction_id}
```

Update a transaction.

**Request Body:**
```json
{
  "description": "Updated description",
  "amount": -65.50,
  "category": "food"
}
```

**Response:**
Updated transaction object.

### Delete Transaction

```
DELETE /transactions/{transaction_id}
```

Delete a transaction.

**Response:**
```json
{
  "message": "Transaction deleted successfully"
}
```

### Get Transaction Statistics

```
GET /transactions/stats
```

Get transaction statistics.

**Query Parameters:**
- `start_date`: Start date for statistics (ISO format)
- `end_date`: End date for statistics (ISO format)

**Response:**
```json
{
  "total_income": 5000.0,
  "total_expenses": 3250.75,
  "net_cashflow": 1749.25,
  "period_start": "2025-03-20T00:00:00",
  "period_end": "2025-04-19T00:00:00",
  "categories": {
    "food": -850.25,
    "transport": -450.0,
    "income": 5000.0,
    "utilities": -320.0,
    "entertainment": -280.5
  }
}
```

## Categories

### Get Categories

```
GET /categories
```

Get a list of all categories.

**Response:**
```json
{
  "categories": [
    {
      "name": "food",
      "description": "Food and dining expenses",
      "color": "#FF5733",
      "icon": "utensils",
      "parent_category": null
    },
    ...
  ],
  "total": 9
}
```

### Create Category

```
POST /categories
```

Create a new category.

**Request Body:**
```json
{
  "name": "dining_out",
  "description": "Restaurant and dining out expenses",
  "color": "#FFA07A",
  "icon": "restaurant",
  "parent_category": "food"
}
```

**Response:**
Category object.

### Get Category

```
GET /categories/{category_name}
```

Get a single category by name.

**Response:**
Category object.

### Update Category

```
PUT /categories/{category_name}
```

Update a category.

**Request Body:**
```json
{
  "name": "dining_out",
  "description": "Updated description",
  "color": "#FFA07A",
  "icon": "utensils"
}
```

**Response:**
Updated category object.

### Delete Category

```
DELETE /categories/{category_name}
```

Delete a category.

**Response:**
```json
{
  "message": "Category deleted successfully"
}
```

### Get Category Statistics

```
GET /categories/{category_name}/stats
```

Get statistics for a specific category.

**Query Parameters:**
- `start_date`: Start date for statistics (ISO format)
- `end_date`: End date for statistics (ISO format)

**Response:**
```json
{
  "name": "food",
  "transaction_count": 15,
  "total_amount": 850.25,
  "average_transaction": 56.68,
  "percentage_of_total": 26.2,
  "month_to_month_change": 5.3
}
```

### Get Category Hierarchy

```
GET /categories/hierarchy
```

Get hierarchical structure of categories.

**Response:**
```json
[
  {
    "name": "food",
    "subcategories": [
      {
        "name": "groceries",
        "subcategories": []
      },
      {
        "name": "dining_out",
        "subcategories": []
      }
    ]
  },
  ...
]
```

### Categorize Transactions

```
POST /categories/categorize-transactions
```

Categorize uncategorized transactions using the trained model.

**Response:**
```json
{
  "message": "Successfully categorized 35 transactions"
}
```

## Recommendations

### Get Recommendations

```
GET /recommendations
```

Get a list of recommendations with optional filters.

**Query Parameters:**
- `skip`: Number of recommendations to skip (pagination)
- `limit`: Maximum number of recommendations to return
- `min_priority`: Filter by minimum priority level (1-10)
- `max_priority`: Filter by maximum priority level (1-10)
- `type`: Filter by recommendation type
- `category`: Filter by category

**Response:**
```json
{
  "recommendations": [
    {
      "id": "12345678-1234-5678-1234-567812345678",
      "message": "Your food spending is 28.5% of total expenses in the last 30 days. Consider ways to reduce this category.",
      "type": "high_spending",
      "priority": 8,
      "category": "food",
      "amount": 850.25,
      "percentage": 28.5,
      "created_at": "2025-04-19T14:30:00"
    },
    ...
  ],
  "recommendation_by_type": {
    "high_spending": [
      {
        "id": "12345678-1234-5678-1234-567812345678",
        "message": "Your food spending is 28.5% of total expenses in the last 30 days. Consider ways to reduce this category.",
        "type": "high_spending",
        "priority": 8,
        "category": "food",
        "amount": 850.25,
        "percentage": 28.5,
        "created_at": "2025-04-19T14:30:00"
      }
    ],
    "saving_opportunity": [
      {
        "id": "23456789-2345-6789-2345-678923456789",
        "message": "You spend $125.00 monthly on subscriptions. Consider sharing accounts or using free alternatives to save up to $50.00.",
        "type": "saving_opportunity",
        "priority": 7,
        "category": "entertainment",
        "amount": 125.00,
        "potential_savings": 50.00,
        "created_at": "2025-04-19T14:30:00"
      }
    ]
  },
  "savings_potential": 350.75
}
      "percentage": 28.5,
      "created_at": "2025-04-19T14:30:00"
    },
    ...
  ],
  "total": 12
}
```

### Get Recommendation Filters

```
GET /recommendations/filters
```

Get available filter options for recommendations.

**Response:**
```json
{
  "types": [
    "high_spending",
    "recurring_charge",
    "saving_opportunity",
    "budget_alert",
    "cashflow_improvement",
    "financial_habit"
  ],
  "categories": [
    "food",
    "transport",
    "utilities",
    "entertainment",
    "shopping"
  ],
  "priority_range": [3, 10]
}
```

### Get Recommendation

```
GET /recommendations/{recommendation_id}
```

Get a single recommendation by ID.

**Response:**
Recommendation object.

### Generate Recommendations

```
POST /recommendations/generate
```

Generate new recommendations based on current transaction data.

**Response:**
List of new recommendations.

### Get Recommendation Report

```
GET /recommendations/report
```

Get a comprehensive recommendation report.

**Response:**
```json
{
  "report_id": "report_20250419143000",
  "generated_at": "2025-04-19T14:30:00",
  "top_recommendations": [
    {
      "id": "12345678-1234-5678-1234-567812345678",
      "message": "Your food spending is 28.5% of total expenses in the last 30 days. Consider ways to reduce this category.",
      "type": "high_spending",
      "priority": 8,
      "category": "food",
      "amount": 850.25,