import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class RecommendationEngine:
    """
    Engine for generating personalized financial recommendations
    """
    def __init__(self, threshold_percentile: int = 75):
        """
        Initialize the recommendation engine
        
        Args:
            threshold_percentile: Percentile threshold for high spending (default: 75)
        """
        self.threshold_percentile = threshold_percentile
        self.recommendation_templates = {
            'high_spending': "You've spent {amount}% more on {category} than usual. Consider limiting this to save ${savings} next month.",
            'recurring_charge': "We noticed a recurring charge of ${amount} for {description}. Do you still use this service?",
            'savings_opportunity': "Based on your spending patterns, you could save ${amount} on {category} by {action}.",
            'spending_trend': "Your spending on {category} has increased by {percentage}% over the last {time_period}.",
            'better_timing': "Consider purchasing {category} items in {month} when prices are typically {percentage}% lower.",
            'similar_users': "Users with similar profiles save an average of ${amount} on {category} by {action}."
        }
        
        # Define potential savings actions for each category
        self.savings_actions = {
            'food': [
                "cooking at home more often",
                "using grocery coupons",
                "meal planning",
                "reducing takeout orders"
            ],
            'transport': [
                "using public transportation",
                "carpooling",
                "planning trips more efficiently",
                "comparing gas prices"
            ],
            'shopping': [
                "comparing prices before purchasing",
                "waiting for sales",
                "creating a shopping list and sticking to it",
                "using cashback apps"
            ],
            'utilities': [
                "reducing energy consumption",
                "comparing service providers",
                "negotiating your bills",
                "using programmable thermostats"
            ],
            'entertainment': [
                "sharing subscriptions with family",
                "using free alternatives",
                "looking for discount codes",
                "reducing unused subscriptions"
            ],
            'health': [
                "comparing pharmacy prices",
                "using generic medications",
                "preventive care",
                "using in-network providers"
            ],
            'housing': [
                "refinancing your mortgage",
                "negotiating rent",
                "comparing insurance rates",
                "reducing utility costs"
            ],
            'other': [
                "creating a budget",
                "reviewing expenses regularly",
                "avoiding impulse purchases",
                "setting financial goals"
            ]
        }
    
    def analyze_spending_patterns(self, df: pd.DataFrame,
                                 date_col: str = 'transaction_date',
                                 amount_col: str = 'amount',
                                 category_col: str = 'category') -> Dict[str, Any]:
        """
        Identify spending patterns and anomalies
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories
            
        Returns:
            Dictionary of category statistics
        """
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows with missing dates
        df = df.dropna(subset=[date_col])
        
        # Create month column for aggregation
        df['month'] = df[date_col].dt.to_period('M')
        
        # Handle different data formats for spending amount
        if amount_col in df.columns:
            # If amount column exists, use negative values as expenses
            if df[amount_col].min() < 0:
                df_expenses = df[df[amount_col] < 0].copy()
                df_expenses['spending'] = df_expenses[amount_col].abs()
            else:
                # If no negative values, try withdrawal column or use as is
                if 'withdrawal' in df.columns:
                    df_expenses = df[df['withdrawal'] > 0].copy()
                    df_expenses['spending'] = df_expenses['withdrawal']
                else:
                    # Assume all transactions are expenses
                    df_expenses = df.copy()
                    df_expenses['spending'] = df_expenses[amount_col]
        elif 'withdrawal' in df.columns:
            df_expenses = df[df['withdrawal'] > 0].copy()
            df_expenses['spending'] = df_expenses['withdrawal']
        else:
            raise ValueError("Cannot identify expense amounts in the data")
        
        # If no category column, use 'other' as default
        if category_col not in df_expenses.columns:
            df_expenses[category_col] = 'other'
        
        # Group by month and category
        monthly_spending = df_expenses.groupby(['month', category_col])['spending'].sum().reset_index()
        
        # Calculate statistics for each category
        category_stats = {}
        
        for category in df_expenses[category_col].unique():
            category_data = monthly_spending[monthly_spending[category_col] == category]
            
            if len(category_data) > 0:
                avg_spending = category_data['spending'].mean()
                median_spending = category_data['spending'].median()
                threshold = category_data['spending'].quantile(self.threshold_percentile / 100)
                
                # Get recent months
                recent_months = sorted(df_expenses['month'].unique())[-3:]  # Last 3 months
                recent_data = category_data[category_data['month'].isin(recent_months)]
                
                recent_spending = recent_data['spending'].mean() if len(recent_data) > 0 else 0
                
                category_stats[category] = {
                    'avg_spending': avg_spending,
                    'median_spending': median_spending,
                    'threshold': threshold,
                    'recent_spending': recent_spending,
                    'monthly_data': category_data,
                    'months_count': len(category_data),
                    'recent_trend': recent_spending / avg_spending if avg_spending > 0 else 1
                }
        
        return category_stats
    
    def identify_recurring_charges(self, df: pd.DataFrame,
                                  date_col: str = 'transaction_date',
                                  desc_col: str = 'description',
                                  amount_col: str = 'amount',
                                  min_occurrences: int = 3,
                                  max_variance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Identify recurring charges in transactions
        
        Args:
            df: DataFrame containing transaction data
            date_col: Column name for transaction dates
            desc_col: Column name for transaction descriptions
            amount_col: Column name for transaction amounts
            min_occurrences: Minimum occurrences to consider recurring
            max_variance: Maximum variance in amount to consider recurring
            
        Returns:
            List of recurring charges
        """
        # Ensure the required columns exist
        required_cols = [date_col, desc_col]
        if amount_col in df.columns:
            required_cols.append(amount_col)
        elif 'withdrawal' in df.columns:
            amount_col = 'withdrawal'
            required_cols.append(amount_col)
        else:
            raise ValueError("Cannot identify amount column in the data")
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Group by description
        recurring_charges = []
        
        # Use description groups to identify recurring charges
        for desc, group in df.groupby(desc_col):
            if len(group) >= min_occurrences:
                # Check if amounts are consistent
                if amount_col in df.columns:
                    amounts = group[amount_col]
                    if amount_col == 'amount':
                        # If using 'amount', check for consistent negative values
                        amounts = amounts[amounts < 0].abs()
                    
                    if len(amounts) >= min_occurrences:
                        mean_amount = amounts.mean()
                        variance = amounts.std() / mean_amount if mean_amount > 0 else float('inf')
                        
                        if variance <= max_variance:
                            # Check if timing is consistent
                            dates = sorted(group[date_col].dropna())
                            
                            if len(dates) >= min_occurrences:
                                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                                
                                if intervals and max(intervals) - min(intervals) <= 5:
                                    # This is likely a recurring charge
                                    recurring_charges.append({
                                        'description': desc,
                                        'amount': mean_amount,
                                        'frequency_days': sum(intervals) / len(intervals),
                                        'occurrences': len(group),
                                        'last_date': dates[-1],
                                        'annual_cost': mean_amount * (365 / (sum(intervals) / len(intervals)))
                                    })
        
        return recurring_charges
    
    def generate_recommendations(self, df: pd.DataFrame,
                               forecasts: Optional[Dict[str, pd.DataFrame]] = None,
                               date_col: str = 'transaction_date',
                               amount_col: str = 'amount',
                               category_col: str = 'category',
                               desc_col: str = 'description',
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Generate personalized savings recommendations
        
        Args:
            df: DataFrame containing transaction data
            forecasts: Dictionary of category-specific forecasts (optional)
            date_col: Column name for transaction dates
            amount_col: Column name for transaction amounts
            category_col: Column name for transaction categories
            desc_col: Column name for transaction descriptions
            limit: Maximum number of recommendations to generate
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze spending patterns
        try:
            category_stats = self.analyze_spending_patterns(
                df, 
                date_col=date_col,
                amount_col=amount_col,
                category_col=category_col
            )
            print(f"Analyzed spending patterns for {len(category_stats)} categories")
        except Exception as e:
            print(f"Error analyzing spending patterns: {str(e)}")
            category_stats = {}
        
        # Identify recurring charges
        try:
            recurring_charges = self.identify_recurring_charges(
                df,
                date_col=date_col,
                desc_col=desc_col,
                amount_col=amount_col
            )
            print(f"Identified {len(recurring_charges)} recurring charges")
        except Exception as e:
            print(f"Error identifying recurring charges: {str(e)}")
            recurring_charges = []
        
        # 1. High spending categories
        for category, stats in category_stats.items():
            if stats['recent_spending'] > stats['threshold'] and stats['months_count'] >= 3:
                percent_over = ((stats['recent_spending'] - stats['avg_spending']) / stats['avg_spending']) * 100
                potential_savings = stats['recent_spending'] - stats['avg_spending']
                
                if percent_over > 10 and potential_savings > 5:  # Only significant differences
                    recommendations.append({
                        'type': 'high_spending',
                        'category': category,
                        'amount': round(percent_over, 1),
                        'savings': round(potential_savings, 2),
                        'message': self.recommendation_templates['high_spending'].format(
                            amount=round(percent_over, 1),
                            category=category,
                            savings=round(potential_savings, 2)
                        ),
                        'priority': min(100, int(percent_over))
                    })
        
        # 2. Recurring charges optimization
        for charge in recurring_charges:
            if charge['annual_cost'] > 100:  # Only significant charges
                recommendations.append({
                    'type': 'recurring_charge',
                    'description': charge['description'],
                    'amount': round(charge['amount'], 2),
                    'annual_cost': round(charge['annual_cost'], 2),
                    'message': self.recommendation_templates['recurring_charge'].format(
                        amount=round(charge['amount'], 2),
                        description=charge['description']
                    ),
                    'priority': min(70, int(charge['annual_cost'] / 10))  # Higher for expensive subscriptions
                })
        
        # 3. Category-specific savings opportunities
        for category, stats in category_stats.items():
            if stats['avg_spending'] > 100 and category in self.savings_actions:
                potential_savings = stats['avg_spending'] * 0.15  # Assume 15% potential savings
                action = np.random.choice(self.savings_actions[category])
                
                recommendations.append({
                    'type': 'savings_opportunity',
                    'category': category,
                    'amount': round(potential_savings, 2),
                    'action': action,
                    'message': self.recommendation_templates['savings_opportunity'].format(
                        amount=round(potential_savings, 2),
                        category=category,
                        action=action
                    ),
                    'priority': min(60, int(potential_savings / 5))
                })
        
        # 4. Forecasted increases (if forecasts provided)
        if forecasts:
            for category, forecast_df in forecasts.items():
                if category in category_stats:
                    # Get average forecasted amount
                    avg_forecast = forecast_df['amount'].mean()
                    
                    # Compare with historical average
                    historical_avg = category_stats[category]['avg_spending']
                    
                    if avg_forecast > historical_avg * 1.2:  # 20% increase
                        percent_increase = ((avg_forecast - historical_avg) / historical_avg) * 100
                        
                        recommendations.append({
                            'type': 'spending_trend',
                            'category': category,
                            'percentage': round(percent_increase, 1),
                            'time_period': 'next month',
                            'message': self.recommendation_templates['spending_trend'].format(
                                category=category,
                                percentage=round(percent_increase, 1),
                                time_period='next month'
                            ),
                            'priority': min(80, int(percent_increase))
                        })
        
        # Sort by priority and limit
        recommendations.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return recommendations[:limit]
    
    def visualize_recommendations(self, recommendations: List[Dict[str, Any]],
                                output_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize recommendations by type and priority
        
        Args:
            recommendations: List of recommendations
            output_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        if not recommendations:
            print("No recommendations to visualize")
            return None
        
        # Group by type
        types = [rec['type'] for rec in recommendations]
        priorities = [rec.get('priority', 50) for rec in recommendations]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        bars = ax.barh(types, priorities, color='skyblue')
        
        # Add values
        for i, v in enumerate(priorities):
            ax.text(v + 1, i, str(v), va='center')
        
        # Set title and labels
        ax.set_title('Recommendation Priorities by Type')
        ax.set_xlabel('Priority Score')
        ax.set_ylabel('Recommendation Type')
        
        # Tight layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")

        
        return fig
    
# app/models/recommendation/recommendation_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import os

class RecommendationEngine:
    """
    Engine for generating personalized financial recommendations based on
    transaction history and spending patterns.
    """
    
    def __init__(self, threshold_percentile: float = 75):
        """
        Initialize the recommendation engine
        
        Args:
            threshold_percentile: Percentile threshold for flagging unusual spending (default: 75)
        """
        self.threshold_percentile = threshold_percentile
        
        # Define recommendation types
        self.recommendation_types = {
            'high_spending': 'Reduce spending in high-expense categories',
            'recurring_charge': 'Review recurring subscriptions and services',
            'saving_opportunity': 'Potential saving opportunity identified',
            'budget_alert': 'Budget threshold exceeded',
            'income_opportunity': 'Potential income opportunity',
            'spending_pattern': 'Unusual spending pattern detected',
            'cashflow_improvement': 'Opportunity to improve cash flow',
            'financial_habit': 'Develop better financial habits'
        }
    
    def generate_recommendations(self, 
                                transactions_df: pd.DataFrame,
                                forecasts: Optional[Dict[str, pd.DataFrame]] = None,
                                date_col: str = 'transaction_date',
                                amount_col: str = 'amount',
                                category_col: Optional[str] = 'category',
                                desc_col: Optional[str] = 'description',
                                limit: int = 5) -> List[Dict]:
        """
        Generate personalized financial recommendations based on transaction history
        
        Args:
            transactions_df: DataFrame containing transaction data
            forecasts: Dictionary of forecast DataFrames by category (optional)
            date_col: Name of the date column
            amount_col: Name of the amount column 
            category_col: Name of the category column (optional)
            desc_col: Name of the description column (optional)
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        # Make a copy to avoid modifying the original
        df = transactions_df.copy()
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # List to store recommendations
        recommendations = []
        
        # Get recommendations from each analyzer
        if category_col in df.columns:
            # 1. Analyze category spending
            category_recs = self._analyze_category_spending(df, date_col, amount_col, category_col)
            recommendations.extend(category_recs)
            
            # 2. Analyze spending trends
            if forecasts:
                trend_recs = self._analyze_spending_trends(df, forecasts, date_col, amount_col, category_col)
                recommendations.extend(trend_recs)
        
        # 3. Detect recurring charges
        if desc_col in df.columns:
            recurring_recs = self._detect_recurring_charges(df, date_col, amount_col, desc_col)
            recommendations.extend(recurring_recs)
        
        # 4. Analyze cash flow
        cashflow_recs = self._analyze_cash_flow(df, date_col, amount_col)
        recommendations.extend(cashflow_recs)
        
        # 5. Generate savings recommendations
        savings_recs = self._generate_saving_recommendations(df, date_col, amount_col, category_col if category_col in df.columns else None)
        recommendations.extend(savings_recs)
        
        # Sort recommendations by priority (higher is more important)
        recommendations = sorted(recommendations, key=lambda x: x['priority'], reverse=True)
        
        # Limit the number of recommendations if specified
        if limit and limit > 0:
            recommendations = recommendations[:limit]
            
        return recommendations
    
    def _analyze_category_spending(self, 
                                 df: pd.DataFrame, 
                                 date_col: str,
                                 amount_col: str, 
                                 category_col: str) -> List[Dict]:
        """
        Analyze spending by category to identify high-spend areas
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Filter for expenses (negative amounts)
        expenses_df = df[df[amount_col] < 0].copy()
        expenses_df[amount_col] = expenses_df[amount_col].abs()  # Convert to positive for easier analysis
        
        # Skip if no expenses
        if len(expenses_df) == 0:
            return recommendations
        
        # Get recent transactions (last 30 days)
        last_date = expenses_df[date_col].max()
        start_date = last_date - timedelta(days=30)
        recent_expenses = expenses_df[expenses_df[date_col] >= start_date]
        
        # Skip if no recent expenses
        if len(recent_expenses) == 0:
            return recommendations
        
        # Group by category
        category_spending = recent_expenses.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
        
        # Calculate percentage of total spending
        total_spending = category_spending.sum()
        category_percentage = (category_spending / total_spending * 100).round(1)
        
        # Get top spending categories (excluding income or positive categories)
        for category, percentage in category_percentage.items():
            if percentage >= 20 and category.lower() not in ['income', 'deposit', 'transfer']:
                recommendations.append({
                    'type': 'high_spending',
                    'category': category,
                    'message': f"Your {category} spending is {percentage}% of total expenses in the last 30 days. Consider ways to reduce this category.",
                    'amount': float(category_spending[category]),
                    'percentage': float(percentage),
                    'priority': min(int(percentage / 5), 10)  # Scale priority based on percentage, max 10
                })
        
        # Compare current spending with historical average
        if len(expenses_df) > len(recent_expenses):
            # Get previous period for comparison
            older_start = start_date - timedelta(days=30)
            older_expenses = expenses_df[(expenses_df[date_col] >= older_start) & (expenses_df[date_col] < start_date)]
            
            if len(older_expenses) > 0:
                older_spending = older_expenses.groupby(category_col)[amount_col].sum()
                
                # Find categories with significant increases
                for category in category_spending.index:
                    if category in older_spending and older_spending[category] > 0:
                        current = category_spending[category]
                        previous = older_spending[category]
                        
                        percentage_increase = ((current - previous) / previous * 100).round(1)
                        
                        if percentage_increase >= 25 and current >= 100:  # Significant increase and meaningful amount
                            recommendations.append({
                                'type': 'spending_pattern',
                                'category': category,
                                'message': f"Your {category} spending increased by {percentage_increase}% compared to the previous month. Consider reviewing this category.",
                                'current_amount': float(current),
                                'previous_amount': float(previous),
                                'percentage_change': float(percentage_increase),
                                'priority': min(int(percentage_increase / 10), 9)  # Scale priority based on increase
                            })
        
        return recommendations
    
    def _detect_recurring_charges(self, 
                               df: pd.DataFrame, 
                               date_col: str,
                               amount_col: str, 
                               desc_col: str) -> List[Dict]:
        """
        Detect potential recurring charges (subscriptions, etc.)
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            desc_col: Name of description column
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Filter for expenses (negative amounts)
        expenses_df = df[df[amount_col] < 0].copy()
        expenses_df[amount_col] = expenses_df[amount_col].abs()  # Convert to positive for easier analysis
        
        # Skip if not enough data
        if len(expenses_df) < 10:
            return recommendations
        
        # Group by description
        desc_groups = expenses_df.groupby(desc_col)
        
        # Keywords suggesting subscriptions
        subscription_keywords = [
            'subscription', 'monthly', 'recurring', 'netflix', 'spotify', 
            'hulu', 'amazon prime', 'disney+', 'membership', 'gym'
        ]
        
        # Find potential recurring charges
        recurring_charges = []
        
        for desc, group in desc_groups:
            # Skip if less than 2 transactions
            if len(group) < 2:
                continue
                
            # Check if it's a potential subscription based on keywords
            is_subscription = any(keyword in str(desc).lower() for keyword in subscription_keywords)
            
            # Check if amount is consistent
            amounts = group[amount_col].unique()
            consistent_amount = len(amounts) == 1 or (len(amounts) == 2 and abs(amounts[0] - amounts[1]) / amounts[0] < 0.05)
            
            # Check if dates suggest monthly charges
            dates = sorted(group[date_col])
            date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            
            # Check if any gaps are between 28-32 days (monthly) or 13-15 days (bi-weekly)
            monthly_pattern = any(28 <= diff <= 32 for diff in date_diffs)
            biweekly_pattern = any(13 <= diff <= 15 for diff in date_diffs)
            
            if (is_subscription or (consistent_amount and (monthly_pattern or biweekly_pattern))) and group[amount_col].mean() > 5:
                avg_amount = group[amount_col].mean()
                
                recurring_charges.append({
                    'description': desc,
                    'amount': avg_amount,
                    'transactions': len(group),
                    'most_recent': group[date_col].max()
                })
        
        # Sort by amount (highest first)
        recurring_charges = sorted(recurring_charges, key=lambda x: x['amount'], reverse=True)
        
        # Generate recommendations for the top recurring charges
        for i, charge in enumerate(recurring_charges[:5]):  # Limit to top 5
            recommendations.append({
                'type': 'recurring_charge',
                'description': charge['description'],
                'message': f"You have a recurring charge of ${charge['amount']:.2f} for {charge['description']}. Review if this subscription is still valuable to you.",
                'amount': float(charge['amount']),
                'transactions': charge['transactions'],
                'priority': min(8, 3 + i)  # Priority based on amount ranking, max 8
            })
        
        # If they have many subscriptions, add a general recommendation
        if len(recurring_charges) >= 5:
            total_subscription_amount = sum(charge['amount'] for charge in recurring_charges)
            recommendations.append({
                'type': 'saving_opportunity',
                'message': f"You have at least {len(recurring_charges)} subscription services totaling ${total_subscription_amount:.2f} per month. Consider reviewing which ones you actually use.",
                'amount': float(total_subscription_amount),
                'priority': 9
            })
        
        return recommendations
    
    def _analyze_cash_flow(self, 
                         df: pd.DataFrame, 
                         date_col: str,
                         amount_col: str) -> List[Dict]:
        """
        Analyze cash flow patterns to identify improvements
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Skip if not enough data
        if len(df) < 20:
            return recommendations
        
        # Get recent transactions (last 90 days)
        last_date = df[date_col].max()
        start_date = last_date - timedelta(days=90)
        recent_df = df[df[date_col] >= start_date]
        
        # Skip if not enough recent data
        if len(recent_df) < 10:
            return recommendations
        
        # Aggregate by month
        recent_df['month'] = recent_df[date_col].dt.to_period('M')
        monthly_flow = recent_df.groupby('month')[amount_col].sum()
        
        # Check for negative months
        negative_months = sum(1 for flow in monthly_flow if flow < 0)
        
        if negative_months == len(monthly_flow) and len(monthly_flow) >= 2:
            # Consistently negative cash flow
            avg_monthly_deficit = abs(monthly_flow.mean())
            recommendations.append({
                'type': 'cashflow_improvement',
                'message': f"You've had negative cash flow for {negative_months} consecutive months, averaging -${avg_monthly_deficit:.2f} per month. Consider ways to increase income or reduce expenses.",
                'amount': float(avg_monthly_deficit),
                'priority': 10  # Highest priority
            })
        elif negative_months >= 1:
            # Some negative months
            avg_deficit = abs(monthly_flow[monthly_flow < 0].mean())
            recommendations.append({
                'type': 'cashflow_improvement',
                'message': f"You had negative cash flow in {negative_months} of the last {len(monthly_flow)} months. Try to build a buffer for these periods.",
                'amount': float(avg_deficit) if not np.isnan(avg_deficit) else 0,
                'priority': 7
            })
        
        # Check for income timing vs. major expenses
        incomes = recent_df[recent_df[amount_col] > 0]
        expenses = recent_df[recent_df[amount_col] < 0]
        
        if len(incomes) > 0 and len(expenses) > 0:
            # Check if large expenses often occur soon after income
            incomes = incomes.sort_values(date_col)
            
            pattern_detected = False
            for _, income_row in incomes.iterrows():
                income_date = income_row[date_col]
                income_amount = income_row[amount_col]
                
                # Look for large expenses within 3 days after income
                next_3_days = income_date + timedelta(days=3)
                quick_expenses = expenses[(expenses[date_col] > income_date) & 
                                        (expenses[date_col] <= next_3_days)]
                
                total_quick_spending = abs(quick_expenses[amount_col].sum())
                
                # If more than 40% of income is spent quickly, flag it
                if total_quick_spending > 0.4 * income_amount and income_amount > 100:
                    pattern_detected = True
                    quick_spending_pct = (total_quick_spending / income_amount * 100).round(1)
                    break
            
            if pattern_detected:
                recommendations.append({
                    'type': 'financial_habit',
                    'message': f"You tend to spend {quick_spending_pct}% of your income within a few days of receiving it. Consider allocating funds to savings immediately when you get paid.",
                    'percentage': float(quick_spending_pct) if 'quick_spending_pct' in locals() else 0,
                    'priority': 8
                })
        
        return recommendations
    
    def _analyze_spending_trends(self, 
                               df: pd.DataFrame,
                               forecasts: Dict[str, pd.DataFrame],
                               date_col: str,
                               amount_col: str,
                               category_col: str) -> List[Dict]:
        """
        Analyze spending trends using forecast data
        
        Args:
            df: Transaction DataFrame
            forecasts: Dictionary of forecast DataFrames by category
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for categories that are projected to increase significantly
        for category, forecast_df in forecasts.items():
            # Skip non-expense categories
            if category.lower() in ['income', 'deposit', 'transfer']:
                continue
            
            # Get historical data for comparison
            category_data = df[df[category_col] == category]
            
            # Skip if not enough data
            if len(category_data) < 5 or len(forecast_df) < 5:
                continue
            
            # Calculate current average monthly spending
            last_date = df[date_col].max()
            month_ago = last_date - timedelta(days=30)
            recent_spending = category_data[category_data[date_col] >= month_ago]
            
            if len(recent_spending) == 0:
                continue
                
            current_amount = abs(recent_spending[amount_col].sum())
            
            # Get projected next month spending
            if 'yhat' in forecast_df.columns:  # Prophet forecast format
                forecast_amount = abs(forecast_df['yhat'].sum())
            else:
                forecast_amount = abs(forecast_df[amount_col].sum())
            
            # Calculate percentage increase
            if current_amount > 0:
                pct_increase = ((forecast_amount - current_amount) / current_amount * 100).round(1)
                
                # Flag significant increases
                if pct_increase > 15 and forecast_amount > 100:
                    recommendations.append({
                        'type': 'budget_alert',
                        'category': category,
                        'message': f"Your {category} spending is projected to increase by {pct_increase}% next month. Consider setting a budget for this category.",
                        'current_amount': float(current_amount),
                        'projected_amount': float(forecast_amount),
                        'percentage_increase': float(pct_increase),
                        'priority': min(int(pct_increase / 5), 9)  # Scale priority based on increase, max 9
                    })
        
        return recommendations
    
    def _generate_saving_recommendations(self, 
                                      df: pd.DataFrame,
                                      date_col: str,
                                      amount_col: str,
                                      category_col: Optional[str] = None) -> List[Dict]:
        """
        Generate general saving recommendations based on spending patterns
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column (optional)
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Filter for expenses (negative amounts)
        expenses_df = df[df[amount_col] < 0].copy()
        expenses_df[amount_col] = expenses_df[amount_col].abs()  # Convert to positive for easier analysis
        
        # Skip if not enough data
        if len(expenses_df) < 10:
            return recommendations
        
        # Get recent transactions (last 90 days)
        last_date = expenses_df[date_col].max()
        start_date = last_date - timedelta(days=90)
        recent_expenses = expenses_df[expenses_df[date_col] >= start_date]
        
        # Skip if not enough recent data
        if len(recent_expenses) < 5:
            return recommendations
        
        # Calculate total monthly expenses
        recent_expenses['month'] = recent_expenses[date_col].dt.to_period('M')
        monthly_expenses = recent_expenses.groupby('month')[amount_col].sum()
        avg_monthly_expense = monthly_expenses.mean()
        
        # Check if there are any income transactions
        income_df = df[df[amount_col] > 0]
        
        if len(income_df) > 0:
            # Calculate average monthly income
            income_df['month'] = income_df[date_col].dt.to_period('M')
            monthly_income = income_df.groupby('month')[amount_col].sum()
            avg_monthly_income = monthly_income.mean()
            
            # Calculate savings rate
            if avg_monthly_income > 0:
                savings_rate = ((avg_monthly_income - avg_monthly_expense) / avg_monthly_income * 100).round(1)
                
                if savings_rate < 10:
                    # Low savings rate
                    target_savings = avg_monthly_income * 0.2  # 20% target
                    amount_to_reduce = target_savings - (avg_monthly_income - avg_monthly_expense)
                    
                    recommendations.append({
                        'type': 'saving_opportunity',
                        'message': f"Your current savings rate is {max(0, savings_rate):.1f}%. Aim to save at least 20% of your income by reducing monthly expenses by ${amount_to_reduce:.2f}.",
                        'current_savings_rate': float(max(0, savings_rate)),
                        'target_reduction': float(amount_to_reduce),
                        'priority': 8
                    })
                elif savings_rate < 20:
                    # Moderate savings rate
                    target_savings = avg_monthly_income * 0.2  # 20% target
                    amount_to_reduce = target_savings - (avg_monthly_income - avg_monthly_expense)
                    
                    recommendations.append({
                        'type': 'saving_opportunity',
                        'message': f"You're currently saving {savings_rate:.1f}% of your income. Consider increasing to 20% by reducing expenses by ${amount_to_reduce:.2f} per month.",
                        'current_savings_rate': float(savings_rate),
                        'target_reduction': float(amount_to_reduce),
                        'priority': 6
                    })
        
        # Generate category-specific saving tips if category data is available
        if category_col in expenses_df.columns:
            category_spending = recent_expenses.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
            
            # Food saving tip
            if 'food' in category_spending or 'dining' in category_spending or 'restaurant' in category_spending:
                food_categories = [cat for cat in category_spending.index if cat.lower() in ['food', 'dining', 'restaurant', 'groceries']]
                
                if food_categories:
                    food_spending = sum(category_spending[cat] for cat in food_categories)
                    monthly_food = food_spending / len(monthly_expenses)
                    
                    if monthly_food > 400:  # Threshold for high food spending
                        potential_savings = monthly_food * 0.3  # 30% reduction target
                        
                        recommendations.append({
                            'type': 'saving_opportunity',
                            'category': 'food',
                            'message': f"You spend ${monthly_food:.2f} monthly on food. Meal planning and cooking at home could save you up to ${potential_savings:.2f} per month.",
                            'monthly_amount': float(monthly_food),
                            'potential_savings': float(potential_savings),
                            'priority': 7
                        })
            
            # Transportation saving tip
            if 'transport' in category_spending or 'transportation' in category_spending or 'uber' in category_spending or 'lyft' in category_spending:
                transport_categories = [cat for cat in category_spending.index if cat.lower() in ['transport', 'transportation', 'uber', 'lyft', 'taxi', 'car']]
                
                if transport_categories:
                    transport_spending = sum(category_spending[cat] for cat in transport_categories)
                    monthly_transport = transport_spending / len(monthly_expenses)
                    
                    if monthly_transport > 300:  # Threshold for high transport spending
                        potential_savings = monthly_transport * 0.25  # 25% reduction target
                        
                        recommendations.append({
                            'type': 'saving_opportunity',
                            'category': 'transportation',
                            'message': f"Your transportation costs are ${monthly_transport:.2f} per month. Consider carpooling, public transit, or combining trips to save up to ${potential_savings:.2f} monthly.",
                            'monthly_amount': float(monthly_transport),
                            'potential_savings': float(potential_savings),
                            'priority': 6
                        })
            
            # Subscription saving tip
            if 'entertainment' in category_spending or 'subscription' in category_spending:
                entertainment_categories = [cat for cat in category_spending.index if cat.lower() in ['entertainment', 'subscription', 'streaming']]
                
                if entertainment_categories:
                    entertainment_spending = sum(category_spending[cat] for cat in entertainment_categories)
                    monthly_entertainment = entertainment_spending / len(monthly_expenses)
                    
                    if monthly_entertainment > 100:  # Threshold for high entertainment spending
                        potential_savings = monthly_entertainment * 0.4  # 40% reduction target
                        
                        recommendations.append({
                            'type': 'saving_opportunity',
                            'category': 'entertainment',
                            'message': f"You spend ${monthly_entertainment:.2f} monthly on entertainment and subscriptions. Consider sharing accounts or using free alternatives to save up to ${potential_savings:.2f}.",
                            'monthly_amount': float(monthly_entertainment),
                            'potential_savings': float(potential_savings),
                            'priority': 5
                        })
        
        # Add a general saving tip if no category-specific ones were added
        if not any(rec['type'] == 'saving_opportunity' for rec in recommendations):
            recommendations.append({
                'type': 'saving_opportunity',
                'message': f"Consider the 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings. Your current monthly expenses are ${avg_monthly_expense:.2f}.",
                'monthly_expenses': float(avg_monthly_expense),
                'priority': 4
            })
        
        return recommendations
    
    def visualize_recommendations(self, recommendations: List[Dict], output_path: Optional[str] = None) -> None:
        """
        Create visualization of recommendations by priority and type
        
        Args:
            recommendations: List of recommendation dictionaries
            output_path: Path to save the visualization image (optional)
        """
        if not recommendations:
            print("No recommendations to visualize")
            return
        
        # Extract data for plotting
        types = [rec['type'] for rec in recommendations]
        priorities = [rec['priority'] for rec in recommendations]
        descriptions = [rec['message'][:50] + '...' if len(rec['message']) > 50 else rec['message'] for rec in recommendations]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(priorities)), priorities, color='skyblue')
        
        # Add recommendation text
        for i, (bar, desc) in enumerate(zip(bars, descriptions)):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    desc, va='center', fontsize=10)
        
        # Add recommendation type labels
        for i, rec_type in enumerate(types):
            plt.text(0.5, i, rec_type.replace('_', ' ').title(), 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Set labels and title
        plt.xlabel('Priority Score')
        plt.title('Financial Recommendations by Priority')
        plt.yticks(range(len(priorities)), [''] * len(priorities))  # Hide y labels
        plt.xlim(0, 11)  # Priority scale
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            print(f"Recommendation visualization saved to {output_path}")
        else:
            plt.show()