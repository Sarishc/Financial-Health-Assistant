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