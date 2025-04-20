# app/models/recommendation/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Optional, Union
import json

class FinancialReportGenerator:
    """
    Generate personalized financial reports based on transaction data and recommendations
    """
    
    def __init__(self):
        """Initialize the financial report generator"""
        self.report_sections = [
            'summary',
            'income_analysis',
            'expense_analysis',
            'category_breakdown',
            'recommendations',
            'forecast'
        ]
    
    def generate_report(self, 
                      transactions_df: pd.DataFrame,
                      recommendations: List[Dict],
                      forecasts: Optional[Dict[str, pd.DataFrame]] = None,
                      user_info: Optional[Dict] = None,
                      output_dir: str = 'app/data/processed/reports',
                      date_col: str = 'transaction_date',
                      amount_col: str = 'amount',
                      category_col: Optional[str] = 'category') -> Dict[str, Any]:
        """
        Generate a complete financial report
        
        Args:
            transactions_df: DataFrame containing transaction data
            recommendations: List of recommendation dictionaries
            forecasts: Dictionary of forecast DataFrames by category (optional)
            user_info: Dictionary containing user information (optional)
            output_dir: Directory to save report files
            date_col: Name of the date column
            amount_col: Name of the amount column
            category_col: Name of the category column (optional)
            
        Returns:
            Dictionary containing report data
        """
        # Make a copy to avoid modifying the original
        df = transactions_df.copy()
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Initialize report data
        report = {
            'report_id': f"financial_report_{timestamp}",
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        # Add user info if provided
        if user_info:
            report['user_info'] = user_info
        
        # Generate each section
        report['sections']['summary'] = self._generate_summary_section(
            df, recommendations, date_col, amount_col, category_col)
        
        report['sections']['income_analysis'] = self._generate_income_section(
            df, date_col, amount_col, category_col)
        
        report['sections']['expense_analysis'] = self._generate_expense_section(
            df, date_col, amount_col, category_col)
        
        if category_col in df.columns:
            report['sections']['category_breakdown'] = self._generate_category_section(
                df, date_col, amount_col, category_col, output_dir)
        
        report['sections']['recommendations'] = self._format_recommendations(
            recommendations, output_dir)
        
        if forecasts:
            report['sections']['forecast'] = self._generate_forecast_section(
                forecasts, df, date_col, amount_col, category_col, output_dir)
        
        # Save full report to JSON
        report_path = os.path.join(output_dir, f"financial_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Financial report saved to {report_path}")
        
        return report
    
    def _generate_summary_section(self, 
                               df: pd.DataFrame,
                               recommendations: List[Dict],
                               date_col: str,
                               amount_col: str,
                               category_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate the summary section of the report
        
        Args:
            df: Transaction DataFrame
            recommendations: List of recommendation dictionaries 
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column (optional)
            
        Returns:
            Dictionary containing summary data
        """
        # Get date range
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        date_range = (end_date - start_date).days
        
        # Calculate totals
        income = df[df[amount_col] > 0][amount_col].sum()
        expenses = abs(df[df[amount_col] < 0][amount_col].sum())
        net_cashflow = income - expenses
        
        # Recent activity (last 30 days)
        last_30_days = df[df[date_col] >= (end_date - timedelta(days=30))]
        recent_income = last_30_days[last_30_days[amount_col] > 0][amount_col].sum()
        recent_expenses = abs(last_30_days[last_30_days[amount_col] < 0][amount_col].sum())
        recent_net = recent_income - recent_expenses
        
        # Get top recommendations (up to 3)
        top_recommendations = sorted(recommendations, key=lambda x: x['priority'], reverse=True)[:3]
        top_rec_messages = [rec['message'] for rec in top_recommendations]
        
        # Calculate monthly averages
        months = max(1, date_range // 30)
        monthly_income_avg = income / months
        monthly_expense_avg = expenses / months
        
        # Calculate savings rate
        savings_rate = (income - expenses) / income * 100 if income > 0 else 0
        
        # Get top categories if available
        top_categories = {}
        if category_col in df.columns:
            # Top expense categories
            expense_df = df[df[amount_col] < 0].copy()
            expense_df[amount_col] = expense_df[amount_col].abs()
            
            if not expense_df.empty:
                category_expenses = expense_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
                top_expense_categories = category_expenses.head(3).to_dict()
                top_categories['expenses'] = {k: float(v) for k, v in top_expense_categories.items()}
            
            # Top income categories
            income_df = df[df[amount_col] > 0]
            if not income_df.empty and category_col in income_df.columns:
                category_income = income_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
                top_income_categories = category_income.head(3).to_dict()
                top_categories['income'] = {k: float(v) for k, v in top_income_categories.items()}
        
        # Create summary data
        summary = {
            'date_range': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': date_range
            },
            'total': {
                'income': float(income),
                'expenses': float(expenses),
                'net_cashflow': float(net_cashflow)
            },
            'monthly_average': {
                'income': float(monthly_income_avg),
                'expenses': float(monthly_expense_avg),
                'net_cashflow': float(monthly_income_avg - monthly_expense_avg)
            },
            'recent_30_days': {
                'income': float(recent_income),
                'expenses': float(recent_expenses),
                'net_cashflow': float(recent_net)
            },
            'savings_rate': float(savings_rate),
            'transaction_count': len(df),
            'top_recommendations': top_rec_messages
        }
        
        if top_categories:
            summary['top_categories'] = top_categories
        
        return summary
    
    def _generate_income_section(self, 
                              df: pd.DataFrame,
                              date_col: str,
                              amount_col: str,
                              category_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate the income analysis section of the report
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column (optional)
            
        Returns:
            Dictionary containing income analysis data
        """
        # Filter for income transactions
        income_df = df[df[amount_col] > 0].copy()
        
        if income_df.empty:
            return {
                'status': 'No income transactions found',
                'total_income': 0,
                'income_sources': [],
                'income_stability': 'unknown'
            }
        
        # Calculate total income
        total_income = income_df[amount_col].sum()
        
        # Extract income sources if category column is available
        income_sources = []
        if category_col in income_df.columns:
            source_totals = income_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
            
            for source, amount in source_totals.items():
                # Skip tiny amounts
                if amount < 0.01 * total_income:
                    continue
                    
                income_sources.append({
                    'source': source,
                    'amount': float(amount),
                    'percentage': float(amount / total_income * 100)
                })
        
        # Analyze income stability
        # Create a time series of monthly income
        income_df['month'] = income_df[date_col].dt.to_period('M')
        monthly_income = income_df.groupby('month')[amount_col].sum()
        
        if len(monthly_income) >= 3:
            # Calculate coefficient of variation (lower is more stable)
            cv = monthly_income.std() / monthly_income.mean() if monthly_income.mean() > 0 else float('inf')
            
            # Determine stability level
            if cv < 0.1:
                stability = 'very stable'
            elif cv < 0.2:
                stability = 'stable'
            elif cv < 0.3:
                stability = 'moderately stable'
            elif cv < 0.5:
                stability = 'variable'
            else:
                stability = 'highly variable'
            
            # Get monthly stats
            monthly_stats = {
                'mean': float(monthly_income.mean()),
                'median': float(monthly_income.median()),
                'min': float(monthly_income.min()),
                'max': float(monthly_income.max()),
                'std_dev': float(monthly_income.std()),
                'coefficient_of_variation': float(cv)
            }
        else:
            stability = 'insufficient data'
            monthly_stats = None
        
        # Build income analysis data
        income_analysis = {
            'total_income': float(total_income),
            'income_sources': income_sources,
            'income_stability': stability,
            'monthly_income_count': len(monthly_income),
        }
        
        if monthly_stats:
            income_analysis['monthly_stats'] = monthly_stats
        
        # Check for income growth trend if enough months
        if len(monthly_income) >= 4:
            # Simple linear regression to detect trend
            months = np.arange(len(monthly_income))
            income_values = monthly_income.values
            
            # Calculate slope and correlation coefficient
            if np.std(months) > 0 and np.std(income_values) > 0:
                correlation = np.corrcoef(months, income_values)[0, 1]
                slope = correlation * (np.std(income_values) / np.std(months))
                
                # Calculate percentage change
                start_value = monthly_income.iloc[0]
                end_value = monthly_income.iloc[-1]
                if start_value > 0:
                    percent_change = (end_value - start_value) / start_value * 100
                    
                    income_analysis['income_trend'] = {
                        'slope': float(slope),
                        'correlation': float(correlation),
                        'percent_change': float(percent_change),
                        'interpretation': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    }
        
        return income_analysis
    
    def _generate_expense_section(self, 
                               df: pd.DataFrame,
                               date_col: str,
                               amount_col: str,
                               category_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate the expense analysis section of the report
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column (optional)
            
        Returns:
            Dictionary containing expense analysis data
        """
        # Filter for expense transactions
        expense_df = df[df[amount_col] < 0].copy()
        expense_df[amount_col] = expense_df[amount_col].abs()  # Convert to positive for easier analysis
        
        if expense_df.empty:
            return {
                'status': 'No expense transactions found',
                'total_expenses': 0,
                'expense_categories': [],
                'expense_stability': 'unknown'
            }
        
        # Calculate total expenses
        total_expenses = expense_df[amount_col].sum()
        
        # Extract expense categories if category column is available
        expense_categories = []
        if category_col in expense_df.columns:
            category_totals = expense_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
            
            for category, amount in category_totals.items():
                # Skip tiny amounts
                if amount < 0.01 * total_expenses:
                    continue
                    
                expense_categories.append({
                    'category': category,
                    'amount': float(amount),
                    'percentage': float(amount / total_expenses * 100)
                })
        
        # Analyze expense stability
        # Create a time series of monthly expenses
        expense_df['month'] = expense_df[date_col].dt.to_period('M')
        monthly_expenses = expense_df.groupby('month')[amount_col].sum()
        
        if len(monthly_expenses) >= 3:
            # Calculate coefficient of variation (lower is more stable)
            cv = monthly_expenses.std() / monthly_expenses.mean() if monthly_expenses.mean() > 0 else float('inf')
            
            # Determine stability level
            if cv < 0.1:
                stability = 'very stable'
            elif cv < 0.2:
                stability = 'stable'
            elif cv < 0.3:
                stability = 'moderately stable'
            elif cv < 0.5:
                stability = 'variable'
            else:
                stability = 'highly variable'
            
            # Get monthly stats
            monthly_stats = {
                'mean': float(monthly_expenses.mean()),
                'median': float(monthly_expenses.median()),
                'min': float(monthly_expenses.min()),
                'max': float(monthly_expenses.max()),
                'std_dev': float(monthly_expenses.std()),
                'coefficient_of_variation': float(cv)
            }
        else:
            stability = 'insufficient data'
            monthly_stats = None
        
        # Build expense analysis data
        expense_analysis = {
            'total_expenses': float(total_expenses),
            'expense_categories': expense_categories,
            'expense_stability': stability,
            'monthly_expense_count': len(monthly_expenses),
        }
        
        if monthly_stats:
            expense_analysis['monthly_stats'] = monthly_stats
        
        # Check for expense growth trend if enough months
        if len(monthly_expenses) >= 4:
            # Simple linear regression to detect trend
            months = np.arange(len(monthly_expenses))
            expense_values = monthly_expenses.values
            
            # Calculate slope and correlation coefficient
            if np.std(months) > 0 and np.std(expense_values) > 0:
                correlation = np.corrcoef(months, expense_values)[0, 1]
                slope = correlation * (np.std(expense_values) / np.std(months))
                
                # Calculate percentage change
                start_value = monthly_expenses.iloc[0]
                end_value = monthly_expenses.iloc[-1]
                if start_value > 0:
                    percent_change = (end_value - start_value) / start_value * 100
                    
                    expense_analysis['expense_trend'] = {
                        'slope': float(slope),
                        'correlation': float(correlation),
                        'percent_change': float(percent_change),
                        'interpretation': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    }
        
        return expense_analysis
    
    def _generate_category_section(self,
                                df: pd.DataFrame,
                                date_col: str,
                                amount_col: str,
                                category_col: str,
                                output_dir: str) -> Dict[str, Any]:
        """
        Generate category breakdown section of the report
        
        Args:
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary containing category breakdown data
        """
        # Prepare visualization directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Filter for expense transactions
        expense_df = df[df[amount_col] < 0].copy()
        expense_df[amount_col] = expense_df[amount_col].abs()  # Convert to positive for easier analysis
        
        # Prepare empty result if no data
        if expense_df.empty:
            return {
                'status': 'No expense transactions found',
                'categories': []
            }
        
        # Calculate totals by category
        category_totals = expense_df.groupby(category_col)[amount_col].sum().sort_values(ascending=False)
        total_expenses = category_totals.sum()
        
        # Calculate percentages
        category_percentages = (category_totals / total_expenses * 100).round(1)
        
        # Convert to list of dictionaries
        categories = []
        for category, amount in category_totals.items():
            percentage = category_percentages[category]
            
            # Skip very small categories (less than 1%)
            if percentage < 1:
                continue
                
            # Perform time analysis for this category
            category_df = expense_df[expense_df[category_col] == category]
            
            # Monthly analysis
            category_df['month'] = category_df[date_col].dt.to_period('M')
            monthly_spending = category_df.groupby('month')[amount_col].sum()
            
            # Calculate month-to-month change if we have at least 2 months
            month_to_month_change = None
            if len(monthly_spending) >= 2:
                last_month = monthly_spending.iloc[-1]
                previous_month = monthly_spending.iloc[-2]
                
                if previous_month > 0:
                    month_to_month_change = (last_month - previous_month) / previous_month * 100
            
            # Calculate average transaction size
            avg_transaction = category_df[amount_col].mean()
            
            # Add category data
            categories.append({
                'category': category,
                'amount': float(amount),
                'percentage': float(percentage),
                'transaction_count': len(category_df),
                'average_transaction': float(avg_transaction),
                'month_to_month_change': float(month_to_month_change) if month_to_month_change is not None else None
            })
        
        # Create pie chart of categories
        plt.figure(figsize=(10, 6))
        categories_to_plot = category_percentages[category_percentages >= 3]  # Only plot categories with at least 3%
        if len(categories_to_plot) < len(category_percentages):
            # Add an "Other" category for the rest
            other_pct = category_percentages[category_percentages < 3].sum()
            categories_to_plot_with_other = pd.Series(
                list(categories_to_plot) + [other_pct],
                index=list(categories_to_plot.index) + ['Other']
            )
            plt.pie(categories_to_plot_with_other, 
                    labels=categories_to_plot_with_other.index, 
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True)
        else:
            plt.pie(categories_to_plot, 
                    labels=categories_to_plot.index, 
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True)
        
        plt.axis('equal')
        plt.title('Expense Categories')
        plt.tight_layout()
        
        # Save pie chart
        pie_chart_path = os.path.join(vis_dir, 'expense_categories_pie.png')
        plt.savefig(pie_chart_path)
        plt.close()
        
        # Return category data with visualization paths
        return {
            'categories': categories,
            'total_expenses': float(total_expenses),
            'visualizations': {
                'pie_chart': os.path.basename(pie_chart_path)
            }
        }
    
    def _format_recommendations(self,
                             recommendations: List[Dict],
                             output_dir: str) -> Dict[str, Any]:
        """
        Format and visualize recommendations
        
        Args:
            recommendations: List of recommendation dictionaries
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary containing formatted recommendations data
        """
        # Prepare visualization directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        if not recommendations:
            return {
                'status': 'No recommendations available',
                'recommendations': []
            }
        
        # Group recommendations by type
        recommendation_groups = {}
        for rec in recommendations:
            rec_type = rec['type']
            
            if rec_type not in recommendation_groups:
                recommendation_groups[rec_type] = []
                
            recommendation_groups[rec_type].append(rec)
        
        # Create type count visualization
        type_counts = {t: len(recs) for t, recs in recommendation_groups.items()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            [t.replace('_', ' ').title() for t in type_counts.keys()],
            type_counts.values()
        )
        plt.title('Recommendation Types')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save type counts chart
        type_counts_path = os.path.join(vis_dir, 'recommendation_types.png')
        plt.savefig(type_counts_path)
        plt.close()
        
        # Create priority visualization
        priorities = [rec['priority'] for rec in recommendations]
        sorted_recs = sorted(recommendations, key=lambda r: r['priority'], reverse=True)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            range(len(sorted_recs)),
            [rec['priority'] for rec in sorted_recs],
            color='skyblue'
        )
        
        # Add recommendation text
        for i, rec in enumerate(sorted_recs):
            plt.text(
                bars[i].get_width() + 0.1,
                bars[i].get_y() + bars[i].get_height()/2,
                rec['message'][:50] + ('...' if len(rec['message']) > 50 else ''),
                va='center',
                fontsize=8
            )
        
        plt.yticks([])  # Hide y-axis labels
        plt.xlabel('Priority Score')
        plt.title('Recommendations by Priority')
        plt.tight_layout()
        
        # Save priority chart
        priority_path = os.path.join(vis_dir, 'recommendation_priorities.png')
        plt.savefig(priority_path)
        plt.close()
        
        # Return formatted recommendations data
        return {
            'recommendation_count': len(recommendations),
            'recommendation_groups': {
                rec_type: [rec['message'] for rec in recs]
                for rec_type, recs in recommendation_groups.items()
            },
            'top_recommendations': [
                {'message': rec['message'], 'priority': rec['priority'], 'type': rec['type']}
                for rec in sorted(recommendations, key=lambda r: r['priority'], reverse=True)[:5]
            ],
            'visualizations': {
                'type_counts': os.path.basename(type_counts_path),
                'priorities': os.path.basename(priority_path)
            }
        }
    
    def _generate_forecast_section(self,
                                forecasts: Dict[str, pd.DataFrame],
                                df: pd.DataFrame,
                                date_col: str,
                                amount_col: str,
                                category_col: Optional[str] = None,
                                output_dir: str = None) -> Dict[str, Any]:
        """
        Generate forecast section of the report
        
        Args:
            forecasts: Dictionary of forecast DataFrames by category
            df: Transaction DataFrame
            date_col: Name of date column
            amount_col: Name of amount column
            category_col: Name of category column (optional)
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary containing forecast data
        """
        # Prepare visualization directory if provided
        vis_dir = None
        if output_dir:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        if not forecasts:
            return {
                'status': 'No forecast data available',
                'forecasts': []
            }
        
        # Process each category forecast
        forecast_data = []
        visualization_paths = {}
        
        for category, forecast_df in forecasts.items():
            # Skip non-expense categories
            if category.lower() in ['income', 'deposit', 'transfer']:
                continue
            
            # Extract forecast values
            if 'yhat' in forecast_df.columns:  # Prophet forecast format
                forecast_values = forecast_df['yhat'].abs()
                lower_bound = forecast_df['yhat_lower'].abs() if 'yhat_lower' in forecast_df.columns else None
                upper_bound = forecast_df['yhat_upper'].abs() if 'yhat_upper' in forecast_df.columns else None
                forecast_dates = forecast_df['ds'] if 'ds' in forecast_df.columns else forecast_df.index
            else:
                forecast_values = forecast_df[amount_col].abs() if amount_col in forecast_df.columns else forecast_df.iloc[:, 0].abs()
                lower_bound = None
                upper_bound = None
                forecast_dates = forecast_df.index
            
            # Calculate total forecasted amount
            total_forecast = forecast_values.sum()
            
            # Get historical data for comparison if category column is available
            historical_comparison = None
            if category_col in df.columns:
                category_data = df[(df[category_col] == category) & (df[amount_col] < 0)]
                
                if len(category_data) > 0:
                    last_date = df[date_col].max()
                    month_ago = last_date - timedelta(days=30)
                    recent_spending = category_data[category_data[date_col] >= month_ago]
                    
                    if len(recent_spending) > 0:
                        current_amount = abs(recent_spending[amount_col].sum())
                        
                        # Calculate percentage change
                        if current_amount > 0:
                            pct_change = ((total_forecast - current_amount) / current_amount * 100).round(1)
                            
                            historical_comparison = {
                                'current_amount': float(current_amount),
                                'forecast_amount': float(total_forecast),
                                'percentage_change': float(pct_change),
                                'direction': 'increase' if pct_change > 0 else 'decrease'
                            }
            
            # Create forecast visualization if directory provided
            if vis_dir:
                plt.figure(figsize=(10, 6))
                
                # Plot forecast
                plt.plot(forecast_dates, forecast_values, 'b-', label='Forecast')
                
                # Plot confidence interval if available
                if lower_bound is not None and upper_bound is not None:
                    plt.fill_between(forecast_dates, lower_bound, upper_bound, color='b', alpha=0.2, label='Confidence Interval')
                
                plt.title(f'{category.title()} Spending Forecast')
                plt.xlabel('Date')
                plt.ylabel('Amount')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save forecast chart
                forecast_path = os.path.join(vis_dir, f'{category}_forecast.png')
                plt.savefig(forecast_path)
                plt.close()
                
                visualization_paths[category] = os.path.basename(forecast_path)
            
            # Add forecast data
            forecast_data.append({
                'category': category,
                'forecast_total': float(total_forecast),
                'forecast_period_days': len(forecast_values),
                'historical_comparison': historical_comparison
            })
        
        # Return forecast section data
        forecast_section = {
            'forecasts': forecast_data
        }
        
        if visualization_paths:
            forecast_section['visualizations'] = visualization_paths
        
        return forecast_section