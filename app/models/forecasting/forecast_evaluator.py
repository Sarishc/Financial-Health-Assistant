import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

from app.models.forecasting.base_forecaster import BaseForecaster

class ForecastEvaluator:
    """
    Evaluates and compares forecasting models.
    Provides metrics and visualizations for forecast evaluation.
    """
    
    def __init__(self, output_dir: str = 'notebooks/visualizations/forecasts'):
        """
        Initialize the forecast evaluator.
        
        Args:
            output_dir: Directory to save evaluation results and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = ['mae', 'rmse', 'mape']
        self.results = {}
        
    def evaluate_forecaster(self, 
                          forecaster: BaseForecaster, 
                          test_data: Dict[str, pd.DataFrame],
                          metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a forecaster on test data.
        
        Args:
            forecaster: Forecaster to evaluate
            test_data: Dictionary mapping category names to test DataFrames
            metrics: List of metrics to calculate (default: ['mae', 'rmse', 'mape'])
            
        Returns:
            Dictionary of evaluation metrics for each category
        """
        if not forecaster.is_fitted:
            raise ValueError(f"Forecaster '{forecaster.name}' must be fitted before evaluation")
            
        if metrics is None:
            metrics = self.metrics
            
        # Evaluate the forecaster
        evaluation_results = forecaster.evaluate(test_data, metrics=metrics)
        
        # Store results
        self.results[forecaster.name] = evaluation_results
        
        return evaluation_results
    
    def compare_forecasters(self, 
                          forecasters: List[BaseForecaster], 
                          test_data: Dict[str, pd.DataFrame],
                          metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare multiple forecasters on test data.
        
        Args:
            forecasters: List of forecasters to compare
            test_data: Dictionary mapping category names to test DataFrames
            metrics: List of metrics to calculate (default: ['mae', 'rmse', 'mape'])
            
        Returns:
            Dictionary mapping forecaster names to evaluation results
        """
        if metrics is None:
            metrics = self.metrics
            
        # Evaluate each forecaster
        for forecaster in forecasters:
            self.evaluate_forecaster(forecaster, test_data, metrics=metrics)
            
        return self.results
    
    def get_best_forecaster(self, 
                          metric: str = 'rmse', 
                          category: Optional[str] = None,
                          lower_is_better: bool = True) -> Dict[str, str]:
        """
        Get the best forecaster for each category based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            category: Specific category to get best forecaster for (if None, get for all categories)
            lower_is_better: Whether lower metric values are better (default: True)
            
        Returns:
            Dictionary mapping category names to best forecaster names
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_forecaster() or compare_forecasters() first.")
            
        best_forecasters = {}
        
        # Get all categories
        all_categories = set()
        for forecaster_results in self.results.values():
            all_categories.update(forecaster_results.keys())
            
        # Filter categories if specified
        categories = [category] if category else all_categories
        
        # Find best forecaster for each category
        for cat in categories:
            # Collect metric values for each forecaster
            metric_values = {}
            
            for forecaster_name, forecaster_results in self.results.items():
                if cat in forecaster_results and metric in forecaster_results[cat]:
                    metric_values[forecaster_name] = forecaster_results[cat][metric]
            
            if not metric_values:
                continue
                
            # Find best forecaster
            if lower_is_better:
                best_forecaster = min(metric_values.items(), key=lambda x: x[1])[0]
            else:
                best_forecaster = max(metric_values.items(), key=lambda x: x[1])[0]
                
            best_forecasters[cat] = best_forecaster
            
        return best_forecasters
    
    def visualize_forecast_comparison(self, 
                                   forecasters: List[BaseForecaster],
                                   historical_data: Dict[str, pd.DataFrame],
                                   category: str,
                                   title: Optional[str] = None,
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization comparing forecasts from multiple forecasters.
        
        Args:
            forecasters: List of forecasters to compare
            historical_data: Dictionary mapping category names to historical DataFrames
            category: Category to visualize
            title: Plot title (optional)
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure with the plot
        """
        if category not in historical_data:
            raise ValueError(f"Category '{category}' not found in historical data")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        historical_df = historical_data[category]
        
        if 'transaction_date' in historical_df.columns:
            historical_df = historical_df.set_index('transaction_date')
        elif 'ds' in historical_df.columns:
            historical_df = historical_df.set_index('ds')
            
        value_col = 'amount_sum' if 'amount_sum' in historical_df.columns else 'amount'
        
        ax.plot(historical_df.index, historical_df[value_col], 'k-', label='Historical')
        
        # Plot forecast for each forecaster
        for forecaster in forecasters:
            if not forecaster.is_fitted or category not in forecaster.forecasts:
                continue
                
            forecast_df = forecaster.forecasts[category]
            
            # Set date as index if not already
            if 'ds' in forecast_df.columns:
                forecast_df = forecast_df.set_index('ds')
                
            value_col = 'amount' if 'amount' in forecast_df.columns else 'value'
            
            ax.plot(forecast_df.index, forecast_df[value_col], '-', label=f'{forecaster.name} Forecast')
            
            # Plot confidence intervals if available
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                ax.fill_between(forecast_df.index, 
                               forecast_df['lower_bound'], 
                               forecast_df['upper_bound'],
                               alpha=0.2)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Forecast Comparison for {category}')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Amount')
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        return fig
    
    def visualize_metric_comparison(self, 
                                  metric: str = 'rmse', 
                                  categories: Optional[List[str]] = None,
                                  title: Optional[str] = None,
                                  output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar chart comparing forecasters based on a specific metric.
        
        Args:
            metric: Metric to visualize
            categories: List of categories to include (if None, include all categories)
            title: Plot title (optional)
            output_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure with the plot
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_forecaster() or compare_forecasters() first.")
            
        # Get all categories
        all_categories = set()
        for forecaster_results in self.results.values():
            all_categories.update(forecaster_results.keys())
            
        # Filter categories if specified
        if categories:
            all_categories = [cat for cat in categories if cat in all_categories]
        else:
            all_categories = list(all_categories)
            
        if not all_categories:
            raise ValueError("No categories available for visualization")
            
        # Get all forecasters
        forecasters = list(self.results.keys())
        
        # Create data for the bar chart
        metric_data = []
        
        for cat in all_categories:
            for forecaster in forecasters:
                if cat in self.results[forecaster] and metric in self.results[forecaster][cat]:
                    metric_value = self.results[forecaster][cat][metric]
                    metric_data.append({
                        'Category': cat,
                        'Forecaster': forecaster,
                        'Metric': metric,
                        'Value': metric_value
                    })
        
        if not metric_data:
            raise ValueError(f"No data available for metric '{metric}'")
            
        # Create DataFrame for plotting
        metric_df = pd.DataFrame(metric_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        category_groups = metric_df.groupby('Category')
        num_categories = len(all_categories)
        num_forecasters = len(forecasters)
        bar_width = 0.8 / num_forecasters
        
        for i, forecaster in enumerate(forecasters):
            positions = np.arange(num_categories) + i * bar_width
            values = []
            
            for cat in all_categories:
                cat_data = metric_df[(metric_df['Category'] == cat) & (metric_df['Forecaster'] == forecaster)]
                if not cat_data.empty:
                    values.append(cat_data['Value'].values[0])
                else:
                    values.append(np.nan)
            
            ax.bar(positions, values, bar_width, label=forecaster)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{metric.upper()} Comparison by Category')
            
        ax.set_xlabel('Category')
        ax.set_ylabel(metric.upper())
        ax.set_xticks(np.arange(num_categories) + (num_forecasters - 1) * bar_width / 2)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        
        return fig
    
    def create_evaluation_report(self, 
                               forecasters: List[BaseForecaster],
                               test_data: Dict[str, pd.DataFrame],
                               historical_data: Dict[str, pd.DataFrame],
                               metrics: Optional[List[str]] = None,
                               output_dir: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report with metrics and visualizations.
        
        Args:
            forecasters: List of forecasters to evaluate
            test_data: Dictionary mapping category names to test DataFrames
            historical_data: Dictionary mapping category names to historical DataFrames
            metrics: List of metrics to calculate (default: ['mae', 'rmse', 'mape'])
            output_dir: Directory to save the report (optional)
            
        Returns:
            Path to the generated report
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        if metrics is None:
            metrics = self.metrics
        
        # Compare forecasters
        self.compare_forecasters(forecasters, test_data, metrics=metrics)
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"forecast_evaluation_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create CSV with evaluation metrics
        metric_rows = []
        
        for forecaster_name, forecaster_results in self.results.items():
            for category, category_metrics in forecaster_results.items():
                for metric, value in category_metrics.items():
                    metric_rows.append({
                        'Forecaster': forecaster_name,
                        'Category': category,
                        'Metric': metric,
                        'Value': value
                    })
        
        metric_df = pd.DataFrame(metric_rows)
        metric_csv_path = os.path.join(report_dir, "evaluation_metrics.csv")
        metric_df.to_csv(metric_csv_path, index=False)
        
        # Create pivot table for easier reading
        pivot_df = metric_df.pivot_table(
            index=['Forecaster', 'Category'],
            columns='Metric',
            values='Value'
        ).reset_index()
        
        pivot_csv_path = os.path.join(report_dir, "evaluation_metrics_pivot.csv")
        pivot_df.to_csv(pivot_csv_path, index=False)
        
        # Get best forecasters for each category and metric
        best_forecasters = {}
        
        for metric in metrics:
            best_forecasters[metric] = self.get_best_forecaster(metric=metric)
            
        # Create visualizations
        visualization_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Metric comparison visualizations
        for metric in metrics:
            fig = self.visualize_metric_comparison(
                metric=metric,
                title=f'{metric.upper()} Comparison by Category and Forecaster'
            )
            
            output_path = os.path.join(visualization_dir, f"{metric}_comparison.png")
            plt.savefig(output_path)
            plt.close(fig)
        
        # Forecast comparison visualizations
        for category in test_data.keys():
            if category in historical_data:
                fig = self.visualize_forecast_comparison(
                    forecasters,
                    historical_data,
                    category,
                    title=f'Forecast Comparison for {category}'
                )
                
                output_path = os.path.join(visualization_dir, f"{category}_forecast_comparison.png")
                plt.savefig(output_path)
                plt.close(fig)
        
        # Create HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-value {{ text-align: right; }}
                .best {{ font-weight: bold; color: green; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Forecast Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary of Best Forecasters</h2>
            <table>
                <tr>
                    <th>Category</th>
        """
        
        # Add metric columns
        for metric in metrics:
            html_report += f"<th>{metric.upper()}</th>\n"
            
        html_report += "</tr>\n"
        
        # Add rows for each category
        all_categories = set()
        for metric_dict in best_forecasters.values():
            all_categories.update(metric_dict.keys())
            
        for category in sorted(all_categories):
            html_report += f"<tr><td>{category}</td>\n"
            
            for metric in metrics:
                if category in best_forecasters[metric]:
                    best_forecaster = best_forecasters[metric][category]
                    html_report += f"<td class='best'>{best_forecaster}</td>\n"
                else:
                    html_report += "<td>-</td>\n"
                    
            html_report += "</tr>\n"
            
        html_report += """
            </table>
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Forecaster</th>
                    <th>Category</th>
        """
        
        # Add metric columns
        for metric in metrics:
            html_report += f"<th>{metric.upper()}</th>\n"
            
        html_report += "</tr>\n"
        
        # Add rows for each forecaster and category
        pivot_records = pivot_df.to_dict(orient='records')
        
        for record in pivot_records:
            forecaster = record['Forecaster']
            category = record['Category']
            
            html_report += f"<tr><td>{forecaster}</td><td>{category}</td>\n"
            
            for metric in metrics:
                if metric in record:
                    value = record[metric]
                    is_best = (category in best_forecasters[metric] and 
                              best_forecasters[metric][category] == forecaster)
                    
                    if is_best:
                        html_report += f"<td class='metric-value best'>{value:.4f}</td>\n"
                    else:
                        html_report += f"<td class='metric-value'>{value:.4f}</td>\n"
                else:
                    html_report += "<td>-</td>\n"
                    
            html_report += "</tr>\n"
            
        html_report += """
            </table>
            
            <h2>Metric Comparisons</h2>
        """
        
        # Add metric comparison visualizations
        for metric in metrics:
            html_report += f"""
            <h3>{metric.upper()} Comparison</h3>
            <img src="visualizations/{metric}_comparison.png" alt="{metric.upper()} Comparison">
            """
            
        html_report += """
            <h2>Forecast Comparisons</h2>
        """
        
        # Add forecast comparison visualizations
        for category in test_data.keys():
            if category in historical_data:
                html_report += f"""
                <h3>{category}</h3>
                <img src="visualizations/{category}_forecast_comparison.png" alt="Forecast Comparison for {category}">
                """
                
        html_report += """
        </body>
        </html>
        """
        
        # Write HTML report
        html_path = os.path.join(report_dir, "evaluation_report.html")
        
        with open(html_path, 'w') as f:
            f.write(html_report)
            
        print(f"Evaluation report generated at {html_path}")
        
        return html_path