import React, { useState, useEffect } from 'react';
import './Dashboard.css';
import TransactionSummary from './TransactionSummary';
import RecentTransactions from './RecentTransactions';
import SpendingForecast from './SpendingForecast';
import TopRecommendations from './TopRecommendations';


const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [summaryData, setSummaryData] = useState({
    income: 0,
    expenses: 0,
    netCashflow: 0
  });
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch dashboard data
    const fetchDashboardData = async () => {
      try {
        // In a real app, you would fetch data from your API:
        // const response = await fetch('/api/v1/transactions/stats');
        // const data = await response.json();
        
        // For demo, we'll use mock data
        setTimeout(() => {
          const mockData = {
            income: 5280.45,
            expenses: 3451.23,
            netCashflow: 5280.45 - 3451.23
          };
          
          setSummaryData(mockData);
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError('Failed to load dashboard data');
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, []);
  
  if (loading) {
    return <div className="loading">Loading dashboard data...</div>;
  }
  
  if (error) {
    return <div className="error">{error}</div>;
  }
  
  return (
    <div className="dashboard">
      <h1>Financial Dashboard</h1>
      
      <div className="dashboard-summary">
        <TransactionSummary
          income={summaryData.income}
          expenses={summaryData.expenses}
          netCashflow={summaryData.netCashflow}
        />
      </div>
      
      <div className="dashboard-widgets">
        <div className="widget">
          <h2>Recent Transactions</h2>
          <RecentTransactions />
        </div>
        
        <div className="widget">
          <h2>Spending Forecast</h2>
          <SpendingForecast />
        </div>
        
        <div className="widget">
          <h2>Top Recommendations</h2>
          <TopRecommendations />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;