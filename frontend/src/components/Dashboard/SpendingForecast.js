import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const SpendingForecast = () => {
  const [forecasts, setForecasts] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch forecasts
    const fetchForecasts = async () => {
      try {
        // In a real app, you would fetch from your API:
        // const response = await fetch('/api/v1/forecasts/summary');
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockForecasts = [
            {
              category: 'food',
              current_month: 420.75,
              next_month: 450.25,
              percent_change: 7.01
            },
            {
              category: 'transport',
              current_month: 250.30,
              next_month: 230.10,
              percent_change: -8.07
            },
            {
              category: 'entertainment',
              current_month: 175.45,
              next_month: 185.80,
              percent_change: 5.90
            }
          ];
          
          setForecasts(mockForecasts);
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching forecasts:', err);
        setLoading(false);
      }
    };
    
    fetchForecasts();
  }, []);
  
  if (loading) {
    return <div className="loading">Loading forecasts...</div>;
  }
  
  if (forecasts.length === 0) {
    return <div className="no-data">No forecast data available</div>;
  }
  
  return (
    <div className="spending-forecast">
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #eee' }}>Category</th>
            <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>Current</th>
            <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>Next Month</th>
            <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>Change</th>
            <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>Change</th>
          </tr>
        </thead>
        <tbody>
          {forecasts.map((forecast, index) => (
            <tr key={index}>
              <td style={{ padding: '8px', borderBottom: '1px solid #eee', textTransform: 'capitalize' }}>
                {forecast.category}
              </td>
              <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                ${forecast.current_month.toFixed(2)}
              </td>
              <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                ${forecast.next_month.toFixed(2)}
              </td>
              <td style={{ 
                textAlign: 'right', 
                padding: '8px', 
                borderBottom: '1px solid #eee',
                color: forecast.percent_change >= 0 ? 'var(--danger-color)' : 'var(--secondary-color)'
              }}>
                {forecast.percent_change >= 0 ? '+' : ''}
                {forecast.percent_change.toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="view-all" style={{ marginTop: '15px', textAlign: 'right' }}>
        <Link to="/forecasts">View Full Forecast</Link>
      </div>
    </div>
  );
};

export default SpendingForecast;