import React, { useState, useEffect } from 'react';

const Forecasts = () => {
  const [forecasts, setForecasts] = useState({});
  const [loading, setLoading] = useState(true);
  const [forecastDays, setForecastDays] = useState(30);
  
  useEffect(() => {
    // Fetch forecasts
    const fetchForecasts = async () => {
      try {
        // In a real app, you would fetch from your API:
        // const response = await fetch(`/api/v1/forecasts?days=${forecastDays}`);
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockCategories = ['food', 'transport', 'entertainment', 'utilities', 'shopping'];
          const mockForecasts = {};
          
          mockCategories.forEach(category => {
            // Create days worth of forecast data for this category
            const startDate = new Date();
            const forecastPoints = [];
            
            let baseAmount = 0;
            switch(category) {
              case 'food': baseAmount = 15; break;
              case 'transport': baseAmount = 8; break;
              case 'entertainment': baseAmount = 5; break;
              case 'utilities': baseAmount = 10; break;
              case 'shopping': baseAmount = 12; break;
              default: baseAmount = 10;
            }
            
            for (let i = 0; i < forecastDays; i++) {
              const currentDate = new Date(startDate);
              currentDate.setDate(currentDate.getDate() + i);
              
              // Add some variability to daily amounts
              const dayAmount = baseAmount + (Math.random() * baseAmount * 0.5);
              
              // Weekend adjustment
              const isWeekend = currentDate.getDay() === 0 || currentDate.getDay() === 6;
              const adjustedAmount = isWeekend ? dayAmount * 1.5 : dayAmount;
              
              forecastPoints.push({
                date: currentDate.toISOString().split('T')[0],
                amount: adjustedAmount,
                lower_bound: adjustedAmount * 0.8,
                upper_bound: adjustedAmount * 1.2
              });
            }
            
            mockForecasts[category] = {
              category,
              forecast_points: forecastPoints,
              total_forecast: forecastPoints.reduce((sum, point) => sum + point.amount, 0),
              current_month_avg: baseAmount,
              forecast_period_days: forecastDays
            };
          });
          
          setForecasts(mockForecasts);
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching forecasts:', err);
        setLoading(false);
      }
    };
    
    fetchForecasts();
  }, [forecastDays]);
  
  const handleDaysChange = (e) => {
    setForecastDays(Number(e.target.value));
    setLoading(true);
  };
  
  if (loading && Object.keys(forecasts).length === 0) {
    return <div className="loading">Loading forecasts...</div>;
  }
  
  return (
    <div className="forecasts-page">
      <h1>Spending Forecasts</h1>
      
      <div className="forecast-controls" style={{ marginBottom: '20px', backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '8px' }}>
        <label htmlFor="forecastDays">Forecast period: </label>
        <select 
          id="forecastDays" 
          value={forecastDays} 
          onChange={handleDaysChange}
          style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '10px' }}
        >
          <option value="7">7 days</option>
          <option value="14">14 days</option>
          <option value="30">30 days</option>
          <option value="60">60 days</option>
          <option value="90">90 days</option>
        </select>
      </div>
      
      <div className="forecast-summary" style={{ marginBottom: '30px', backgroundColor: 'white', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)' }}>
        <h2>Forecast Summary</h2>
        <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
          {Object.values(forecasts).map(forecast => (
            <div key={forecast.category} style={{ textAlign: 'center', padding: '15px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
              <div style={{ textTransform: 'capitalize', fontSize: '1.1rem', fontWeight: '600', color: 'var(--primary-color)', marginBottom: '10px' }}>
                {forecast.category}
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: '700', marginBottom: '5px' }}>
                ${forecast.total_forecast.toFixed(2)}
              </div>
              <div style={{ color: '#777', fontSize: '0.9rem' }}>
                Forecast for {forecast.forecast_period_days} days
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="forecast-details">
        <h2>Category Details</h2>
        
        {Object.values(forecasts).map(forecast => (
          <div 
            key={forecast.category} 
            style={{ 
              backgroundColor: 'white', 
              marginBottom: '20px', 
              padding: '20px', 
              borderRadius: '8px', 
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)' 
            }}
          >
            <h3 style={{ marginBottom: '15px', textTransform: 'capitalize' }}>{forecast.category}</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px', marginBottom: '20px' }}>
              <div style={{ backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ color: '#777', marginBottom: '5px' }}>Total Forecast</div>
                <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>${forecast.total_forecast.toFixed(2)}</div>
              </div>
              
              <div style={{ backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ color: '#777', marginBottom: '5px' }}>Daily Average</div>
                <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>
                  ${(forecast.total_forecast / forecast.forecast_period_days).toFixed(2)}
                </div>
              </div>
              
              <div style={{ backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ color: '#777', marginBottom: '5px' }}>Previous Average</div>
                <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>${forecast.current_month_avg.toFixed(2)}</div>
              </div>
            </div>
            
            <div style={{ color: '#777', marginBottom: '10px' }}>Daily Forecast (7-day sample)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Date</th>
                    <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Amount</th>
                    <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Min</th>
                    <th style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #ddd' }}>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.forecast_points.slice(0, 7).map((point, i) => (
                    <tr key={i}>
                      <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>
                        {new Date(point.date).toLocaleDateString()}
                      </td>
                      <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                        ${point.amount.toFixed(2)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                        ${point.lower_bound.toFixed(2)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '8px', borderBottom: '1px solid #eee' }}>
                        ${point.upper_bound.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Forecasts;