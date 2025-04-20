import React, { useState, useEffect } from 'react';

const Recommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedType, setSelectedType] = useState('');
  
  useEffect(() => {
    // Fetch recommendations with filters
    const fetchRecommendations = async () => {
      try {
        // In a real app, you would fetch from your API with filters:
        // const queryParams = new URLSearchParams();
        // if (selectedType) queryParams.append('type', selectedType);
        
        // const response = await fetch(`/api/v1/recommendations?${queryParams}`);
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockRecommendations = [
            {
              id: '1',
              message: 'Your food spending is 28% of total expenses. Consider ways to reduce this category.',
              type: 'high_spending',
              category: 'food',
              priority: 8
            },
            {
              id: '2',
              message: 'You have 5 subscription services totaling $65.95 per month. Review which ones you actually use.',
              type: 'recurring_charge',
              category: 'entertainment',
              priority: 7
            },
            {
              id: '3',
              message: 'Set up an emergency fund of at least $1,500 to cover unexpected expenses.',
              type: 'financial_habit',
              category: null,
              priority: 9
            },
            {
              id: '4',
              message: 'Your transport spending increased by 32% compared to the previous month. Consider reviewing this category.',
              type: 'spending_pattern',
              category: 'transport',
              priority: 6
            },
            {
              id: '5',
              message: 'You spend $150 monthly on dining out. Cooking at home could save you up to $80 per month.',
              type: 'saving_opportunity',
              category: 'food',
              priority: 7
            },
            {
              id: '6',
              message: 'Your entertainment spending is projected to increase by 15% next month. Consider setting a budget for this category.',
              type: 'budget_alert',
              category: 'entertainment',
              priority: 5
            },
            {
              id: '7',
              message: 'You had negative cash flow in 2 of the last 3 months. Try to build a buffer for these periods.',
              type: 'cashflow_improvement',
              category: null,
              priority: 8
            }
          ];
          
          // Filter by type if selected
          const filtered = selectedType 
            ? mockRecommendations.filter(rec => rec.type === selectedType)
            : mockRecommendations;
            
          setRecommendations(filtered);
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setLoading(false);
      }
    };
    
    fetchRecommendations();
  }, [selectedType]);
  
  const handleTypeChange = (e) => {
    setSelectedType(e.target.value);
  };
  
  // Function to regenerate recommendations
  const handleRegenerate = () => {
    setLoading(true);
    // In a real app, you would call an API endpoint to regenerate:
    // fetch('/api/v1/recommendations/generate', { method: 'POST' })
    //   .then(response => response.json())
    //   .then(data => {
    //     setRecommendations(data.recommendations);
    //     setLoading(false);
    //   })
    //   .catch(err => {
    //     console.error('Error regenerating recommendations:', err);
    //     setLoading(false);
    //   });
    
    // For demo, just wait a bit and reuse the same data
    setTimeout(() => {
      setLoading(false);
    }, 1500);
  };
  
  if (loading && recommendations.length === 0) {
    return <div className="loading">Loading recommendations...</div>;
  }
  
  return (
    <div className="recommendations-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1>Financial Recommendations</h1>
        <button 
          onClick={handleRegenerate}
          style={{
            padding: '8px 16px',
            backgroundColor: 'var(--primary-color)',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Regenerate Recommendations
        </button>
      </div>
      
      <div className="filters" style={{ marginBottom: '20px', backgroundColor: '#f5f5f5', padding: '15px', borderRadius: '8px' }}>
        <label htmlFor="type">Filter by type: </label>
        <select 
          id="type" 
          value={selectedType} 
          onChange={handleTypeChange}
          style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '10px', minWidth: '200px' }}
        >
          <option value="">All Types</option>
          <option value="high_spending">High Spending</option>
          <option value="recurring_charge">Recurring Charges</option>
          <option value="saving_opportunity">Saving Opportunities</option>
          <option value="budget_alert">Budget Alerts</option>
          <option value="spending_pattern">Spending Patterns</option>
          <option value="cashflow_improvement">Cash Flow Improvements</option>
          <option value="financial_habit">Financial Habits</option>
        </select>
      </div>
      
      {recommendations.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '30px', backgroundColor: '#f9f9f9', borderRadius: '8px' }}>
          No recommendations found for the selected filter.
        </div>
      ) : (
        <div className="recommendations-list">
          {recommendations.map(recommendation => (
            <div 
              key={recommendation.id} 
              style={{
                backgroundColor: 'white',
                borderRadius: '8px',
                padding: '20px',
                marginBottom: '15px',
                boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                borderLeft: `4px solid ${recommendation.priority >= 8 ? 'var(--danger-color)' : 
                  recommendation.priority >= 6 ? 'var(--warning-color)' : 'var(--primary-color)'}`
              }}
            >
              <div style={{ marginBottom: '10px', fontSize: '1.1rem', fontWeight: '500' }}>
                {recommendation.message}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#777', fontSize: '0.9rem' }}>
                <div>
                  <span style={{ 
                    display: 'inline-block', 
                    padding: '3px 8px', 
                    backgroundColor: '#f5f5f5', 
                    borderRadius: '12px',
                    marginRight: '10px',
                    textTransform: 'capitalize'
                  }}>
                    {recommendation.type.replace(/_/g, ' ')}
                  </span>
                  {recommendation.category && (
                    <span style={{ 
                      display: 'inline-block', 
                      padding: '3px 8px', 
                      backgroundColor: '#e3f2fd', 
                      borderRadius: '12px',
                      textTransform: 'capitalize'
                    }}>
                      {recommendation.category}
                    </span>
                  )}
                </div>
                <div>
                  <span style={{ fontWeight: '600' }}>
                    Priority: {recommendation.priority}/10
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Recommendations;