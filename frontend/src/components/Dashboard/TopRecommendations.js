import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const TopRecommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch recommendations
    const fetchRecommendations = async () => {
      try {
        // In a real app, you would fetch from your API:
        // const response = await fetch('/api/v1/recommendations?limit=3');
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockRecommendations = [
            {
              id: '1',
              message: 'Your food spending is 28% of total expenses. Consider ways to reduce this category.',
              type: 'high_spending',
              priority: 8
            },
            {
              id: '2',
              message: 'You have 5 subscription services totaling $65.95 per month. Review which ones you actually use.',
              type: 'recurring_charge',
              priority: 7
            },
            {
              id: '3',
              message: 'Set up an emergency fund of at least $1,500 to cover unexpected expenses.',
              type: 'financial_habit',
              priority: 9
            }
          ];
          
          setRecommendations(mockRecommendations);
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setLoading(false);
      }
    };
    
    fetchRecommendations();
  }, []);
  
  if (loading) {
    return <div className="loading">Loading recommendations...</div>;
  }
  
  if (recommendations.length === 0) {
    return <div className="no-data">No recommendations available</div>;
  }
  
  return (
    <div className="top-recommendations">
      {recommendations.map(recommendation => (
        <div key={recommendation.id} className="recommendation-item">
          <div className="recommendation-message">{recommendation.message}</div>
          <div className="recommendation-meta">
            <span className="recommendation-type">{recommendation.type.replace('_', ' ')}</span>
            <span className="recommendation-priority">Priority: {recommendation.priority}/10</span>
          </div>
        </div>
      ))}
      <div className="view-all" style={{ marginTop: '15px', textAlign: 'right' }}>
        <Link to="/recommendations">View All Recommendations</Link>
      </div>
    </div>
  );
};

export default TopRecommendations;