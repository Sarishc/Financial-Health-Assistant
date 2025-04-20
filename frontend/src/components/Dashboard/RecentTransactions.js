import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const RecentTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch recent transactions
    const fetchTransactions = async () => {
      try {
        // In a real app, you would fetch from your API:
        // const response = await fetch('/api/v1/transactions?limit=5');
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockTransactions = [
            {
              id: '1',
              transaction_date: '2023-04-15T10:30:00',
              description: 'Grocery Store',
              category: 'food',
              amount: -85.75
            },
            {
              id: '2',
              transaction_date: '2023-04-14T14:45:00',
              description: 'Salary Deposit',
              category: 'income',
              amount: 2500.00
            },
            {
              id: '3',
              transaction_date: '2023-04-12T09:15:00',
              description: 'Gas Station',
              category: 'transport',
              amount: -45.50
            },
            {
              id: '4',
              transaction_date: '2023-04-10T20:20:00',
              description: 'Online Streaming',
              category: 'entertainment',
              amount: -14.99
            },
            {
              id: '5',
              transaction_date: '2023-04-08T12:10:00',
              description: 'Restaurant',
              category: 'food',
              amount: -68.20
            }
          ];
          
          setTransactions(mockTransactions);
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching transactions:', err);
        setLoading(false);
      }
    };
    
    fetchTransactions();
  }, []);
  
  if (loading) {
    return <div className="loading">Loading transactions...</div>;
  }
  
  if (transactions.length === 0) {
    return <div className="no-data">No recent transactions</div>;
  }
  
  return (
    <div className="recent-transactions">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Description</th>
            <th>Category</th>
            <th>Amount</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map(transaction => (
            <tr key={transaction.id}>
              <td>{new Date(transaction.transaction_date).toLocaleDateString()}</td>
              <td>{transaction.description}</td>
              <td>{transaction.category}</td>
              <td className={transaction.amount < 0 ? 'negative' : 'positive'}>
                ${Math.abs(transaction.amount).toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="view-all" style={{ marginTop: '15px', textAlign: 'right' }}>
        <Link to="/transactions">View All Transactions</Link>
      </div>
    </div>
  );
};

export default RecentTransactions;