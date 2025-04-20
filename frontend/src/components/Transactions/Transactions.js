import React, { useState, useEffect } from 'react';

const Transactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState({
    category: '',
    startDate: '',
    endDate: '',
    minAmount: '',
    maxAmount: ''
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  
  // Fetch transactions with filters and pagination
  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        // In a real app, you would fetch from your API with filters:
        // const queryParams = new URLSearchParams();
        // if (filter.category) queryParams.append('category', filter.category);
        // if (filter.startDate) queryParams.append('start_date', filter.startDate);
        // if (filter.endDate) queryParams.append('end_date', filter.endDate);
        // if (filter.minAmount) queryParams.append('min_amount', filter.minAmount);
        // if (filter.maxAmount) queryParams.append('max_amount', filter.maxAmount);
        // queryParams.append('page', currentPage);
        // queryParams.append('page_size', 10);
        
        // const response = await fetch(`/api/v1/transactions?${queryParams}`);
        // const data = await response.json();
        
        // For demo, use mock data
        setTimeout(() => {
          const mockTransactions = Array(15).fill(0).map((_, i) => ({
            id: `${i + 1}`,
            transaction_date: new Date(Date.now() - (i * 86400000)).toISOString(),
            description: ['Grocery Store', 'Salary Deposit', 'Gas Station', 'Online Shopping', 'Restaurant'][i % 5],
            category: ['food', 'income', 'transport', 'shopping', 'food'][i % 5],
            amount: [i % 5 === 1 ? 2500 : -85.75, -45.50, -120.99, -68.20][i % 4]
          }));
          
          setTransactions(mockTransactions);
          setTotalPages(3); // Mock 3 pages total
          setLoading(false);
        }, 1000);
      } catch (err) {
        console.error('Error fetching transactions:', err);
        setLoading(false);
      }
    };
    
    fetchTransactions();
  }, [currentPage, filter]);
  
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilter(prev => ({ ...prev, [name]: value }));
    setCurrentPage(1); // Reset to first page when filter changes
  };
  
  const handlePageChange = (page) => {
    setCurrentPage(page);
  };
  
  if (loading && transactions.length === 0) {
    return <div className="loading">Loading transactions...</div>;
  }
  
  return (
    <div className="transactions-page">
      <h1>Transactions</h1>
      
      <div className="filters" style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h3>Filters</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px', marginTop: '10px' }}>
          <div>
            <label htmlFor="category">Category</label>
            <select 
              id="category" 
              name="category" 
              value={filter.category} 
              onChange={handleFilterChange}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="">All Categories</option>
              <option value="food">Food</option>
              <option value="transport">Transport</option>
              <option value="shopping">Shopping</option>
              <option value="utilities">Utilities</option>
              <option value="entertainment">Entertainment</option>
              <option value="income">Income</option>
            </select>
          </div>
          
          <div>
            <label htmlFor="startDate">Start Date</label>
            <input 
              type="date" 
              id="startDate" 
              name="startDate" 
              value={filter.startDate} 
              onChange={handleFilterChange}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>
          
          <div>
            <label htmlFor="endDate">End Date</label>
            <input 
              type="date" 
              id="endDate" 
              name="endDate" 
              value={filter.endDate} 
              onChange={handleFilterChange}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>
          
          <div>
            <label htmlFor="minAmount">Min Amount</label>
            <input 
              type="number" 
              id="minAmount" 
              name="minAmount" 
              value={filter.minAmount} 
              onChange={handleFilterChange}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>
          
          <div>
            <label htmlFor="maxAmount">Max Amount</label>
            <input 
              type="number" 
              id="maxAmount" 
              name="maxAmount" 
              value={filter.maxAmount} 
              onChange={handleFilterChange}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>
        </div>
      </div>
      
      <div className="transactions-table" style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '12px 8px', backgroundColor: '#f5f5f5', borderBottom: '2px solid #ddd' }}>Date</th>
              <th style={{ textAlign: 'left', padding: '12px 8px', backgroundColor: '#f5f5f5', borderBottom: '2px solid #ddd' }}>Description</th>
              <th style={{ textAlign: 'left', padding: '12px 8px', backgroundColor: '#f5f5f5', borderBottom: '2px solid #ddd' }}>Category</th>
              <th style={{ textAlign: 'right', padding: '12px 8px', backgroundColor: '#f5f5f5', borderBottom: '2px solid #ddd' }}>Amount</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map(transaction => (
              <tr key={transaction.id}>
                <td style={{ padding: '12px 8px', borderBottom: '1px solid #eee' }}>
                  {new Date(transaction.transaction_date).toLocaleDateString()}
                </td>
                <td style={{ padding: '12px 8px', borderBottom: '1px solid #eee' }}>
                  {transaction.description}
                </td>
                <td style={{ padding: '12px 8px', borderBottom: '1px solid #eee', textTransform: 'capitalize' }}>
                  {transaction.category}
                </td>
                <td style={{ 
                  textAlign: 'right', 
                  padding: '12px 8px', 
                  borderBottom: '1px solid #eee',
                  color: transaction.amount < 0 ? 'var(--danger-color)' : 'var(--secondary-color)',
                  fontWeight: '600'
                }}>
                  ${Math.abs(transaction.amount).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="pagination" style={{ marginTop: '20px', display: 'flex', justifyContent: 'center' }}>
        <button 
          onClick={() => handlePageChange(currentPage - 1)} 
          disabled={currentPage === 1}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: currentPage === 1 ? '#f5f5f5' : 'var(--primary-color)', 
            color: currentPage === 1 ? '#999' : 'white',
            border: '1px solid #ddd',
            borderRadius: '4px 0 0 4px',
            cursor: currentPage === 1 ? 'not-allowed' : 'pointer'
          }}
        >
          Previous
        </button>
        
        <div style={{ padding: '8px 16px', backgroundColor: 'white', border: '1px solid #ddd', borderLeft: 'none', borderRight: 'none' }}>
          Page {currentPage} of {totalPages}
        </div>
        
        <button 
          onClick={() => handlePageChange(currentPage + 1)} 
          disabled={currentPage === totalPages}
          style={{ 
            padding: '8px 16px', 
            backgroundColor: currentPage === totalPages ? '#f5f5f5' : 'var(--primary-color)', 
            color: currentPage === totalPages ? '#999' : 'white',
            border: '1px solid #ddd',
            borderRadius: '0 4px 4px 0',
            cursor: currentPage === totalPages ? 'not-allowed' : 'pointer'
          }}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default Transactions;