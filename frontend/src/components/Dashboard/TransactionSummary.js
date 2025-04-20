import React from 'react';

const TransactionSummary = ({ income, expenses, netCashflow }) => {
  return (
    <>
      <div className="summary-card">
        <span className="summary-title">Total Income</span>
        <span className="summary-value positive">${income.toFixed(2)}</span>
      </div>
      
      <div className="summary-card">
        <span className="summary-title">Total Expenses</span>
        <span className="summary-value negative">${expenses.toFixed(2)}</span>
      </div>
      
      <div className="summary-card">
        <span className="summary-title">Net Cash Flow</span>
        <span className={`summary-value ${netCashflow >= 0 ? 'positive' : 'negative'}`}>
          ${Math.abs(netCashflow).toFixed(2)}
        </span>
      </div>
    </>
  );
};

export default TransactionSummary;