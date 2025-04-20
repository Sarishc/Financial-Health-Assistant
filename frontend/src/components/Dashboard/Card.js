import React from 'react';
import './Card.css';

const Card = ({ title, children, className = '', onAction, actionText }) => {
  return (
    <div className={`card ${className}`}>
      {title && (
        <div className="card-header">
          {title}
          {onAction && actionText && (
            <button className="card-action" onClick={onAction}>
              {actionText}
            </button>
          )}
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  );
};

export default Card;