import React from 'react';
import { Link } from 'react-router-dom';

const Header = ({ user, onLogout }) => {
  return (
    <header className="header">
      <div className="logo">
        <Link to="/">Financial Health Assistant</Link>
      </div>
      
      <div className="header-right">
        <div className="user-info">
          <span className="welcome-message">Welcome, {user?.name || 'User'}</span>
          <button className="logout-button" onClick={onLogout}>Logout</button>
        </div>
      </div>
    </header>
  );
};

export default Header;