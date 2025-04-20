import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Component imports
import Dashboard from './components/Dashboard/Dashboard';
import Transactions from './components/Transactions/Transactions';
import Recommendations from './components/Recommendations/Recommendations';
import Forecasts from './components/Forecasts/Forecasts';
import Login from './components/Auth/Login';
import Register from './components/Auth/Register';
import Header from './components/common/Header';
import Sidebar from './components/common/sidebar';

// Styles
import './styles/App.css';

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  
  // Check if user is authenticated on load
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // You would typically validate the token here
      setAuthenticated(true);
      // Get user info
      setUser({ name: "Demo User" });
    }
  }, []);

  // For development purposes, automatically authenticate
  useEffect(() => {
    setAuthenticated(true);
    setUser({ name: "Demo User" });
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setAuthenticated(false);
    setUser(null);
  };

  // Render auth screens if not authenticated
  if (!authenticated) {
    return (
      <Router>
        <div className="auth-container">
          <Routes>
            <Route path="/login" element={<Login setAuthenticated={setAuthenticated} />} />
            <Route path="/register" element={<Register />} />
            <Route path="*" element={<Navigate to="/login" />} />
          </Routes>
        </div>
      </Router>
    );
  }

  // Render main app if authenticated
  return (
    <Router>
      <div className="app-container">
        <Header user={user} onLogout={handleLogout} />
        <div className="content-container">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/transactions" element={<Transactions />} />
              <Route path="/recommendations" element={<Recommendations />} />
              <Route path="/forecasts" element={<Forecasts />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;