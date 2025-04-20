import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-toggle" onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? '→' : '←'}
      </div>
      
      <nav className="sidebar-nav">
        <ul>
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-home"></i>
              <span className="nav-text">Dashboard</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/transactions" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-exchange-alt"></i>
              <span className="nav-text">Transactions</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/recommendations" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-lightbulb"></i>
              <span className="nav-text">Recommendations</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/forecasts" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="fas fa-chart-line"></i>
              <span className="nav-text">Forecasts</span>
            </NavLink>
          </li>
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar;