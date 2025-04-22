// src/config.js

// API configuration
const DEV_API_URL = 'http://localhost:8000/api/v1';
const PROD_API_URL = 'https://api.financialhealthassistant.com/api/v1';

export const API_URL = __DEV__ ? DEV_API_URL : PROD_API_URL;

// App configuration
export const APP_CONFIG = {
  // Default currency code for formatting
  currencyCode: 'USD',
  
  // Default date format
  dateFormat: 'YYYY-MM-DD',
  
  // Default chart colors
  chartColors: {
    primary: '#2196F3',
    secondary: '#03A9F4',
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    gray: '#9E9E9E',
  },
  
  // Category configuration
  categories: {
    food: {
      color: '#FF5722',
      icon: 'food',
      description: 'Food and dining expenses',
    },
    transport: {
      color: '#03A9F4',
      icon: 'car',
      description: 'Transportation expenses',
    },
    utilities: {
      color: '#4CAF50',
      icon: 'flash',
      description: 'Utility bills and services',
    },
    entertainment: {
      color: '#9C27B0',
      icon: 'movie',
      description: 'Entertainment and recreation',
    },
    shopping: {
      color: '#FF9800',
      icon: 'shopping',
      description: 'Retail and online shopping',
    },
    health: {
      color: '#F44336',
      icon: 'medical-bag',
      description: 'Healthcare and medical expenses',
    },
    housing: {
      color: '#3F51B5',
      icon: 'home',
      description: 'Rent, mortgage, and home expenses',
    },
    income: {
      color: '#4CAF50',
      icon: 'cash',
      description: 'Salary and other income',
    },
    other: {
      color: '#9E9E9E',
      icon: 'dots-horizontal',
      description: 'Miscellaneous expenses',
    }
  },
  
  // Recommendation types
  recommendationTypes: {
    spending_alert: {
      icon: 'alert-circle',
      description: 'Notifications about unusual spending patterns',
    },
    savings_opportunity: {
      icon: 'piggy-bank',
      description: 'Opportunities to increase savings',
    },
    budget_recommendation: {
      icon: 'cash',
      description: 'Suggestions for budgeting improvements',
    },
    subscription_alert: {
      icon: 'refresh',
      description: 'Alerts about subscription services',
    },
    income_opportunity: {
      icon: 'trending-up',
      description: 'Opportunities to increase income',
    },
    debt_management: {
      icon: 'credit-card',
      description: 'Strategies to manage and reduce debt',
    },
  },
  
  // Development settings
  dev: {
    // Set to true to use mock data instead of real API calls
    useMockData: true,
    
    // Artificial delay for mock API calls (in milliseconds)
    mockDelay: 500,
  }
};

// Utility functions
export const getCategoryColor = (categoryName) => {
  return APP_CONFIG.categories[categoryName]?.color || APP_CONFIG.categories.other.color;
};

export const getCategoryIcon = (categoryName) => {
  return APP_CONFIG.categories[categoryName]?.icon || APP_CONFIG.categories.other.icon;
};

export const getRecommendationTypeIcon = (typeName) => {
  return APP_CONFIG.recommendationTypes[typeName]?.icon || 'lightbulb-on';
};

export default APP_CONFIG;