// src/store/actions/authActions.js
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authStart, authSuccess, authFail, authLogout } from '../reducers/authReducer';
import { API_URL } from '../../config';

// Login action
export const login = (email, password) => {
  return async dispatch => {
    dispatch(authStart());
    
    try {
      // For development/testing, you can use mock data
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Mock successful login
        const userData = {
          id: 'user-123',
          name: 'Test User',
          email: email,
          token: 'mock-jwt-token'
        };
        
        // Store token
        await AsyncStorage.setItem('userToken', 'mock-jwt-token');
        
        // Dispatch success
        dispatch(authSuccess(userData));
        return userData;
      }
      
      // API call for real backend
      const response = await axios.post(`${API_URL}/auth/login`, {
        email,
        password
      });
      
      const { user, token } = response.data;
      
      // Store token
      await AsyncStorage.setItem('userToken', token);
      
      // Set auth header for future requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      // Dispatch success
      dispatch(authSuccess(user));
      return user;
    } catch (error) {
      dispatch(authFail(error.response?.data?.message || 'Authentication failed'));
      throw error;
    }
  };
};

// Register action
export const register = (userData) => {
  return async dispatch => {
    dispatch(authStart());
    
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Mock successful registration
        const newUser = {
          id: 'user-' + Math.floor(Math.random() * 1000),
          name: userData.name,
          email: userData.email,
          token: 'mock-jwt-token'
        };
        
        // Store token
        await AsyncStorage.setItem('userToken', 'mock-jwt-token');
        
        // Dispatch success
        dispatch(authSuccess(newUser));
        return newUser;
      }
      
      // API call for real backend
      const response = await axios.post(`${API_URL}/auth/register`, userData);
      
      const { user, token } = response.data;
      
      // Store token
      await AsyncStorage.setItem('userToken', token);
      
      // Set auth header for future requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      // Dispatch success
      dispatch(authSuccess(user));
      return user;
    } catch (error) {
      dispatch(authFail(error.response?.data?.message || 'Registration failed'));
      throw error;
    }
  };
};

// Logout action
export const logout = () => {
  return async dispatch => {
    // Remove token from storage
    await AsyncStorage.removeItem('userToken');
    
    // Remove auth header
    delete axios.defaults.headers.common['Authorization'];
    
    // Dispatch logout
    dispatch(authLogout());
  };
};

// Auto login on app start
export const autoLogin = () => {
  return async dispatch => {
    const token = await AsyncStorage.getItem('userToken');
    
    if (!token) {
      dispatch(authLogout());
      return;
    }
    
    // Set auth header for future requests
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    
    try {
      // Fetch user data using token
      const response = await axios.get(`${API_URL}/auth/me`);
      
      dispatch(authSuccess(response.data.user));
    } catch (error) {
      // Token might be expired or invalid
      await AsyncStorage.removeItem('userToken');
      delete axios.defaults.headers.common['Authorization'];
      dispatch(authLogout());
    }
  };
};

// src/store/actions/transactionActions.js
import axios from 'axios';
import { 
  fetchTransactionsStart,
  fetchTransactionsSuccess,
  fetchTransactionsFail,
  updateTransactionSuccess,
  deleteTransactionSuccess,
  addTransactionSuccess,
  fetchSimilarTransactionsSuccess
} from '../reducers/transactionReducer';
import { API_URL } from '../../config';

// Sample transaction data for development/testing
const MOCK_TRANSACTIONS = [
  {
    id: 'txn-1',
    description: 'Grocery Store',
    amount: -120.50,
    transaction_date: '2025-04-15T10:30:00Z',
    category: 'food'
  },
  {
    id: 'txn-2',
    description: 'Salary Deposit',
    amount: 3000.00,
    transaction_date: '2025-04-01T09:00:00Z',
    category: 'income'
  },
  {
    id: 'txn-3',
    description: 'Gas Station',
    amount: -45.75,
    transaction_date: '2025-04-18T15:45:00Z',
    category: 'transport'
  },
  {
    id: 'txn-4',
    description: 'Netflix Subscription',
    amount: -15.99,
    transaction_date: '2025-04-05T00:00:00Z',
    category: 'entertainment'
  },
  {
    id: 'txn-5',
    description: 'Rent Payment',
    amount: -1200.00,
    transaction_date: '2025-04-01T08:00:00Z',
    category: 'housing'
  },
  {
    id: 'txn-6',
    description: 'Amazon Purchase',
    amount: -87.65,
    transaction_date: '2025-04-10T14:20:00Z',
    category: 'shopping'
  },
  {
    id: 'txn-7',
    description: 'Electric Bill',
    amount: -95.50,
    transaction_date: '2025-04-08T11:00:00Z',
    category: 'utilities'
  },
  {
    id: 'txn-8',
    description: 'Pharmacy',
    amount: -32.40,
    transaction_date: '2025-04-12T16:30:00Z',
    category: 'health'
  },
  {
    id: 'txn-9',
    description: 'Restaurant Dinner',
    amount: -75.20,
    transaction_date: '2025-04-14T19:45:00Z',
    category: 'food'
  },
  {
    id: 'txn-10',
    description: 'Uber Ride',
    amount: -18.50,
    transaction_date: '2025-04-16T22:10:00Z',
    category: 'transport'
  }
];

// Fetch all transactions
export const fetchTransactions = () => {
  return async dispatch => {
    dispatch(fetchTransactionsStart());
    
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Return mock data after a small delay to simulate network request
        setTimeout(() => {
          dispatch(fetchTransactionsSuccess(MOCK_TRANSACTIONS));
        }, 500);
        return;
      }
      
      // API call for real backend
      const response = await axios.get(`${API_URL}/transactions`);
      dispatch(fetchTransactionsSuccess(response.data.transactions));
    } catch (error) {
      dispatch(fetchTransactionsFail(error.response?.data?.message || 'Failed to fetch transactions'));
    }
  };
};

// Add a new transaction
export const addTransaction = (transactionData) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Create a mock transaction with ID
        const newTransaction = {
          id: `txn-${Math.floor(Math.random() * 1000)}`,
          ...transactionData,
        };
        
        dispatch(addTransactionSuccess(newTransaction));
        return newTransaction;
      }
      
      // API call for real backend
      const response = await axios.post(`${API_URL}/transactions`, transactionData);
      dispatch(addTransactionSuccess(response.data.transaction));
      return response.data.transaction;
    } catch (error) {
      throw error.response?.data?.message || 'Failed to add transaction';
    }
  };
};

// Update a transaction
export const updateTransaction = (transactionData) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Update mock transaction
        dispatch(updateTransactionSuccess(transactionData));
        return transactionData;
      }
      
      // API call for real backend
      const response = await axios.put(`${API_URL}/transactions/${transactionData.id}`, transactionData);
      dispatch(updateTransactionSuccess(response.data.transaction));
      return response.data.transaction;
    } catch (error) {
      throw error.response?.data?.message || 'Failed to update transaction';
    }
  };
};

// Delete a transaction
export const deleteTransaction = (transactionId) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Delete mock transaction
        dispatch(deleteTransactionSuccess(transactionId));
        return;
      }
      
      // API call for real backend
      await axios.delete(`${API_URL}/transactions/${transactionId}`);
      dispatch(deleteTransactionSuccess(transactionId));
    } catch (error) {
      throw error.response?.data?.message || 'Failed to delete transaction';
    }
  };
};

// Fetch similar transactions
export const fetchSimilarTransactions = (description) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Filter mock transactions that contain the description
        const similar = MOCK_TRANSACTIONS.filter(t => 
          t.description.toLowerCase().includes(description.toLowerCase())
        );
        
        dispatch(fetchSimilarTransactionsSuccess(similar));
        return;
      }
      
      // API call for real backend
      const response = await axios.get(`${API_URL}/transactions/similar?query=${encodeURIComponent(description)}`);
      dispatch(fetchSimilarTransactionsSuccess(response.data.transactions));
    } catch (error) {
      console.error('Failed to fetch similar transactions:', error);
      // Don't dispatch error, just don't show similar transactions
      dispatch(fetchSimilarTransactionsSuccess([]));
    }
  };
};

// src/store/actions/recommendationActions.js
import axios from 'axios';
import { 
  fetchRecommendationsStart,
  fetchRecommendationsSuccess,
  fetchRecommendationsFail,
  implementRecommendationSuccess,
  dismissRecommendationSuccess
} from '../reducers/recommendationReducer';
import { API_URL } from '../../config';

// Sample recommendation data for development/testing
const MOCK_RECOMMENDATIONS = [
  {
    id: 'rec-1',
    type: 'spending_alert',
    title: 'High Spending on Food',
    message: 'Your food spending is 30% higher than last month. Consider setting a budget for dining out.',
    priority: 8,
    potential_savings: 150.00,
    implementation_progress: 0,
    implemented: false,
    details: 'You spent $450 on food this month compared to an average of $350 in previous months.',
    implementation_steps: [
      'Review your recent food transactions',
      'Set a weekly budget for dining out',
      'Plan meals in advance to avoid unnecessary takeout'
    ]
  },
  {
    id: 'rec-2',
    type: 'savings_opportunity',
    title: 'Increase Savings Rate',
    message: 'You could save an additional $200 per month by reducing discretionary spending.',
    priority: 7,
    potential_savings: 200.00,
    implementation_progress: 0,
    implemented: false,
    details: 'Based on your income and essential expenses, there\'s potential to increase your savings rate by 5%.',
    implementation_steps: [
      'Identify non-essential spending categories',
      'Set a target to reduce spending in these areas by 10%',
      'Automate transfers to your savings account'
    ]
  },
  {
    id: 'rec-3',
    type: 'subscription_alert',
    title: 'Unused Subscriptions',
    message: 'You have 3 active subscriptions that you haven\'t used in the last month.',
    priority: 6,
    potential_savings: 35.97,
    implementation_progress: 0,
    implemented: false,
    details: 'Your accounts show regular payments for these services, but our analysis suggests you\'re not actively using them.',
    implementation_steps: [
      'Review the list of inactive subscriptions',
      'Decide which ones to cancel',
      'Set calendar reminders for free trial expirations'
    ]
  },
  {
    id: 'rec-4',
    type: 'budget_recommendation',
    title: 'Create a Transportation Budget',
    message: 'Your transportation costs vary significantly month to month. A budget could help you plan better.',
    priority: 5,
    potential_savings: 75.00,
    implementation_progress: 0,
    implemented: false,
    details: 'Over the past 6 months, your transportation expenses have ranged from $120 to $350 per month.',
    implementation_steps: [
      'Calculate your average monthly transportation cost',
      'Set a realistic monthly budget',
      'Track expenses against this budget'
    ]
  },
  {
    id: 'rec-5',
    type: 'debt_management',
    title: 'Optimize Credit Card Payments',
    message: 'Paying off your highest interest credit card first could save you $240 in interest this year.',
    priority: 9,
    potential_savings: 240.00,
    implementation_progress: 0,
    implemented: false,
    details: 'You currently have balances on multiple credit cards with different interest rates.',
    implementation_steps: [
      'Focus on paying off the card with 24.99% APR first',
      'Make minimum payments on other cards',
      'Once highest interest card is paid off, move to the next highest'
    ]
  }
];

// Fetch all recommendations
export const fetchRecommendations = () => {
  return async dispatch => {
    dispatch(fetchRecommendationsStart());
    
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Return mock data after a small delay to simulate network request
        setTimeout(() => {
          dispatch(fetchRecommendationsSuccess(MOCK_RECOMMENDATIONS));
        }, 500);
        return;
      }
      
      // API call for real backend
      const response = await axios.get(`${API_URL}/recommendations`);
      dispatch(fetchRecommendationsSuccess(response.data.recommendations));
    } catch (error) {
      dispatch(fetchRecommendationsFail(error.response?.data?.message || 'Failed to fetch recommendations'));
    }
  };
};

// Mark a recommendation as implemented
export const implementRecommendation = (recommendationId) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        dispatch(implementRecommendationSuccess(recommendationId));
        return;
      }
      
      // API call for real backend
      await axios.post(`${API_URL}/recommendations/${recommendationId}/implement`);
      dispatch(implementRecommendationSuccess(recommendationId));
    } catch (error) {
      throw error.response?.data?.message || 'Failed to implement recommendation';
    }
  };
};

// Dismiss a recommendation
export const dismissRecommendation = (recommendationId) => {
  return async dispatch => {
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        dispatch(dismissRecommendationSuccess(recommendationId));
        return;
      }
      
      // API call for real backend
      await axios.post(`${API_URL}/recommendations/${recommendationId}/dismiss`);
      dispatch(dismissRecommendationSuccess(recommendationId));
    } catch (error) {
      throw error.response?.data?.message || 'Failed to dismiss recommendation';
    }
  };
};

// src/store/actions/forecastActions.js
import axios from 'axios';
import { 
  fetchForecastsStart,
  fetchForecastsSuccess,
  fetchForecastsFail
} from '../reducers/forecastReducer';
import { API_URL } from '../../config';

// Generate mock forecast data
const generateMockForecast = (categoryName, days = 90, startDate = new Date()) => {
  const forecast = [];
  let cumulative = 0;
  
  // Set base amount based on category
  let baseAmount;
  switch (categoryName) {
    case 'food':
      baseAmount = -50;
      break;
    case 'transport':
      baseAmount = -30;
      break;
    case 'utilities':
      baseAmount = -100;
      break;
    case 'entertainment':
      baseAmount = -40;
      break;
    case 'shopping':
      baseAmount = -60;
      break;
    case 'health':
      baseAmount = -20;
      break;
    case 'housing':
      baseAmount = -1200;
      break;
    case 'income':
      baseAmount = 3000;
      break;
    case 'all':
      baseAmount = -150;
      break;
    default:
      baseAmount = -25;
  }
  
  // Special case for income - only generate on 1st and 15th
  const isIncome = categoryName === 'income';
  
  for (let i = 0; i < days; i++) {
    const currentDate = new Date(startDate);
    currentDate.setDate(startDate.getDate() + i);
    
    // For income, only add entries on 1st and 15th of month
    if (isIncome) {
      const day = currentDate.getDate();
      if (day !== 1 && day !== 15) {
        continue;
      }
    }
    
    // Add some randomness to the amount
    const randomFactor = 0.2; // 20% variation
    const randomAmount = baseAmount * (1 + (Math.random() * 2 - 1) * randomFactor);
    
    // For housing, only add on 1st of month
    if (categoryName === 'housing') {
      const day = currentDate.getDate();
      if (day !== 1) {
        continue;
      }
    }
    
    // Apply seasonal factors
    const month = currentDate.getMonth();
    let seasonalFactor = 1;
    
    if (categoryName === 'utilities' && (month === 0 || month === 1 || month === 11)) {
      // Higher utilities in winter months
      seasonalFactor = 1.3;
    } else if (categoryName === 'entertainment' && (month >= 5 && month <= 8)) {
      // Higher entertainment in summer
      seasonalFactor = 1.2;
    }
    
    const amount = randomAmount * seasonalFactor;
    cumulative += amount;
    
    // Calculate confidence interval
    const confidence = 0.1; // 10% of amount
    const lower_bound = amount * (1 - confidence);
    const upper_bound = amount * (1 + confidence);
    
    forecast.push({
      date: currentDate.toISOString().split('T')[0],
      amount,
      cumulative,
      lower_bound,
      upper_bound
    });
  }
  
  return forecast;
};

// Sample forecast data for development/testing
const generateMockForecasts = () => {
  const categories = ['food', 'transport', 'utilities', 'entertainment', 'shopping', 'health', 'housing', 'income', 'all'];
  const forecasts = {};
  
  categories.forEach(category => {
    forecasts[category] = generateMockForecast(category);
  });
  
  return forecasts;
};

// Fetch all forecasts
export const fetchForecasts = () => {
  return async dispatch => {
    dispatch(fetchForecastsStart());
    
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Return mock data after a small delay to simulate network request
        setTimeout(() => {
          const forecasts = generateMockForecasts();
          dispatch(fetchForecastsSuccess(forecasts));
        }, 1000);
        return;
      }
      
      // API call for real backend
      const response = await axios.get(`${API_URL}/forecasts`);
      dispatch(fetchForecastsSuccess(response.data.forecasts));
    } catch (error) {
      dispatch(fetchForecastsFail(error.response?.data?.message || 'Failed to fetch forecasts'));
    }
  };
};