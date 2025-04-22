// src/store/store.js
import { configureStore } from '@reduxjs/toolkit';
import thunk from 'redux-thunk';
import authReducer from './reducers/authReducer';
import transactionReducer from './reducers/transactionReducer';
import recommendationReducer from './reducers/recommendationReducer';
import forecastReducer from './reducers/forecastReducer';

const store = configureStore({
  reducer: {
    auth: authReducer,
    transactions: transactionReducer,
    recommendations: recommendationReducer,
    forecasts: forecastReducer,
  },
  middleware: [thunk],
});

export default store;