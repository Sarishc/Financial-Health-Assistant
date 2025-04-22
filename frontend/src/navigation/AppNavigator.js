// src/navigation/AppNavigator.js
import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

// Import screens
import DashboardScreen from '../screens/Dashboard/DashboardScreen';
import TransactionsScreen from '../screens/transactions/TransactionsScreen';
import TransactionDetailScreen from '../screens/transactions/TransactionDetailScreen';
import AddTransactionScreen from '../screens/transactions/AddTransactionScreen';
import ForecastsScreen from '../screens/forecasts/ForecastsScreen';
import RecommendationsScreen from '../screens/recommendations/RecommendationsScreen';
import ReportsScreen from '../screens/reports/ReportsScreen';
import SettingsScreen from '../screens/settings/SettingsScreen';
import ProfileScreen from '../screens/settings/ProfileScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Stack navigators should be defined as separate functions
const DashboardStack = () => (
  <Stack.Navigator>
    <Stack.Screen 
      name="DashboardMain" 
      component={DashboardScreen} 
      options={{ title: 'Dashboard' }}
    />
    <Stack.Screen 
      name="Reports" 
      component={ReportsScreen} 
      options={{ title: 'Reports & Insights' }}
    />
  </Stack.Navigator>
);

const TransactionsStack = () => (
  <Stack.Navigator>
    <Stack.Screen 
      name="TransactionsMain" 
      component={TransactionsScreen} 
      options={{ title: 'Transactions' }}
    />
    <Stack.Screen 
      name="TransactionDetail" 
      component={TransactionDetailScreen} 
      options={{ title: 'Transaction Details' }}
    />
    <Stack.Screen 
      name="AddTransaction" 
      component={AddTransactionScreen} 
      options={{ title: 'Add Transaction' }}
    />
  </Stack.Navigator>
);

const ForecastsStack = () => (
  <Stack.Navigator>
    <Stack.Screen 
      name="ForecastsMain" 
      component={ForecastsScreen} 
      options={{ title: 'Forecasts' }}
    />
  </Stack.Navigator>
);

const RecommendationsStack = () => (
  <Stack.Navigator>
    <Stack.Screen 
      name="RecommendationsMain" 
      component={RecommendationsScreen} 
      options={{ title: 'Recommendations' }}
    />
  </Stack.Navigator>
);

const SettingsStack = () => (
  <Stack.Navigator>
    <Stack.Screen 
      name="SettingsMain" 
      component={SettingsScreen} 
      options={{ title: 'Settings' }}
    />
    <Stack.Screen 
      name="Profile" 
      component={ProfileScreen} 
      options={{ title: 'Profile' }}
    />
  </Stack.Navigator>
);

// Single default export
const AppNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          switch (route.name) {
            case 'Dashboard':
              iconName = focused ? 'view-dashboard' : 'view-dashboard-outline';
              break;
            case 'Transactions':
              iconName = focused ? 'receipt' : 'receipt-outline';
              break;
            case 'Forecasts':
              iconName = focused ? 'chart-line' : 'chart-line-variant';
              break;
            case 'Recommendations':
              iconName = focused ? 'lightbulb-on' : 'lightbulb-outline';
              break;
            case 'Settings':
              iconName = focused ? 'cog' : 'cog-outline';
              break;
            default:
              iconName = 'circle';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
      })}
      tabBarOptions={{
        activeTintColor: '#2196F3',
        inactiveTintColor: 'gray',
      }}
    >
      <Tab.Screen name="Dashboard" component={DashboardStack} />
      <Tab.Screen name="Transactions" component={TransactionsStack} />
      <Tab.Screen name="Forecasts" component={ForecastsStack} />
      <Tab.Screen name="Recommendations" component={RecommendationsStack} />
      <Tab.Screen name="Settings" component={SettingsStack} />
    </Tab.Navigator>
  );
};

export default AppNavigator;