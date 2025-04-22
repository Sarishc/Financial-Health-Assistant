// src/App.js
import React, { useState, useEffect } from 'react';
import { StatusBar, LogBox, View, ActivityIndicator } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { Provider as PaperProvider, DefaultTheme } from 'react-native-paper';
import { Provider as StoreProvider } from 'react-redux';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Import navigators
import AppNavigator from './navigation/AppNavigator';
import AuthNavigator from './navigation/AuthNavigator';

// Import store
import store from './store/store';

// Import auth context
import { AuthContext } from './contexts/AuthContext';

// Ignore specific deprecation warnings
LogBox.ignoreLogs([
  'ViewPropTypes will be removed',
  'ColorPropType will be removed',
]);

// Define custom theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2196F3',
    accent: '#03A9F4',
    background: '#f5f5f5',
    surface: '#ffffff',
    error: '#F44336',
    text: '#212121',
    onSurface: '#212121',
    disabled: '#9E9E9E',
    placeholder: '#9E9E9E',
    backdrop: 'rgba(0, 0, 0, 0.5)',
    notification: '#FF9800',
  },
};

const App = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [userToken, setUserToken] = useState(null);
  
  // Define authentication context
  const authContext = React.useMemo(() => ({
    signIn: async (token) => {
      setIsLoading(true);
      try {
        await AsyncStorage.setItem('userToken', token);
        setUserToken(token);
      } catch (error) {
        console.error('Error storing token:', error);
      }
      setIsLoading(false);
    },
    signOut: async () => {
      setIsLoading(true);
      try {
        await AsyncStorage.removeItem('userToken');
        setUserToken(null);
      } catch (error) {
        console.error('Error removing token:', error);
      }
      setIsLoading(false);
    },
    signUp: async (token) => {
      setIsLoading(true);
      try {
        await AsyncStorage.setItem('userToken', token);
        setUserToken(token);
      } catch (error) {
        console.error('Error storing token:', error);
      }
      setIsLoading(false);
    },
  }), []);
  
  // Check for existing token on app start
  useEffect(() => {
    const bootstrapAsync = async () => {
      let token = null;
      try {
        token = await AsyncStorage.getItem('userToken');
      } catch (error) {
        console.error('Error retrieving token:', error);
      }
      setUserToken(token);
      setIsLoading(false);
    };
    
    bootstrapAsync();
  }, []);
  
  // Show loading screen
  if (isLoading) {
    return (
      <PaperProvider theme={theme}>
        <StatusBar backgroundColor={theme.colors.primary} barStyle="light-content" />
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
        </View>
      </PaperProvider>
    );
  }
  
  return (
    <StoreProvider store={store}>
      <AuthContext.Provider value={authContext}>
        <PaperProvider theme={theme}>
          <StatusBar backgroundColor={theme.colors.primary} barStyle="light-content" />
          <NavigationContainer>
            {userToken ? (
              <AppNavigator />
            ) : (
              <AuthNavigator />
            )}
          </NavigationContainer>
        </PaperProvider>
      </AuthContext.Provider>
    </StoreProvider>
  );
};

export default App;