// src/screens/auth/ForgotPasswordScreen.js
import React, { useState } from 'react';
import { View, StyleSheet, KeyboardAvoidingView, Platform, ScrollView } from 'react-native';
import { Text, TextInput, Button, Headline, HelperText, Snackbar, IconButton } from 'react-native-paper';
import axios from 'axios';
import { API_URL } from '../../config';

const ForgotPasswordScreen = ({ navigation }) => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);
  const [snackbarVisible, setSnackbarVisible] = useState(false);
  
  const emailError = email.length > 0 && !email.includes('@');
  
  const handleResetPassword = async () => {
    if (emailError) {
      setMessage('Please enter a valid email address');
      setIsError(true);
      setSnackbarVisible(true);
      return;
    }
    
    if (!email) {
      setMessage('Please enter your email address');
      setIsError(true);
      setSnackbarVisible(true);
      return;
    }
    
    setLoading(true);
    setMessage('');
    setIsError(false);
    
    try {
      // For development/testing
      if (__DEV__ && process.env.REACT_APP_USE_MOCK_DATA === 'true') {
        // Simulate API call
        setTimeout(() => {
          setMessage('Password reset instructions have been sent to your email');
          setIsError(false);
          setSnackbarVisible(true);
          setLoading(false);
        }, 1500);
        return;
      }
      
      // API call for real backend
      await axios.post(`${API_URL}/auth/forgot-password`, { email });
      
      setMessage('Password reset instructions have been sent to your email');
      setIsError(false);
      setSnackbarVisible(true);
    } catch (error) {
      setMessage(error.response?.data?.message || 'Failed to send reset instructions. Please try again.');
      setIsError(true);
      setSnackbarVisible(true);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <IconButton
          icon="arrow-left"
          size={24}
          style={styles.backButton}
          onPress={() => navigation.goBack()}
        />
        
        <View style={styles.content}>
          <Headline style={styles.title}>Forgot Password</Headline>
          <Text style={styles.subtitle}>
            Enter your email address and we'll send you instructions to reset your password
          </Text>
          
          <TextInput
            label="Email"
            value={email}
            onChangeText={setEmail}
            style={styles.input}
            autoCapitalize="none"
            autoCompleteType="email"
            keyboardType="email-address"
            error={emailError}
          />
          {emailError && (
            <HelperText type="error" visible={emailError}>
              Please enter a valid email address
            </HelperText>
          )}
          
          <Button
            mode="contained"
            onPress={handleResetPassword}
            style={styles.resetButton}
            contentStyle={styles.buttonContent}
            loading={loading}
            disabled={loading}
          >
            Send Reset Instructions
          </Button>
          
          <Button
            mode="text"
            onPress={() => navigation.navigate('Login')}
            style={styles.backToLoginButton}
          >
            Back to Login
          </Button>
        </View>
      </ScrollView>
      
      <Snackbar
        visible={snackbarVisible}
        onDismiss={() => setSnackbarVisible(false)}
        duration={3000}
        action={{
          label: 'OK',
          onPress: () => setSnackbarVisible(false),
        }}
        style={isError ? styles.errorSnackbar : styles.successSnackbar}
      >
        {message}
      </Snackbar>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  scrollContainer: {
    flexGrow: 1,
    paddingBottom: 24,
  },
  backButton: {
    marginTop: 16,
    marginLeft: 16,
  },
  content: {
    flex: 1,
    padding: 24,
    justifyContent: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#757575',
    marginBottom: 32,
  },
  input: {
    marginBottom: 16,
    backgroundColor: 'transparent',
  },
  resetButton: {
    marginTop: 16,
    marginBottom: 24,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  backToLoginButton: {
    alignSelf: 'center',
  },
  successSnackbar: {
    backgroundColor: '#4CAF50',
  },
  errorSnackbar: {
    backgroundColor: '#F44336',
  },
});

export default ForgotPasswordScreen;