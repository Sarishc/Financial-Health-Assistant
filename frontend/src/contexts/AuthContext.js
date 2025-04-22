// src/contexts/AuthContext.js
import React from 'react';

// Create the authentication context
export const AuthContext = React.createContext({
  signIn: async () => {},
  signOut: async () => {},
  signUp: async () => {},
});