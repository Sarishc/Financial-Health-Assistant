// src/utils/categories.js
import { APP_CONFIG } from '../config';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

/**
 * Get the color associated with a category
 * @param {string} categoryName - Name of the category
 * @param {number} opacity - Optional opacity (0-1, default: 1)
 * @returns {string} - Color for the category (with opacity if specified)
 */
export const getCategoryColor = (categoryName, opacity = 1) => {
  const baseColor = APP_CONFIG.categories[categoryName]?.color || APP_CONFIG.categories.other.color;
  
  if (opacity === 1) {
    return baseColor;
  }
  
  // Convert hex color to rgba
  const r = parseInt(baseColor.slice(1, 3), 16);
  const g = parseInt(baseColor.slice(3, 5), 16);
  const b = parseInt(baseColor.slice(5, 7), 16);
  
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
};

/**
 * Get the icon name associated with a category
 * @param {string} categoryName - Name of the category
 * @returns {string} - Icon name for the category
 */
export const getCategoryIcon = (categoryName) => {
  return APP_CONFIG.categories[categoryName]?.icon || APP_CONFIG.categories.other.icon;
};

/**
 * Get the description associated with a category
 * @param {string} categoryName - Name of the category
 * @returns {string} - Description for the category
 */
export const getCategoryDescription = (categoryName) => {
  return APP_CONFIG.categories[categoryName]?.description || 'Miscellaneous expenses';
};

/**
 * Get all available categories
 * @returns {Array} - Array of category objects with name, color, icon, and description
 */
export const getAllCategories = () => {
  return Object.entries(APP_CONFIG.categories).map(([name, details]) => ({
    name,
    ...details
  }));
};

/**
 * Format a category name for display (e.g., 'food' -> 'Food')
 * @param {string} categoryName - Name of the category
 * @returns {string} - Formatted category name
 */
export const formatCategoryName = (categoryName) => {
  if (!categoryName) return 'Uncategorized';
  
  return categoryName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

/**
 * Automatically assign a category based on transaction description
 * This is a simple implementation; a real app would use ML/NLP
 * @param {string} description - Transaction description
 * @returns {string} - Best matching category name
 */
export const autoCategorizeTransaction = (description) => {
  if (!description) return 'other';
  
  const descLower = description.toLowerCase();
  
  // Simple keyword matching
  const categoryKeywords = {
    food: ['grocery', 'restaurant', 'food', 'cafe', 'coffee', 'pizza', 'burger', 'deli', 'bakery'],
    transport: ['gas', 'uber', 'lyft', 'taxi', 'car', 'auto', 'parking', 'transit', 'train', 'bus', 'subway'],
    utilities: ['electric', 'water', 'gas', 'utility', 'phone', 'internet', 'cable', 'bill'],
    entertainment: ['movie', 'theatre', 'netflix', 'spotify', 'hulu', 'disney', 'game', 'book', 'music'],
    shopping: ['amazon', 'walmart', 'target', 'purchase', 'store', 'shop', 'mall', 'clothing', 'electronics'],
    health: ['doctor', 'hospital', 'pharmacy', 'medical', 'dental', 'health', 'clinic', 'fitness', 'gym'],
    housing: ['rent', 'mortgage', 'home', 'apartment', 'lease', 'property', 'housing', 'insurance'],
    income: ['salary', 'payment', 'direct deposit', 'deposit', 'income', 'payroll', 'wage', 'refund'],
  };
  
  for (const [category, keywords] of Object.entries(categoryKeywords)) {
    if (keywords.some(keyword => descLower.includes(keyword))) {
      return category;
    }
  }
  
  return 'other';
};

/**
 * Calculate total amount by category from transactions
 * @param {Array} transactions - Array of transaction objects
 * @returns {Object} - Object with categories as keys and total amounts as values
 */
export const calculateCategorySummary = (transactions) => {
  if (!transactions || !transactions.length) return {};
  
  const summary = {};
  
  transactions.forEach(transaction => {
    const category = transaction.category || 'other';
    if (!summary[category]) {
      summary[category] = 0;
    }
    summary[category] += transaction.amount;
  });
  
  return summary;
};

/**
 * Get the top spending categories based on total amount
 * @param {Object} categorySummary - Category summary object from calculateCategorySummary
 * @param {number} limit - Maximum number of categories to return
 * @returns {Array} - Array of category objects sorted by amount
 */
export const getTopSpendingCategories = (categorySummary, limit = 5) => {
  if (!categorySummary) return [];
  
  // Filter out income (positive amounts) and convert to array
  const categories = Object.entries(categorySummary)
    .filter(([category, amount]) => amount < 0 && category !== 'income')
    .map(([category, amount]) => ({
      category,
      amount: Math.abs(amount),
      color: getCategoryColor(category),
      icon: getCategoryIcon(category),
    }));
  
  // Sort by amount (highest first) and limit
  return categories
    .sort((a, b) => b.amount - a.amount)
    .slice(0, limit);
};

/**
 * Create data for a category pie chart
 * @param {Object} categorySummary - Category summary object
 * @param {boolean} expensesOnly - Whether to include only expenses (negative amounts)
 * @returns {Array} - Array of data points for a pie chart
 */
export const createCategoryPieChartData = (categorySummary, expensesOnly = true) => {
  if (!categorySummary) return [];
  
  return Object.entries(categorySummary)
    .filter(([category, amount]) => {
      // Filter out income if expensesOnly is true
      return !expensesOnly || (amount < 0 && category !== 'income');
    })
    .map(([category, amount]) => ({
      x: formatCategoryName(category),
      y: Math.abs(amount),
      color: getCategoryColor(category),
    }));
};