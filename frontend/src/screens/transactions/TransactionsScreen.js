// src/screens/transactions/TransactionsScreen.js
import React, { useState, useEffect, useCallback } from 'react';
import { View, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { 
  Searchbar, 
  Chip, 
  Button, 
  List, 
  Avatar, 
  Text, 
  Divider, 
  FAB,
  Menu,
  Modal,
  Portal,
  Provider,
  Card,
  Title
} from 'react-native-paper';
import DateTimePicker from '@react-native-community/datetimepicker';
import { useSelector, useDispatch } from 'react-redux';
import { fetchTransactions } from '../../store/actions/transactionActions';
import { formatCurrency } from '../../utils/formatters';
import { getCategoryColor } from '../../utils/categories';
import Loading from '../../components/common/Loading';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';

const TransactionsScreen = ({ navigation }) => {
  const dispatch = useDispatch();
  const { transactions, loading } = useSelector(state => state.transactions);
  const [filteredTransactions, setFilteredTransactions] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [dateRange, setDateRange] = useState({
    startDate: new Date(new Date().setMonth(new Date().getMonth() - 1)),
    endDate: new Date()
  });
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [datePickerMode, setDatePickerMode] = useState('start'); // 'start' or 'end'
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  
  // Extract unique categories from transactions
  const categories = [...new Set(transactions.map(t => t.category))].filter(Boolean);
  
  // Load transactions on component mount
  useEffect(() => {
    dispatch(fetchTransactions());
  }, [dispatch]);
  
  // Apply filters when transactions, search query, or filters change
  useEffect(() => {
    applyFilters();
  }, [transactions, searchQuery, dateRange, selectedCategory]);
  
  const applyFilters = () => {
    let filtered = [...transactions];
    
    // Apply date range filter
    filtered = filtered.filter(transaction => {
      const transactionDate = new Date(transaction.transaction_date);
      return transactionDate >= dateRange.startDate && transactionDate <= dateRange.endDate;
    });
    
    // Apply category filter
    if (selectedCategory) {
      filtered = filtered.filter(transaction => transaction.category === selectedCategory);
    }
    
    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(transaction => 
        transaction.description.toLowerCase().includes(query) ||
        (transaction.category && transaction.category.toLowerCase().includes(query))
      );
    }
    
    // Sort by date (most recent first)
    filtered.sort((a, b) => new Date(b.transaction_date) - new Date(a.transaction_date));
    
    setFilteredTransactions(filtered);
  };
  
  const handleRefresh = async () => {
    setRefreshing(true);
    await dispatch(fetchTransactions());
    setRefreshing(false);
  };
  
  const handleSearch = query => {
    setSearchQuery(query);
  };
  
  const handleCategorySelect = category => {
    setSelectedCategory(selectedCategory === category ? null : category);
  };
  
  const handleDateChange = (event, selectedDate) => {
    if (selectedDate) {
      setDateRange(prev => ({
        ...prev,
        [datePickerMode === 'start' ? 'startDate' : 'endDate']: selectedDate
      }));
    }
    setShowDatePicker(false);
  };
  
  const showStartDatePicker = () => {
    setDatePickerMode('start');
    setShowDatePicker(true);
  };
  
  const showEndDatePicker = () => {
    setDatePickerMode('end');
    setShowDatePicker(true);
  };
  
  const resetFilters = () => {
    setSearchQuery('');
    setSelectedCategory(null);
    setDateRange({
      startDate: new Date(new Date().setMonth(new Date().getMonth() - 1)),
      endDate: new Date()
    });
    setShowFilterMenu(false);
  };
  
  const renderTransactionItem = ({ item }) => (
    <TouchableOpacity
      onPress={() => navigation.navigate('TransactionDetail', { transaction: item })}
    >
      <List.Item
        title={item.description}
        description={`${new Date(item.transaction_date).toLocaleDateString()} • ${item.category || 'Uncategorized'}`}
        left={props => (
          <Avatar.Icon 
            {...props} 
            size={40} 
            icon={item.amount > 0 ? "arrow-down" : "arrow-up"} 
            color="white"
            style={{
              backgroundColor: item.amount > 0 ? '#4CAF50' : '#F44336'
            }}
          />
        )}
        right={props => (
          <Text style={item.amount > 0 ? styles.incomeText : styles.expenseText}>
            {formatCurrency(Math.abs(item.amount))}
          </Text>
        )}
      />
      <Divider />
    </TouchableOpacity>
  );
  
  const renderEmptyList = () => (
    <View style={styles.emptyContainer}>
      <Icon name="receipt" size={64} color="#BDBDBD" />
      <Text style={styles.emptyText}>No transactions found</Text>
      <Text style={styles.emptySubtext}>
        {searchQuery || selectedCategory 
          ? 'Try changing your filters'
          : 'Add a transaction to get started'}
      </Text>
      {(!searchQuery && !selectedCategory) && (
        <Button 
          mode="contained" 
          onPress={() => navigation.navigate('AddTransaction')}
          style={styles.emptyButton}
        >
          Add Transaction
        </Button>
      )}
    </View>
  );
  
  if (loading && !refreshing) {
    return <Loading />;
  }
  
  return (
    <Provider>
      <View style={styles.container}>
        <Searchbar
          placeholder="Search transactions"
          onChangeText={handleSearch}
          value={searchQuery}
          style={styles.searchBar}
        />
        
        <View style={styles.filterContainer}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <Chip 
              mode="outlined" 
              onPress={() => setShowFilterMenu(true)}
              style={styles.filterChip}
              icon="filter-variant"
            >
              Filters
            </Chip>
            
            <Chip 
              mode={selectedCategory ? "flat" : "outlined"}
              selected={selectedCategory === null}
              onPress={() => setSelectedCategory(null)}
              style={styles.filterChip}
            >
              All
            </Chip>
            
            {categories.map(category => (
              <Chip
                key={category}
                mode={selectedCategory === category ? "flat" : "outlined"}
                selected={selectedCategory === category}
                onPress={() => handleCategorySelect(category)}
                style={styles.filterChip}
                selectedColor={getCategoryColor(category)}
              >
                {category}
              </Chip>
            ))}
          </ScrollView>
        </View>
        
        <Text style={styles.dateRangeText}>
          {dateRange.startDate.toLocaleDateString()} - {dateRange.endDate.toLocaleDateString()}
        </Text>
        
        <FlatList
          data={filteredTransactions}
          renderItem={renderTransactionItem}
          keyExtractor={(item, index) => `${item.id || item.transaction_date}-${index}`}
          contentContainerStyle={filteredTransactions.length === 0 ? styles.emptyList : null}
          ListEmptyComponent={renderEmptyList}
          refreshing={refreshing}
          onRefresh={handleRefresh}
        />
        
        <FAB
          style={styles.fab}
          icon="plus"
          onPress={() => navigation.navigate('AddTransaction')}
        />
        
        <Menu
          visible={showFilterMenu}
          onDismiss={() => setShowFilterMenu(false)}
          anchor={{ x: 20, y: 100 }}
          style={styles.filterMenu}
        >
          <Menu.Item 
            icon="calendar" 
            onPress={showStartDatePicker} 
            title={`Start Date: ${dateRange.startDate.toLocaleDateString()}`} 
          />
          <Menu.Item 
            icon="calendar" 
            onPress={showEndDatePicker} 
            title={`End Date: ${dateRange.endDate.toLocaleDateString()}`} 
          />
          <Divider />
          <Menu.Item icon="refresh" onPress={resetFilters} title="Reset Filters" />
        </Menu>
        
        {showDatePicker && (
          <DateTimePicker
            value={datePickerMode === 'start' ? dateRange.startDate : dateRange.endDate}
            mode="date"
            display="default"
            onChange={handleDateChange}
          />
        )}
      </View>
    </Provider>
  );
};

const styles = StyleSheet.create ( {
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  searchBar: {
    margin: 16,
    borderRadius: 8,
  },
  filterContainer: {
    marginHorizontal: 16,
    marginBottom: 8,
  },
  filterChip: {
    marginRight: 8,
    marginBottom: 8,
  },
  dateRangeText: {
    marginHorizontal: 16,
    marginBottom: 8,
    color: '#757575',
    fontSize: 12,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#2196F3',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#757575',
    marginTop: 8,
    textAlign: 'center',
  },
  emptyButton: {
    marginTop: 16,
  },
  incomeText: {
    color: '#4CAF50',
    fontWeight: 'bold',
    fontSize: 16,
  },
  expenseText: {
    color: '#F44336',
    fontWeight: 'bold',
    fontSize: 16,
  },
  filterMenu: {
    marginTop: 40,
  },
  emptyList: {
    flexGrow: 1,
  }
})