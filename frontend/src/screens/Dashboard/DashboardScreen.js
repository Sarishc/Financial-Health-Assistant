// src/screens/dashboard/DashboardScreen.js
import React, { useState, useEffect } from 'react';
import { View, ScrollView, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { Card, Title, Paragraph, Button, Divider, List, Avatar } from 'react-native-paper';
import { VictoryPie, VictoryChart, VictoryLine, VictoryAxis } from 'victory-native';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useSelector, useDispatch } from 'react-redux';
import { fetchTransactions } from '../../store/actions/transactionActions';
import { fetchRecommendations } from '../../store/actions/recommendationActions';
import { formatCurrency } from '../../utils/formatters';
import { getCategoryColor } from '../../utils/categories';
import Loading from '../../components/common/Loading';

const DashboardScreen = ({ navigation }) => {
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  
  const { transactions, categorySummary } = useSelector(state => state.transactions);
  const { recommendations } = useSelector(state => state.recommendations);
  const { user } = useSelector(state => state.auth);
  
  useEffect(() => {
    loadData();
  }, []);
  
  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        dispatch(fetchTransactions()),
        dispatch(fetchRecommendations())
      ]);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };
  
  const getMonthlyBalance = () => {
    const currentMonth = new Date().getMonth();
    const currentYear = new Date().getFullYear();
    
    let income = 0;
    let expenses = 0;
    
    transactions.forEach(transaction => {
      const transDate = new Date(transaction.transaction_date);
      if (transDate.getMonth() === currentMonth && transDate.getFullYear() === currentYear) {
        if (transaction.amount > 0) {
          income += transaction.amount;
        } else {
          expenses += Math.abs(transaction.amount);
        }
      }
    });
    
    return { income, expenses, balance: income - expenses };
  };
  
  if (loading) {
    return <Loading />;
  }
  
  const { income, expenses, balance } = getMonthlyBalance();
  const monthName = new Date().toLocaleString('default', { month: 'long' });
  
  // Prepare data for category pie chart
  const categoryData = Object.keys(categorySummary || {}).map(category => ({
    x: category,
    y: Math.abs(categorySummary[category]),
    color: getCategoryColor(category)
  }));
  
  // Recent transactions (last 5)
  const recentTransactions = [...transactions]
    .sort((a, b) => new Date(b.transaction_date) - new Date(a.transaction_date))
    .slice(0, 5);
  
  // Top 3 recommendations
  const topRecommendations = [...(recommendations || [])]
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 3);
  
  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      refreshing={refreshing}
      onRefresh={handleRefresh}
    >
      {/* Greeting Card */}
      <Card style={styles.greetingCard}>
        <Card.Content>
          <Title>Hello, {user?.name || 'there'}!</Title>
          <Paragraph>Here's your financial summary for {monthName}</Paragraph>
        </Card.Content>
      </Card>
      
      {/* Monthly Summary Card */}
      <Card style={styles.card}>
        <Card.Content>
          <Title>Monthly Overview</Title>
          <View style={styles.balanceRow}>
            <View style={styles.balanceItem}>
              <Paragraph>Income</Paragraph>
              <Text style={styles.incomeText}>{formatCurrency(income)}</Text>
            </View>
            <View style={styles.balanceItem}>
              <Paragraph>Expenses</Paragraph>
              <Text style={styles.expenseText}>{formatCurrency(expenses)}</Text>
            </View>
            <View style={styles.balanceItem}>
              <Paragraph>Balance</Paragraph>
              <Text style={balance >= 0 ? styles.incomeText : styles.expenseText}>
                {formatCurrency(balance)}
              </Text>
            </View>
          </View>
        </Card.Content>
      </Card>
      
      {/* Category Spending Card */}
      <Card style={styles.card}>
        <Card.Title title="Top Spending Categories" />
        <Card.Content>
          <View style={styles.chartContainer}>
            {categoryData.length > 0 ? (
              <VictoryPie
                data={categoryData}
                width={300}
                height={300}
                colorScale={categoryData.map(d => d.color)}
                labels={({ datum }) => `${datum.x}\n${formatCurrency(datum.y)}`}
                style={{ labels: { fontSize: 12, fill: 'black' } }}
                innerRadius={40}
              />
            ) : (
              <Text style={styles.noDataText}>No category data available</Text>
            )}
          </View>
          <Button 
            mode="contained" 
            onPress={() => navigation.navigate('Reports')}
            style={styles.button}
          >
            View Detailed Reports
          </Button>
        </Card.Content>
      </Card>
      
      {/* Recent Transactions Card */}
      <Card style={styles.card}>
        <Card.Title 
          title="Recent Transactions" 
          right={() => (
            <Button 
              mode="text" 
              onPress={() => navigation.navigate('Transactions')}
            >
              See All
            </Button>
          )}
        />
        <Card.Content>
          {recentTransactions.length > 0 ? (
            recentTransactions.map(transaction => (
              <TouchableOpacity
                key={transaction.id || transaction.transaction_date}
                onPress={() => navigation.navigate('TransactionDetail', { transaction })}
              >
                <List.Item
                  title={transaction.description}
                  description={new Date(transaction.transaction_date).toLocaleDateString()}
                  left={props => (
                    <Avatar.Icon 
                      {...props} 
                      size={40} 
                      icon={transaction.amount > 0 ? "arrow-down" : "arrow-up"} 
                      color="white"
                      style={{
                        backgroundColor: transaction.amount > 0 ? '#4CAF50' : '#F44336'
                      }}
                    />
                  )}
                  right={props => (
                    <Text style={transaction.amount > 0 ? styles.incomeText : styles.expenseText}>
                      {formatCurrency(Math.abs(transaction.amount))}
                    </Text>
                  )}
                />
                <Divider />
              </TouchableOpacity>
            ))
          ) : (
            <Text style={styles.noDataText}>No recent transactions</Text>
          )}
        </Card.Content>
      </Card>
      
      {/* Recommendations Card */}
      <Card style={styles.card}>
        <Card.Title title="Recommendations" />
        <Card.Content>
          {topRecommendations.length > 0 ? (
            topRecommendations.map((recommendation, index) => (
              <View key={index} style={styles.recommendationItem}>
                <View style={styles.recommendationPriority}>
                  <View 
                    style={[
                      styles.priorityIndicator, 
                      { backgroundColor: recommendation.priority > 7 ? '#F44336' : recommendation.priority > 4 ? '#FF9800' : '#4CAF50' }
                    ]} 
                  />
                </View>
                <View style={styles.recommendationContent}>
                  <Text style={styles.recommendationText}>{recommendation.message}</Text>
                  <Text style={styles.recommendationType}>{recommendation.type.replace('_', ' ')}</Text>
                </View>
              </View>
            ))
          ) : (
            <Text style={styles.noDataText}>No recommendations available</Text>
          )}
          <Button 
            mode="contained" 
            onPress={() => navigation.navigate('Recommendations')}
            style={styles.button}
          >
            View All Recommendations
          </Button>
        </Card.Content>
      </Card>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    padding: 16,
  },
  greetingCard: {
    marginBottom: 16,
    backgroundColor: '#E3F2FD',
  },
  card: {
    marginBottom: 16,
  },
  balanceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  balanceItem: {
    alignItems: 'center',
    flex: 1,
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
  chartContainer: {
    alignItems: 'center',
    marginVertical: 16,
  },
  button: {
    marginTop: 8,
  },
  noDataText: {
    textAlign: 'center',
    marginVertical: 16,
    color: '#757575',
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  recommendationPriority: {
    width: 24,
    alignItems: 'center',
    paddingTop: 2,
  },
  priorityIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  recommendationContent: {
    flex: 1,
    marginLeft: 8,
  },
  recommendationText: {
    fontSize: 14,
  },
  recommendationType: {
    fontSize: 12,
    color: '#757575',
    marginTop: 4,
  },
});

export default DashboardScreen;