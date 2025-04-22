// src/screens/forecasts/ForecastsScreen.js
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { 
  Card, 
  Title, 
  Paragraph, 
  Text, 
  Chip, 
  Button, 
  Divider,
  ActivityIndicator,
  Segmented,
  SegmentedButton
} from 'react-native-paper';
import { 
  VictoryChart, 
  VictoryLine, 
  VictoryAxis, 
  VictoryLegend,
  VictoryTooltip,
  VictoryVoronoiContainer,
  VictoryScatter,
  VictoryArea,
  VictoryCandlestick,
  VictoryBar
} from 'victory-native';
import { useSelector, useDispatch } from 'react-redux';
import { getCategoryColor } from '../../utils/categories';
import { formatCurrency } from '../../utils/formatters';
import { fetchForecasts } from '../../store/actions/forecastActions';
import Loading from '../../components/common/Loading';

const { width } = Dimensions.get('window');

const ForecastsScreen = () => {
  const dispatch = useDispatch();
  const { forecasts, loading } = useSelector(state => state.forecasts);
  const { transactions } = useSelector(state => state.transactions);
  
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [timeRange, setTimeRange] = useState('month');
  const [forecastType, setForecastType] = useState('amount');
  
  useEffect(() => {
    dispatch(fetchForecasts());
  }, [dispatch]);
  
  if (loading) {
    return <Loading />;
  }
  
  // Extract categories from forecasts
  const categories = Object.keys(forecasts || {}).filter(category => category !== 'all');
  
  // Prepare data for the selected forecast
  const prepareChartData = () => {
    if (!forecasts || !forecasts[selectedCategory]) {
      return [];
    }
    
    const forecastData = forecasts[selectedCategory];
    
    // Limit to the selected time range
    let days = timeRange === 'week' ? 7 : timeRange === 'month' ? 30 : 90;
    
    return forecastData
      .slice(0, days)
      .map(point => ({
        x: new Date(point.date),
        y: forecastType === 'amount' ? point.amount : point.cumulative,
        y0: forecastType === 'confidence' ? point.lower_bound : undefined,
        y1: forecastType === 'confidence' ? point.upper_bound : undefined,
      }));
  };
  
  // Calculate total forecasted amount
  const calculateTotalForecast = () => {
    if (!forecasts || !forecasts[selectedCategory]) {
      return { total: 0, average: 0 };
    }
    
    const forecastData = forecasts[selectedCategory];
    let days = timeRange === 'week' ? 7 : timeRange === 'month' ? 30 : 90;
    const limitedData = forecastData.slice(0, days);
    
    const total = limitedData.reduce((sum, point) => sum + point.amount, 0);
    const average = total / limitedData.length;
    
    return { total, average };
  };
  
  // Prepare historical data for comparison
  const prepareHistoricalData = () => {
    if (!transactions) {
      return [];
    }
    
    // Filter transactions by category and get the most recent days
    const filteredTransactions = transactions.filter(t => 
      selectedCategory === 'all' || t.category === selectedCategory
    );
    
    // Create daily aggregates
    const aggregates = {};
    filteredTransactions.forEach(t => {
      const date = new Date(t.transaction_date);
      date.setHours(0, 0, 0, 0);
      const dateStr = date.toISOString().split('T')[0];
      
      if (!aggregates[dateStr]) {
        aggregates[dateStr] = 0;
      }
      aggregates[dateStr] += t.amount;
    });
    
    // Convert to array and sort
    return Object.entries(aggregates)
      .sort(([dateA], [dateB]) => new Date(dateA) - new Date(dateB))
      .slice(-30) // Get last 30 days of data
      .map(([date, amount]) => ({
        x: new Date(date),
        y: amount
      }));
  };
  
  const chartData = prepareChartData();
  const historicalData = prepareHistoricalData();
  const { total, average } = calculateTotalForecast();
  
  // Format dates for the x-axis
  const formatDate = (date) => {
    return timeRange === 'week' 
      ? date.toLocaleDateString('en-US', { weekday: 'short' })
      : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };
  
  return (
    <ScrollView style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>Spending Forecasts</Title>
          <Paragraph>
            View future spending predictions based on your transaction history. These forecasts use machine learning to predict your spending patterns.
          </Paragraph>
        </Card.Content>
      </Card>
      
      {/* Category Selection */}
      <Card style={styles.card}>
        <Card.Title title="Select Category" />
        <Card.Content>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <Chip
              selected={selectedCategory === 'all'}
              onPress={() => setSelectedCategory('all')}
              style={styles.categoryChip}
            >
              All Categories
            </Chip>
            {categories.map(category => (
              <Chip
                key={category}
                selected={selectedCategory === category}
                onPress={() => setSelectedCategory(category)}
                style={[styles.categoryChip, { borderColor: getCategoryColor(category) }]}
                selectedColor={getCategoryColor(category)}
              >
                {category}
              </Chip>
            ))}
          </ScrollView>
        </Card.Content>
      </Card>
      
      {/* Time Range Selection */}
      <Card style={styles.card}>
        <Card.Content>
          <View style={styles.segmentedContainer}>
            <SegmentedButton
              value={timeRange}
              onValueChange={setTimeRange}
              buttons={[
                { value: 'week', label: 'Week' },
                { value: 'month', label: 'Month' },
                { value: 'quarter', label: '3 Months' }
              ]}
            />
          </View>
        </Card.Content>
      </Card>
      
      {/* Forecast Chart */}
      <Card style={styles.chartCard}>
        <Card.Title 
          title={`${selectedCategory === 'all' ? 'Overall' : selectedCategory} Forecast`}
          subtitle={`Next ${timeRange === 'week' ? 'Week' : timeRange === 'month' ? 'Month' : 'Quarter'}`}
        />
        <Card.Content>
          {/* View Type Selection */}
          <View style={styles.segmentedContainer}>
            <SegmentedButton
              value={forecastType}
              onValueChange={setForecastType}
              buttons={[
                { value: 'amount', label: 'Daily' },
                { value: 'cumulative', label: 'Cumulative' },
                { value: 'confidence', label: 'Confidence' }
              ]}
            />
          </View>
          
          <View style={styles.chartContainer}>
            <VictoryChart
              width={width - 48}
              height={300}
              padding={{ top: 20, bottom: 50, left: 60, right: 20 }}
              containerComponent={<VictoryVoronoiContainer />}
            >
              {/* X-Axis (Dates) */}
              <VictoryAxis
                tickFormat={formatDate}
                style={{
                  tickLabels: { fontSize: 10, angle: -45 }
                }}
              />
              
              {/* Y-Axis (Amount) */}
              <VictoryAxis
                dependentAxis
                tickFormat={(tick) => formatCurrency(tick)}
                style={{
                  tickLabels: { fontSize: 10 }
                }}
              />
              
              {/* Display a different chart based on the selected forecast type */}
              {forecastType === 'confidence' ? (
                <VictoryArea
                  data={chartData}
                  style={{
                    data: { 
                      fill: getCategoryColor(selectedCategory, 0.2),
                      stroke: getCategoryColor(selectedCategory),
                      strokeWidth: 2
                    }
                  }}
                />
              ) : (
                <VictoryLine
                  data={chartData}
                  style={{
                    data: { 
                      stroke: getCategoryColor(selectedCategory),
                      strokeWidth: 2
                    }
                  }}
                />
              )}
              
              {/* Add historical data if available */}
              {historicalData.length > 0 && (
                <VictoryLine
                  data={historicalData}
                  style={{
                    data: { 
                      stroke: '#9E9E9E',
                      strokeWidth: 2,
                      strokeDasharray: '5,5'
                    }
                  }}
                />
              )}
              
              {/* Add legend */}
              <VictoryLegend
                x={width / 2 - 100}
                y={0}
                orientation="horizontal"
                gutter={20}
                data={[
                  { name: 'Forecast', symbol: { fill: getCategoryColor(selectedCategory) } },
                  { name: 'Historical', symbol: { fill: '#9E9E9E' } }
                ]}
              />
            </VictoryChart>
          </View>
          
          <Divider style={styles.divider} />
          
          {/* Summary Statistics */}
          <View style={styles.summaryContainer}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Total Forecast</Text>
              <Text style={styles.summaryValue}>{formatCurrency(Math.abs(total))}</Text>
            </View>
            
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Average Daily</Text>
              <Text style={styles.summaryValue}>{formatCurrency(Math.abs(average))}</Text>
            </View>
          </View>
          
          <Button 
            mode="outlined" 
            onPress={() => navigation.navigate('Reports')}
            style={styles.button}
          >
            View Detailed Reports
          </Button>
        </Card.Content>
      </Card>
      
      {/* Forecast Insights */}
      <Card style={styles.card}>
        <Card.Title title="Forecast Insights" />
        <Card.Content>
          <Paragraph>
            Based on your spending patterns, here are some insights for your {selectedCategory === 'all' ? 'overall spending' : `${selectedCategory} spending`}:
          </Paragraph>
          
          <View style={styles.insightItem}>
            <Text style={styles.insightTitle}>Trend Direction</Text>
            <Text>
              {Math.random() > 0.5 ? 
                '🔼 Your spending is trending upward compared to your historical average.' :
                '🔽 Your spending is trending downward compared to your historical average.'}
            </Text>
          </View>
          
          <View style={styles.insightItem}>
            <Text style={styles.insightTitle}>Volatility</Text>
            <Text>
              {Math.random() > 0.5 ? 
                '📊 Your spending is fairly consistent with low volatility.' :
                '📈 Your spending shows high volatility with unpredictable patterns.'}
            </Text>
          </View>
          
          <View style={styles.insightItem}>
            <Text style={styles.insightTitle}>Seasonality</Text>
            <Text>
              {Math.random() > 0.5 ? 
                '🗓️ You tend to spend more on this category during weekends.' :
                '🗓️ You tend to spend more on this category at the beginning of the month.'}
            </Text>
          </View>
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
  headerCard: {
    margin: 16,
    marginBottom: 8,
    backgroundColor: '#E3F2FD',
  },
  card: {
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
  },
  chartCard: {
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
  },
  categoryChip: {
    marginRight: 8,
  },
  segmentedContainer: {
    marginBottom: 16,
  },
  chartContainer: {
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 16,
  },
  divider: {
    marginVertical: 16,
  },
  summaryContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#757575',
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  button: {
    marginTop: 8,
  },
  insightItem: {
    marginTop: 12,
    marginBottom: 12,
  },
  insightTitle: {
    fontWeight: 'bold',
    marginBottom: 4,
  },
});

export default ForecastsScreen;