// src/screens/recommendations/RecommendationsScreen.js
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { 
  Card, 
  Title, 
  Paragraph, 
  Text, 
  Chip, 
  Button, 
  Divider,
  List,
  Badge,
  ProgressBar,
  IconButton,
  Portal,
  Dialog,
  Switch
} from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useSelector, useDispatch } from 'react-redux';
import { 
  fetchRecommendations, 
  implementRecommendation,
  dismissRecommendation
} from '../../store/actions/recommendationActions';
import { formatCurrency } from '../../utils/formatters';
import Loading from '../../components/common/Loading';

const RecommendationsScreen = () => {
  const dispatch = useDispatch();
  const { recommendations, loading } = useSelector(state => state.recommendations);
  
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [detailDialogVisible, setDetailDialogVisible] = useState(false);
  const [selectedRecommendation, setSelectedRecommendation] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  
  useEffect(() => {
    dispatch(fetchRecommendations());
  }, [dispatch]);
  
  const handleRefresh = async () => {
    setRefreshing(true);
    await dispatch(fetchRecommendations());
    setRefreshing(false);
  };
  
  const handleImplement = (recommendation) => {
    dispatch(implementRecommendation(recommendation.id));
    setDetailDialogVisible(false);
  };
  
  const handleDismiss = (recommendation) => {
    dispatch(dismissRecommendation(recommendation.id));
    setDetailDialogVisible(false);
  };
  
  const showRecommendationDetails = (recommendation) => {
    setSelectedRecommendation(recommendation);
    setDetailDialogVisible(true);
  };
  
  // Filter recommendations based on selected type
  const getFilteredRecommendations = () => {
    if (!recommendations) return [];
    
    if (selectedFilter === 'all') {
      return recommendations;
    }
    
    return recommendations.filter(rec => rec.type === selectedFilter);
  };
  
  // Get all unique recommendation types
  const getRecommendationTypes = () => {
    if (!recommendations) return [];
    
    const types = new Set(recommendations.map(rec => rec.type));
    return Array.from(types);
  };
  
  if (loading && !refreshing) {
    return <Loading />;
  }
  
  const filteredRecommendations = getFilteredRecommendations();
  const recommendationTypes = getRecommendationTypes();
  
  // Get priority color based on priority value
  const getPriorityColor = (priority) => {
    if (priority >= 8) return '#F44336'; // High priority - red
    if (priority >= 5) return '#FF9800'; // Medium priority - orange
    return '#4CAF50'; // Low priority - green
  };
  
  // Get icon for recommendation type
  const getTypeIcon = (type) => {
    switch (type) {
      case 'spending_alert': return 'alert-circle';
      case 'savings_opportunity': return 'piggy-bank';
      case 'budget_recommendation': return 'cash';
      case 'subscription_alert': return 'refresh';
      case 'income_opportunity': return 'trending-up';
      case 'debt_management': return 'credit-card';
      default: return 'lightbulb-on';
    }
  };
  
  // Get human readable type name
  const getTypeName = (type) => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };
  
  // Render a recommendation item
  const renderRecommendationItem = ({ item }) => (
    <Card style={styles.recommendationCard}>
      <View style={styles.priorityIndicator}>
        <Badge 
          size={24}
          style={[styles.priorityBadge, { backgroundColor: getPriorityColor(item.priority) }]}
        >
          {item.priority}
        </Badge>
      </View>
      
      <Card.Content>
        <View style={styles.typeChipContainer}>
          <Chip 
            icon={() => <Icon name={getTypeIcon(item.type)} size={16} color="#fff" />}
            style={[styles.typeChip, { backgroundColor: getPriorityColor(item.priority) }]}
            textStyle={{ color: '#fff' }}
          >
            {getTypeName(item.type)}
          </Chip>
        </View>
        
        <Title style={styles.recommendationTitle}>{item.title || 'Financial Recommendation'}</Title>
        <Paragraph>{item.message}</Paragraph>
        
        {item.potential_savings > 0 && (
          <View style={styles.savingsContainer}>
            <Text style={styles.savingsLabel}>Potential Savings:</Text>
            <Text style={styles.savingsAmount}>{formatCurrency(item.potential_savings)}</Text>
          </View>
        )}
        
        {item.implementation_progress > 0 && (
          <View style={styles.progressContainer}>
            <Text style={styles.progressLabel}>Implementation Progress</Text>
            <ProgressBar
              progress={item.implementation_progress / 100}
              color={getPriorityColor(item.priority)}
              style={styles.progressBar}
            />
            <Text style={styles.progressText}>{item.implementation_progress}%</Text>
          </View>
        )}
      </Card.Content>
      
      <Card.Actions>
        <Button 
          mode="contained" 
          onPress={() => showRecommendationDetails(item)}
          style={styles.detailsButton}
        >
          Details
        </Button>
        <Button 
          mode={item.implemented ? "outlined" : "contained"} 
          onPress={() => handleImplement(item)}
          disabled={item.implemented}
          style={styles.implementButton}
        >
          {item.implemented ? 'Implemented' : 'Implement'}
        </Button>
      </Card.Actions>
    </Card>
  );
  
  // Render empty state
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <Icon name="trophy" size={64} color="#BDBDBD" />
      <Text style={styles.emptyTitle}>No Recommendations</Text>
      <Text style={styles.emptyText}>
        {selectedFilter === 'all' 
          ? 'You\'re in good financial shape! Check back later for new recommendations.'
          : `No ${getTypeName(selectedFilter)} recommendations available.`}
      </Text>
    </View>
  );
  
  return (
    <View style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>Financial Recommendations</Title>
          <Paragraph>
            Personalized recommendations to improve your financial health based on your transaction history and spending patterns.
          </Paragraph>
        </Card.Content>
      </Card>
      
      {/* Filter Chips */}
      <View style={styles.filtersContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <Chip
            selected={selectedFilter === 'all'}
            onPress={() => setSelectedFilter('all')}
            style={styles.filterChip}
          >
            All
          </Chip>
          
          {recommendationTypes.map(type => (
            <Chip
              key={type}
              selected={selectedFilter === type}
              onPress={() => setSelectedFilter(type)}
              style={styles.filterChip}
              icon={() => <Icon name={getTypeIcon(type)} size={16} color={selectedFilter === type ? '#fff' : '#000'} />}
            >
              {getTypeName(type)}
            </Chip>
          ))}
        </ScrollView>
      </View>
      
      {/* Recommendations List */}
      <FlatList
        data={filteredRecommendations}
        renderItem={renderRecommendationItem}
        keyExtractor={(item, index) => `recommendation-${item.id || index}`}
        contentContainerStyle={styles.listContainer}
        ListEmptyComponent={renderEmptyState}
        refreshing={refreshing}
        onRefresh={handleRefresh}
      />
      
      {/* Recommendation Detail Dialog */}
      <Portal>
        <Dialog
          visible={detailDialogVisible}
          onDismiss={() => setDetailDialogVisible(false)}
          style={styles.dialog}
        >
          {selectedRecommendation && (
            <>
              <Dialog.Title>{selectedRecommendation.title || 'Recommendation Details'}</Dialog.Title>
              <Dialog.Content>
                <View style={styles.dialogTypeContainer}>
                  <Chip 
                    icon={() => <Icon name={getTypeIcon(selectedRecommendation.type)} size={16} color="#fff" />}
                    style={[styles.typeChip, { backgroundColor: getPriorityColor(selectedRecommendation.priority) }]}
                    textStyle={{ color: '#fff' }}
                  >
                    {getTypeName(selectedRecommendation.type)}
                  </Chip>
                  <Text style={styles.priorityText}>
                    Priority: {selectedRecommendation.priority}/10
                  </Text>
                </View>
                
                <Divider style={styles.divider} />
                
                <Paragraph style={styles.detailMessage}>
                  {selectedRecommendation.message}
                </Paragraph>
                
                {selectedRecommendation.details && (
                  <Paragraph style={styles.detailParagraph}>
                    {selectedRecommendation.details}
                  </Paragraph>
                )}
                
                {selectedRecommendation.potential_savings > 0 && (
                  <View style={styles.detailRow}>
                    <Text style={styles.detailLabel}>Potential Savings:</Text>
                    <Text style={styles.savingsAmount}>
                      {formatCurrency(selectedRecommendation.potential_savings)}
                    </Text>
                  </View>
                )}
                
                {selectedRecommendation.implementation_steps && (
                  <View style={styles.stepsContainer}>
                    <Text style={styles.stepsTitle}>Implementation Steps:</Text>
                    {selectedRecommendation.implementation_steps.map((step, index) => (
                      <View key={index} style={styles.stepItem}>
                        <Text style={styles.stepNumber}>{index + 1}</Text>
                        <Text style={styles.stepText}>{step}</Text>
                      </View>
                    ))}
                  </View>
                )}
                
                {selectedRecommendation.implemented && (
                  <View style={styles.implementedContainer}>
                    <Icon name="check-circle" size={24} color="#4CAF50" />
                    <Text style={styles.implementedText}>Implemented</Text>
                  </View>
                )}
              </Dialog.Content>
              <Dialog.Actions>
                <Button onPress={() => handleDismiss(selectedRecommendation)}>
                  Dismiss
                </Button>
                <Button 
                  mode={selectedRecommendation.implemented ? "outlined" : "contained"}
                  onPress={() => handleImplement(selectedRecommendation)}
                  disabled={selectedRecommendation.implemented}
                >
                  {selectedRecommendation.implemented ? 'Implemented' : 'Implement'}
                </Button>
              </Dialog.Actions>
            </>
          )}
        </Dialog>
      </Portal>
    </View>
  )}