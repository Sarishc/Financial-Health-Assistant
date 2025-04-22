// src/screens/transactions/TransactionDetailScreen.js
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { 
  Card, 
  Title, 
  Paragraph, 
  Button, 
  Divider,
  IconButton,
  Menu,
  List,
  Chip,
  Text,
  Dialog,
  Portal,
  TextInput,
  RadioButton
} from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { useDispatch, useSelector } from 'react-redux';
import { formatCurrency } from '../../utils/formatters';
import { getCategoryColor } from '../../utils/categories';
import { 
  updateTransaction, 
  deleteTransaction,
  fetchSimilarTransactions 
} from '../../store/actions/transactionActions';

const TransactionDetailScreen = ({ route, navigation }) => {
  const { transaction } = route.params;
  const dispatch = useDispatch();
  
  const { categories, similarTransactions } = useSelector(state => state.transactions);
  const [menuVisible, setMenuVisible] = useState(false);
  const [editDialogVisible, setEditDialogVisible] = useState(false);
  const [categoryDialogVisible, setCategoryDialogVisible] = useState(false);
  const [editedTransaction, setEditedTransaction] = useState({ ...transaction });
  const [selectedCategory, setSelectedCategory] = useState(transaction.category || 'Uncategorized');
  
  // Fetch similar transactions on component mount
  useEffect(() => {
    dispatch(fetchSimilarTransactions(transaction.description));
  }, [dispatch, transaction.description]);
  
  const handleEdit = () => {
    setEditedTransaction({ ...transaction });
    setMenuVisible(false);
    setEditDialogVisible(true);
  };
  
  const handleChangeCategory = () => {
    setMenuVisible(false);
    setCategoryDialogVisible(true);
  };
  
  const handleDelete = () => {
    setMenuVisible(false);
    
    Alert.alert(
      'Delete Transaction',
      'Are you sure you want to delete this transaction? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: () => {
            dispatch(deleteTransaction(transaction.id)).then(() => {
              navigation.goBack();
            });
          }
        }
      ]
    );
  };
  
  const handleSaveEdit = () => {
    dispatch(updateTransaction({
      ...transaction,
      ...editedTransaction
    })).then(() => {
      setEditDialogVisible(false);
      // Refresh transaction data
      navigation.setParams({ transaction: { ...transaction, ...editedTransaction } });
    });
  };
  
  const handleSaveCategory = () => {
    dispatch(updateTransaction({
      ...transaction,
      category: selectedCategory
    })).then(() => {
      setCategoryDialogVisible(false);
      // Refresh transaction data
      navigation.setParams({ transaction: { ...transaction, category: selectedCategory } });
    });
  };
  
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };
  
  return (
    <ScrollView style={styles.container}>
      {/* Main Transaction Card */}
      <Card style={styles.mainCard}>
        <Card.Content>
          <View style={styles.headerRow}>
            <Title>Transaction Details</Title>
            <IconButton
              icon="dots-vertical"
              onPress={() => setMenuVisible(true)}
              size={24}
            />
            <Menu
              visible={menuVisible}
              onDismiss={() => setMenuVisible(false)}
              anchor={{ x: 0, y: 0 }}
              style={styles.menu}
            >
              <Menu.Item icon="pencil" onPress={handleEdit} title="Edit Transaction" />
              <Menu.Item icon="tag" onPress={handleChangeCategory} title="Change Category" />
              <Divider />
              <Menu.Item icon="delete" onPress={handleDelete} title="Delete Transaction" />
            </Menu>
          </View>
          
          <View style={styles.amountContainer}>
            <Text style={[
              styles.amountText,
              transaction.amount > 0 ? styles.incomeText : styles.expenseText
            ]}>
              {formatCurrency(Math.abs(transaction.amount))}
            </Text>
            <Chip 
              style={[styles.categoryChip, { backgroundColor: getCategoryColor(transaction.category) }]}
              textStyle={styles.categoryChipText}
            >
              {transaction.category || 'Uncategorized'}
            </Chip>
          </View>
          
          <Divider style={styles.divider} />
          
          <Title style={styles.descriptionText}>{transaction.description}</Title>
          
          <List.Item
            title="Date"
            description={formatDate(transaction.transaction_date)}
            left={props => <List.Icon {...props} icon="calendar" />}
            style={styles.listItem}
          />
          
          {transaction.account && (
            <List.Item
              title="Account"
              description={transaction.account}
              left={props => <List.Icon {...props} icon="bank" />}
              style={styles.listItem}
            />
          )}
          
          {transaction.notes && (
            <List.Item
              title="Notes"
              description={transaction.notes}
              left={props => <List.Icon {...props} icon="text" />}
              style={styles.listItem}
            />
          )}
        </Card.Content>
      </Card>
      
      {/* Similar Transactions Card */}
      {similarTransactions && similarTransactions.length > 0 && (
        <Card style={styles.card}>
          <Card.Title title="Similar Transactions" />
          <Card.Content>
            {similarTransactions.slice(0, 3).map((similar, index) => (
              <React.Fragment key={similar.id || index}>
                <List.Item
                  title={similar.description}
                  description={`${formatDate(similar.transaction_date)} • ${formatCurrency(Math.abs(similar.amount))}`}
                  left={props => <List.Icon {...props} icon="receipt" />}
                  right={props => (
                    <Chip 
                      style={{ backgroundColor: getCategoryColor(similar.category) }}
                      textStyle={styles.categoryChipText}
                      size={20}
                    >
                      {similar.category || 'Uncategorized'}
                    </Chip>
                  )}
                />
                {index < similarTransactions.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </Card.Content>
        </Card>
      )}
      
      {/* Actions Card */}
      <Card style={styles.card}>
        <Card.Title title="Actions" />
        <Card.Content>
          <Button 
            mode="contained" 
            icon="tag" 
            onPress={handleChangeCategory}
            style={styles.actionButton}
          >
            Change Category
          </Button>
          
          <Button 
            mode="outlined" 
            icon="pencil" 
            onPress={handleEdit}
            style={styles.actionButton}
          >
            Edit Details
          </Button>
          
          <Button 
            mode="outlined" 
            icon="delete" 
            onPress={handleDelete}
            style={[styles.actionButton, styles.deleteButton]}
            color="#F44336"
          >
            Delete Transaction
          </Button>
        </Card.Content>
      </Card>
      
      {/* Edit Dialog */}
      <Portal>
        <Dialog visible={editDialogVisible} onDismiss={() => setEditDialogVisible(false)}>
          <Dialog.Title>Edit Transaction</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Description"
              value={editedTransaction.description}
              onChangeText={text => setEditedTransaction({...editedTransaction, description: text})}
              style={styles.input}
            />
            
            <TextInput
              label="Amount"
              value={String(Math.abs(editedTransaction.amount))}
              onChangeText={text => {
                const numValue = parseFloat(text) || 0;
                const sign = editedTransaction.amount < 0 ? -1 : 1;
                setEditedTransaction({...editedTransaction, amount: numValue * sign});
              }}
              keyboardType="numeric"
              style={styles.input}
            />
            
            <View style={styles.radioGroup}>
              <Text>Transaction Type</Text>
              <RadioButton.Group
                onValueChange={value => {
                  const absAmount = Math.abs(editedTransaction.amount);
                  setEditedTransaction({
                    ...editedTransaction, 
                    amount: value === 'expense' ? -absAmount : absAmount
                  });
                }}
                value={editedTransaction.amount < 0 ? 'expense' : 'income'}
              >
                <View style={styles.radioButton}>
                  <RadioButton value="expense" />
                  <Text>Expense</Text>
                </View>
                <View style={styles.radioButton}>
                  <RadioButton value="income" />
                  <Text>Income</Text>
                </View>
              </RadioButton.Group>
            </View>
            
            <TextInput
              label="Notes"
              value={editedTransaction.notes || ''}
              onChangeText={text => setEditedTransaction({...editedTransaction, notes: text})}
              multiline
              numberOfLines={3}
              style={styles.input}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setEditDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleSaveEdit}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
      
      {/* Category Dialog */}
      <Portal>
        <Dialog visible={categoryDialogVisible} onDismiss={() => setCategoryDialogVisible(false)}>
          <Dialog.Title>Change Category</Dialog.Title>
          <Dialog.Content>
            <RadioButton.Group
              onValueChange={value => setSelectedCategory(value)}
              value={selectedCategory}
            >
              <RadioButton.Item label="Uncategorized" value="Uncategorized" />
              <Divider />
              {categories.map(category => (
                <RadioButton.Item 
                  key={category} 
                  label={category} 
                  value={category}
                  labelStyle={{ color: getCategoryColor(category) }}
                />
              ))}
            </RadioButton.Group>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setCategoryDialogVisible(false)}>Cancel</Button>
            <Button onPress={handleSaveCategory}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  mainCard: {
    margin: 16,
  },
  card: {
    margin: 16,
    marginTop: 0,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  menu: {
    marginTop: 50,
  },
  amountContainer: {
    alignItems: 'center',
    marginVertical: 16,
  },
  amountText: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  incomeText: {
    color: '#4CAF50',
  },
  expenseText: {
    color: '#F44336',
  },
  divider: {
    marginVertical: 16,
  },
  descriptionText: {
    fontSize: 20,
    marginBottom: 16,
  },
  categoryChip: {
    paddingHorizontal: 8,
  },
  categoryChipText: {
    color: 'white',
  },
  listItem: {
    paddingLeft: 0,
  },
  actionButton: {
    marginBottom: 8,
  },
  deleteButton: {
    borderColor: '#F44336',
  },
  input: {
    marginBottom: 16,
    backgroundColor: 'transparent',
  },
  radioGroup: {
    marginBottom: 16,
  },
  radioButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
});

export default TransactionDetailScreen;