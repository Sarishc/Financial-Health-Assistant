from app.data.schema import Transaction, CategoryList, MonthlySummary
from datetime import date

# Create a sample transaction
sample_transaction = Transaction(
    transaction_date=date.today(),
    description="Grocery Store Purchase",
    amount=-85.42,
    category="food"
)

# Print the transaction
print("Sample Transaction:")
# Use model_dump_json instead of json for Pydantic v2
print(sample_transaction.model_dump_json(indent=2))

# Test category list
categories = CategoryList(categories=[
    {"name": "food", "description": "Food and dining", "is_expense": True},
    {"name": "income", "description": "Income sources", "is_expense": False}
])

print("\nCategory List:")
print(categories.model_dump_json(indent=2))

print("\nSchema validation works correctly!")