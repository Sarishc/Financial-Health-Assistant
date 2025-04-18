// Add to main.js
function setupBudgetManagement() {
    const budgets = {};
    
    // Load budgets from local storage or API
    function loadBudgets() {
        fetch(`${API_BASE_URL}/budgets`)
            .then(response => response.json())
            .then(data => {
                data.forEach(budget => {
                    budgets[budget.category] = budget.amount;
                });
                displayBudgets();
            });
    }
    
    // Display budgets and progress
    function displayBudgets() {
        const budgetContainer = document.getElementById('budget-container');
        if (!budgetContainer) return;
        
        let html = '';
        
        Object.entries(budgets).forEach(([category, amount]) => {
            // Get actual spending
            const spent = getSpendingForCategory(category);
            const percentage = Math.min(100, Math.round((spent / amount) * 100));
            
            html += `
            <div class="budget-item">
                <div class="d-flex justify-content-between">
                    <span>${category}</span>
                    <span>$${spent.toFixed(2)} / $${amount.toFixed(2)}</span>
                </div>
                <div class="progress mt-1">
                    <div class="progress-bar ${percentage > 90 ? 'bg-danger' : ''}" 
                         role="progressbar" 
                         style="width: ${percentage}%" 
                         aria-valuenow="${percentage}" 
                         aria-valuemin="0" 
                         aria-valuemax="100"></div>
                </div>
            </div>`;
        });
        
        budgetContainer.innerHTML = html;
    }
}