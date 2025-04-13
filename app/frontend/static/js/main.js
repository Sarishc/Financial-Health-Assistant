// API Base URL
const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

// Chart color palette
const CHART_COLORS = {
    'food': '#FF6384',
    'transport': '#36A2EB',
    'shopping': '#FFCE56',
    'utilities': '#4BC0C0',
    'entertainment': '#9966FF',
    'health': '#FF9F40',
    'housing': '#8BC34A',
    'income': '#4CAF50',
    'transfer': '#9C27B0',
    'other': '#607D8B'
};

// Initialize charts
let categoryChart = null;
let trendChart = null;
let forecastChart = null;
let forecastCategoryChart = null;

// Current active section
let currentSection = 'dashboard';

// Handle navigation
document.addEventListener('DOMContentLoaded', function() {
    // Set up navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show target section
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                    currentSection = targetId;
                    
                    // Load section data if needed
                    if (targetId === 'dashboard') {
                        loadDashboardData();
                    } else if (targetId === 'transactions') {
                        loadTransactions();
                    } else if (targetId === 'forecast') {
                        loadForecastData();
                    } else if (targetId === 'recommendations') {
                        loadRecommendations();
                    }
                }
            });
        });
    });
    
    // Set up file upload
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadTransactions();
        });
    }
    
    // Set up transaction search
    const searchInput = document.getElementById('transaction-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterTransactions(this.value);
        });
    }
    
    // Set up category filter for forecast
    const categorySelect = document.getElementById('forecast-category');
    if (categorySelect) {
        categorySelect.addEventListener('change', function() {
            updateForecastChart(this.value);
        });
    }
    
    // Load initial dashboard data
    loadDashboardData();
});

// Load dashboard data
function loadDashboardData() {
    // Fetch transaction summary
    fetch(`${API_BASE_URL}/transactions/summary`)
        .then(response => response.json())
        .then(data => {
            updateSummaryStats(data);
            updateCategoryChart(data.category_breakdown);
            
            // Create spending trend chart (simulated data for now)
            createTrendChart();
        })
        .catch(error => {
            console.error('Error fetching summary:', error);
            document.getElementById('summary-stats').innerHTML = 
                '<div class="alert alert-danger">Error loading summary data</div>';
        });
}

// Update summary statistics
function updateSummaryStats(data) {
    const summaryElement = document.getElementById('summary-stats');
    
    if (!data || !data.financials) {
        summaryElement.innerHTML = '<div class="alert alert-warning">No transaction data available</div>';
        return;
    }
    
    const dateRange = data.date_range.start && data.date_range.end ? 
        `<p><strong>Date Range:</strong> ${data.date_range.start.substring(0, 10)} to ${data.date_range.end.substring(0, 10)}</p>` :
        '';
    
    summaryElement.innerHTML = `
        <p><strong>Total Transactions:</strong> ${data.total_transactions}</p>
        ${dateRange}
        <p><strong>Total Spending:</strong> <span class="amount-negative">$${data.financials.total_spending.toFixed(2)}</span></p>
        <p><strong>Total Income:</strong> <span class="amount-positive">$${data.financials.total_income.toFixed(2)}</span></p>
        <p><strong>Net Cash Flow:</strong> <span class="${data.financials.net_cashflow >= 0 ? 'amount-positive' : 'amount-negative'}">
            $${Math.abs(data.financials.net_cashflow).toFixed(2)} ${data.financials.net_cashflow >= 0 ? '' : '(negative)'}
        </span></p>
    `;
}

// Create category chart
function updateCategoryChart(categoryData) {
    if (!categoryData) return;
    
    const ctx = document.getElementById('category-chart').getContext('2d');
    
    // Prepare chart data
    const categories = Object.keys(categoryData);
    const values = Object.values(categoryData);
    
    // Get colors for categories
    const colors = categories.map(cat => CHART_COLORS[cat.toLowerCase()] || '#607D8B');
    
    // Destroy existing chart if it exists
    if (categoryChart) {
        categoryChart.destroy();
    }
    
    // Create new chart
    categoryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `$${value.toFixed(2)} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Create trend chart with simulated data
function createTrendChart() {
    const ctx = document.getElementById('trend-chart').getContext('2d');
    
    // Generate labels for the last 6 months
    const labels = [];
    const currentDate = new Date();
    for (let i = 5; i >= 0; i--) {
        const date = new Date(currentDate);
        date.setMonth(currentDate.getMonth() - i);
        const monthName = date.toLocaleString('default', { month: 'short' });
        const year = date.getFullYear();
        labels.push(`${monthName} ${year}`);
    }
    
    // Simulated data - replace with actual API data when available
    const data = {
        labels: labels,
        datasets: [{
            label: 'Monthly Spending',
            data: [3245.67, 2986.42, 3456.78, 3021.45, 2897.34, 3112.56],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 2,
            tension: 0.3
        }]
    };
    
    // Destroy existing chart if it exists
    if (trendChart) {
        trendChart.destroy();
    }
    
    // Create new chart
    trendChart = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Load transactions
function loadTransactions() {
    // In a real app, you would fetch transaction data from your API
    // For this demo, we'll simulate some transactions
    const sampleTransactions = [
        { date: '2023-01-15', description: 'Grocery Store', category: 'food', amount: -85.42 },
        { date: '2023-01-16', description: 'Monthly Rent', category: 'housing', amount: -1200.00 },
        { date: '2023-01-18', description: 'Gas Station', category: 'transport', amount: -45.23 },
        { date: '2023-01-20', description: 'Salary Deposit', category: 'income', amount: 2500.00 },
        { date: '2023-01-21', description: 'Netflix Subscription', category: 'entertainment', amount: -14.99 },
        { date: '2023-01-23', description: 'Electric Bill', category: 'utilities', amount: -95.67 },
        { date: '2023-01-24', description: 'Amazon Purchase', category: 'shopping', amount: -65.89 },
        { date: '2023-01-27', description: 'Doctor Visit', category: 'health', amount: -125.00 }
    ];
    
    displayTransactions(sampleTransactions);
}

// Display transactions in table
function displayTransactions(transactions) {
    const tableBody = document.getElementById('transactions-table');
    
    if (!transactions || transactions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No transactions found</td></tr>';
        return;
    }
    
    // Sort transactions by date (newest first)
    transactions.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    // Build table rows
    let html = '';
    transactions.forEach(transaction => {
        const amountClass = transaction.amount < 0 ? 'amount-negative' : 'amount-positive';
        const amountSign = transaction.amount < 0 ? '-' : '+';
        const amountValue = Math.abs(transaction.amount).toFixed(2);
        
        html += `
            <tr data-search="${transaction.description.toLowerCase()}">
                <td>${transaction.date}</td>
                <td>${transaction.description}</td>
                <td><span class="category-tag category-${transaction.category}">${transaction.category}</span></td>
                <td class="${amountClass}">${amountSign}$${amountValue}</td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

// Filter transactions by search term
function filterTransactions(searchTerm) {
    const rows = document.querySelectorAll('#transactions-table tr');
    searchTerm = searchTerm.toLowerCase();
    
    rows.forEach(row => {
        const searchable = row.getAttribute('data-search');
        if (searchable && searchable.includes(searchTerm)) {
            row.style.display = '';
        } else if (searchable) {
            row.style.display = 'none';
        }
    });
}

// Upload transactions
function uploadTransactions() {
    const fileInput = document.getElementById('transaction-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file to upload');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    const uploadButton = document.querySelector('#upload-form button');
    const originalText = uploadButton.innerHTML;
    uploadButton.innerHTML = 'Uploading...';
    uploadButton.disabled = true;
    
    // Upload file
    fetch(`${API_BASE_URL}/transactions/upload-csv`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        return response.json();
    })
    .then(data => {
        alert(`Successfully processed ${data.message}`);
        
        // Display sample of processed transactions
        if (data.sample && data.sample.length > 0) {
            displayTransactions(data.sample);
        }
        
        // Reload dashboard data
        loadDashboardData();
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        alert('Failed to upload transactions. Please try again.');
    })
    .finally(() => {
        // Reset button
        uploadButton.innerHTML = originalText;
        uploadButton.disabled = false;
        fileInput.value = '';
    });
}

// Load forecast data
function loadForecastData() {
    fetch(`${API_BASE_URL}/transactions/forecast`)
        .then(response => response.json())
        .then(data => {
            if (data.forecasts && data.forecasts.length > 0) {
                // Populate category dropdown
                populateForecastCategories(data.forecasts);
                
                // Display forecast data
                updateForecastChart('all');
                updateForecastSummary(data.forecasts);
                updateForecastCategoryChart(data.forecasts);
            } else {
                document.getElementById('forecast-summary').innerHTML = 
                    '<div class="alert alert-warning">No forecast data available</div>';
            }
        })
        .catch(error => {
            console.error('Error fetching forecast:', error);
            document.getElementById('forecast-summary').innerHTML = 
                '<div class="alert alert-danger">Error loading forecast data</div>';
        });
}

// Populate forecast category dropdown
function populateForecastCategories(forecasts) {
    const categorySelect = document.getElementById('forecast-category');
    
    // Clear existing options (keep the "All Categories" option)
    while (categorySelect.options.length > 1) {
        categorySelect.remove(1);
    }
    
    // Add options for each category
    forecasts.forEach(forecast => {
        const option = document.createElement('option');
        option.value = forecast.category;
        option.textContent = forecast.category.charAt(0).toUpperCase() + forecast.category.slice(1);
        categorySelect.appendChild(option);
    });
}

// Update forecast chart based on selected category
function updateForecastChart(selectedCategory) {
    fetch(`${API_BASE_URL}/transactions/forecast`)
        .then(response => response.json())
        .then(data => {
            if (!data.forecasts || data.forecasts.length === 0) {
                return;
            }
            
            const ctx = document.getElementById('forecast-chart').getContext('2d');
            
            let chartData;
            if (selectedCategory === 'all') {
                // Combine all forecasts
                const allDates = new Set();
                data.forecasts.forEach(f => f.forecast.forEach(p => allDates.add(p.date)));
                
                const sortedDates = Array.from(allDates).sort();
                const combinedAmounts = {};
                
                sortedDates.forEach(date => {
                    combinedAmounts[date] = 0;
                    
                    data.forecasts.forEach(forecast => {
                        const point = forecast.forecast.find(p => p.date === date);
                        if (point) {
                            combinedAmounts[date] += point.amount;
                        }
                    });
                });
                
                chartData = {
                    labels: sortedDates,
                    datasets: [{
                        label: 'All Categories',
                        data: sortedDates.map(date => combinedAmounts[date]),
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.3,
                        fill: true
                    }]
                };
            } else {
                // Find selected category
                const selectedForecast = data.forecasts.find(f => f.category === selectedCategory);
                
                if (!selectedForecast) {
                    return;
                }
                
                const color = CHART_COLORS[selectedCategory.toLowerCase()] || '#607D8B';
                
                chartData = {
                    labels: selectedForecast.forecast.map(p => p.date),
                    datasets: [{
                        label: selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1),
                        data: selectedForecast.forecast.map(p => p.amount),
                        borderColor: color,
                        backgroundColor: color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                        tension: 0.3,
                        fill: true
                    }]
                };
                
                // Add confidence interval if available
                if (selectedForecast.forecast[0].lower_bound && selectedForecast.forecast[0].upper_bound) {
                    chartData.datasets.push({
                        label: 'Upper Bound',
                        data: selectedForecast.forecast.map(p => p.upper_bound),
                        borderColor: color.replace(')', ', 0.5)').replace('rgb', 'rgba'),
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        tension: 0.3,
                        fill: false
                    });
                    
                    chartData.datasets.push({
                        label: 'Lower Bound',
                        data: selectedForecast.forecast.map(p => p.lower_bound),
                        borderColor: color.replace(')', ', 0.5)').replace('rgb', 'rgba'),
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        tension: 0.3,
                        fill: false
                    });
                }
            }
            
            // Destroy existing chart if it exists
            if (forecastChart) {
                forecastChart.destroy();
            }
            
            // Create new chart
            forecastChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return '$' + value;
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return '$' + context.parsed.y.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error updating forecast chart:', error);
        });
}

// Update forecast summary
function updateForecastSummary(forecasts) {
    const summaryElement = document.getElementById('forecast-summary');
    
    if (!forecasts || forecasts.length === 0) {
        summaryElement.innerHTML = '<div class="alert alert-warning">No forecast data available</div>';
        return;
    }
    
    // Calculate total forecast
    const totalForecast = forecasts.reduce((sum, f) => sum + f.total_forecasted, 0);
    
    // Find highest day
    let highestDay = { date: '', amount: 0 };
    
    forecasts.forEach(forecast => {
        forecast.forecast.forEach(point => {
            if (point.amount > highestDay.amount) {
                highestDay = { date: point.date, amount: point.amount };
            }
        });
    });
    
    summaryElement.innerHTML = `
        <p><strong>Total Forecasted Spending:</strong> <span class="amount-negative">$${totalForecast.toFixed(2)}</span></p>
        <p><strong>Forecast Period:</strong> 30 days</p>
        <p><strong>Highest Spending Day:</strong> ${highestDay.date} <span class="amount-negative">($${highestDay.amount.toFixed(2)})</span></p>
        <p><strong>Daily Average:</strong> <span class="amount-negative">$${(totalForecast / 30).toFixed(2)}</span></p>
        <p><strong>Categories Forecast:</strong> ${forecasts.length}</p>
    `;
}

// Update forecast category chart
function updateForecastCategoryChart(forecasts) {
    if (!forecasts || forecasts.length === 0) return;
    
    const ctx = document.getElementById('forecast-category-chart').getContext('2d');
    
    // Prepare chart data
    const categories = forecasts.map(f => f.category);
    const values = forecasts.map(f => f.total_forecasted);
    
    // Get colors for categories
    const colors = categories.map(cat => CHART_COLORS[cat.toLowerCase()] || '#607D8B');
    
    // Destroy existing chart if it exists
    if (forecastCategoryChart) {
        forecastCategoryChart.destroy();
    }
    
    // Create new chart
    forecastCategoryChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.raw.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value;
                        }
                    }
                }
            }
        }
    });
}

// Load recommendations
function loadRecommendations() {
    fetch(`${API_BASE_URL}/transactions/recommendations`)
        .then(response => response.json())
        .then(data => {
            displayRecommendations(data.recommendations);
        })
        .catch(error => {
            console.error('Error fetching recommendations:', error);
            document.getElementById('recommendations-list').innerHTML = 
                '<div class="alert alert-danger">Error loading recommendations</div>';
        });
}

// Display recommendations
function displayRecommendations(recommendations) {
    const recommendationsElement = document.getElementById('recommendations-list');
    
    if (!recommendations || recommendations.length === 0) {
        recommendationsElement.innerHTML = '<div class="alert alert-warning">No recommendations available</div>';
        return;
    }
    
    let html = '';
    
    recommendations.forEach(recommendation => {
        // Determine priority class
        let priorityClass = 'low-priority';
        let priorityLabel = 'Low Priority';
        
        if (recommendation.priority >= 80) {
            priorityClass = 'high-priority';
            priorityLabel = 'High Priority';
        } else if (recommendation.priority >= 50) {
            priorityClass = 'medium-priority';
            priorityLabel = 'Medium Priority';
        }
        
        // Get appropriate color for priority indicator
        let indicatorColor = '#28a745'; // green for low
        if (recommendation.priority >= 80) {
            indicatorColor = '#dc3545'; // red for high
        } else if (recommendation.priority >= 50) {
            indicatorColor = '#ffc107'; // yellow for medium
        }
        
        html += `
            <div class="recommendation-card ${priorityClass}">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>
                        <span class="priority-indicator" style="background-color: ${indicatorColor}"></span>
                        <small>${priorityLabel}</small>
                    </span>
                    <span class="badge bg-primary">${recommendation.type}</span>
                </div>
                <p class="mb-0">${recommendation.message}</p>
            </div>
        `;
    });
    
    recommendationsElement.innerHTML = html;
}