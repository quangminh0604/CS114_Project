/**
 * JavaScript file for the Compare Solution page functionality
 */

document.addEventListener('DOMContentLoaded', function() {
  // Initialize the model comparison form
  initCompareForm();
});

/**
 * Initialize the model comparison form
 */
function initCompareForm() {
  const form = document.getElementById('compare-form');
  const fileInput = document.getElementById('compare-csv-file');
  const fileNameDisplay = document.getElementById('compare-file-name');
  const chartContainer = document.getElementById('comparison-chart-container');
  let comparisonChart = null;
  
  if (!form || !fileInput) return;
  
  // Handle file selection
  fileInput.addEventListener('change', function() {
    if (fileInput.files.length > 0) {
      const fileName = fileInput.files[0].name;
      if (fileNameDisplay) {
        fileNameDisplay.textContent = fileName;
      }
      
      // Check file extension
      if (!fileName.toLowerCase().endsWith('.csv')) {
        showNotification('Please select a CSV file.', 'warning');
        fileInput.value = '';
        if (fileNameDisplay) {
          fileNameDisplay.textContent = 'No file selected';
        }
      }
    } else {
      if (fileNameDisplay) {
        fileNameDisplay.textContent = 'No file selected';
      }
    }
  });
  
  // Handle form submission
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    if (!fileInput.files.length) {
      showNotification('Please select a CSV file to upload.', 'warning');
      return;
    }
    
    // Show loading state
    showLoading('compare-form', true);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Send data to server
    fetch('/api/compare-models', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.error || 'Failed to compare models. Please try again.');
        });
      }
      return response.json();
    })
    .then(data => {
      // Hide loading state
      showLoading('compare-form', false);
      
      // Display the comparison results
      displayComparisonResults(data, chartContainer, comparisonChart);
      
      // Update the chart reference
      comparisonChart = createComparisonCharts(data, chartContainer);
    })
    .catch(error => {
      handleFormError(error, 'compare-form');
    });
  });
}

/**
 * Display model comparison results
 * @param {Object} results - The comparison results from the API
 * @param {HTMLElement} container - The container element for the chart
 * @param {Object} existingChart - Reference to existing chart (if any)
 */
function displayComparisonResults(results, container, existingChart) {
  if (!container) return;
  
  // Clear existing chart if any
  if (existingChart) {
    existingChart.destroy();
  }
  
  // Clear container
  container.innerHTML = '';
  
  // Create heading
  const heading = document.createElement('h4');
  heading.textContent = 'Model Performance Comparison';
  heading.className = 'mb-4 text-center';
  container.appendChild(heading);
  
  // Create row for charts
  const row = document.createElement('div');
  row.className = 'row';
  container.appendChild(row);
  
  // Create column for accuracy chart
  const accuracyCol = document.createElement('div');
  accuracyCol.className = 'col-12 mb-4';
  row.appendChild(accuracyCol);
  
  const accuracyCard = document.createElement('div');
  accuracyCard.className = 'card';
  accuracyCol.appendChild(accuracyCard);
  
  const accuracyCardBody = document.createElement('div');
  accuracyCardBody.className = 'card-body';
  accuracyCard.appendChild(accuracyCardBody);
  
  const accuracyChartTitle = document.createElement('h5');
  accuracyChartTitle.className = 'card-title text-center mb-3';
  accuracyChartTitle.textContent = 'Accuracy Comparison';
  accuracyCardBody.appendChild(accuracyChartTitle);
  
  const accuracyChartCanvas = document.createElement('canvas');
  accuracyChartCanvas.id = 'accuracy-chart';
  accuracyCardBody.appendChild(accuracyChartCanvas);
  
  // Create metrics table
  const tableContainer = document.createElement('div');
  tableContainer.className = 'col-12 mt-4';
  row.appendChild(tableContainer);
  
  const tableCard = document.createElement('div');
  tableCard.className = 'card';
  tableContainer.appendChild(tableCard);
  
  const tableCardBody = document.createElement('div');
  tableCardBody.className = 'card-body';
  tableCard.appendChild(tableCardBody);
  
  const tableTitle = document.createElement('h5');
  tableTitle.className = 'card-title text-center mb-3';
  tableTitle.textContent = 'Detailed Model Metrics';
  tableCardBody.appendChild(tableTitle);
  
  const tableResponsive = document.createElement('div');
  tableResponsive.className = 'table-responsive';
  tableCardBody.appendChild(tableResponsive);
  
  const table = document.createElement('table');
  table.className = 'table table-bordered table-hover';
  tableResponsive.appendChild(table);
  
  // Create table header
  const tableHeader = document.createElement('thead');
  table.appendChild(tableHeader);
  
  const headerRow = document.createElement('tr');
  tableHeader.appendChild(headerRow);
  
  const modelHeaderCell = document.createElement('th');
  modelHeaderCell.textContent = 'Model';
  headerRow.appendChild(modelHeaderCell);
  
  const accuracyHeaderCell = document.createElement('th');
  accuracyHeaderCell.textContent = 'Accuracy';
  headerRow.appendChild(accuracyHeaderCell);
  
  // Create table body
  const tableBody = document.createElement('tbody');
  table.appendChild(tableBody);
  
  // Populate table with results
  const modelNames = {
    'logistic_regression_sklearn': 'Logistic Regression (sklearn)',
    'logistic_regression_scratch': 'Logistic Regression (from scratch)',
    'svm_sklearn': 'SVM (sklearn)',
    'svm_scratch': 'SVM (from scratch)',
    'decision_tree_sklearn': 'Decision Tree (sklearn)',
    'decision_tree_scratch': 'Decision Tree (from scratch)',
    'random_forest_sklearn': 'Random Forest (sklearn)',
    'random_forest_scratch': 'Random Forest (from scratch)'
  };
  
  for (const modelKey in results) {
    const modelData = results[modelKey];
    
    // Skip if there was an error with this model
    if (modelData.error) {
      continue;
    }
    
    const row = document.createElement('tr');
    tableBody.appendChild(row);
    
    const modelCell = document.createElement('td');
    modelCell.textContent = modelNames[modelKey] || modelKey;
    row.appendChild(modelCell);
    
    const accuracyCell = document.createElement('td');
    accuracyCell.textContent = modelData.accuracy !== undefined ? (modelData.accuracy * 100).toFixed(2) + '%' : 'N/A';
    row.appendChild(accuracyCell);
  }
  
  // Create the charts
  createComparisonCharts(results);
  
  // Show the chart container
  container.style.display = 'block';
}

/**
 * Create comparison charts using Chart.js
 * @param {Object} results - The comparison results from the API
 * @returns {Object} Object containing reference to created chart
 */
function createComparisonCharts(results) {
  // Define colors for each model
  const colors = {
    'logistic_regression_sklearn': '#4285F4',    // Blue
    'logistic_regression_scratch': '#0D5BCE',    // Dark Blue
    'svm_sklearn': '#34A853',                    // Green
    'svm_scratch': '#1D7D36',                    // Dark Green
    'decision_tree_sklearn': '#FBBC05',          // Yellow
    'decision_tree_scratch': '#E0A800',          // Dark Yellow
    'random_forest_sklearn': '#EA4335',          // Red
    'random_forest_scratch': '#B31412'           // Dark Red
  };
  
  // Define model display names
  const modelNames = {
    'logistic_regression_sklearn': 'Logistic Regression (sklearn)',
    'logistic_regression_scratch': 'Logistic Regression (from scratch)',
    'svm_sklearn': 'SVM (sklearn)',
    'svm_scratch': 'SVM (from scratch)',
    'decision_tree_sklearn': 'Decision Tree (sklearn)',
    'decision_tree_scratch': 'Decision Tree (from scratch)',
    'random_forest_sklearn': 'Random Forest (sklearn)',
    'random_forest_scratch': 'Random Forest (from scratch)'
  };
  
  // Extract model names and metrics
  const labels = [];
  const accuracyData = [];
  
  for (const modelKey in results) {
    const modelData = results[modelKey];
    
    // Skip if there was an error with this model
    if (modelData.error) {
      continue;
    }
    
    labels.push(modelNames[modelKey] || modelKey);
    accuracyData.push(modelData.accuracy !== undefined ? modelData.accuracy * 100 : 0);
  }
  
  // Generate background colors
  const backgroundColors = labels.map((label, index) => {
    const modelKey = Object.keys(modelNames).find(key => modelNames[key] === label);
    return colors[modelKey] || `hsl(${index * 60}, 70%, 60%)`;
  });
  
  // Create accuracy chart
  const accuracyChart = new Chart(
    document.getElementById('accuracy-chart'),
    {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracyData,
            backgroundColor: backgroundColors,
            borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
            borderWidth: 1
          }
        ]
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
                return `Accuracy: ${context.raw.toFixed(2)}%`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'Accuracy (%)'
            }
          }
        }
      }
    }
  );
  
  return {
    accuracyChart
  };
}
