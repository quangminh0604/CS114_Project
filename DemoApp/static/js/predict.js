/**
 * JavaScript file for the Prediction page functionality
 */

document.addEventListener('DOMContentLoaded', function() {
  // Initialize the prediction form
  initPredictForm();

  // Initialize the batch prediction form
  initBatchUploadForm();

  // Set up tab switching
  initTabSwitching();

  // Initialize clinic finder
  initClinicFinder();
});

/**
 * Initialize the prediction form with event listeners
 */
function initPredictForm() {
  const form = document.getElementById('prediction-form');
  if (!form) return;

  const cigsPerDayInput = document.getElementById('cigsPerDay');

  form.addEventListener('submit', function(e) {
    e.preventDefault();

    // Validate form inputs
    const { isValid, errorMessage } = validateForm(form);
    if (!isValid) {
      showNotification(errorMessage, 'warning');
      return;
    }

    // Show loading state
    showLoading('prediction-form', true);

    // Determine currentSmoker from cigsPerDay
    const cigsPerDayValue = parseFloat(cigsPerDayInput.value || 0);
    const currentSmokerValue = cigsPerDayValue > 0;

    // Gather form data
    const formData = {
      algorithm: document.getElementById('algorithm').value,
      male: document.getElementById('male').checked,
      age: parseInt(document.getElementById('age').value),
      currentSmoker: currentSmokerValue,
      cigsPerDay: cigsPerDayValue,
      BPMeds: document.getElementById('BPMeds').checked,
      prevalentStroke: document.getElementById('prevalentStroke').checked,
      prevalentHyp: document.getElementById('prevalentHyp').checked,
      diabetes: document.getElementById('diabetes').checked,
      totChol: parseFloat(document.getElementById('totChol').value),
      sysBP: parseFloat(document.getElementById('sysBP').value),
      diaBP: parseFloat(document.getElementById('diaBP').value),
      BMI: parseFloat(document.getElementById('BMI').value),
      heartRate: parseFloat(document.getElementById('heartRate').value),
      glucose: parseFloat(document.getElementById('glucose').value)
    };

    // Send data to the server
    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData)
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.error || 'Failed to predict risk. Please try again.');
        });
      }
      return response.json();
    })
    .then(data => {
      // Hide loading state
      showLoading('prediction-form', false);

      // Display the result
      displayPredictionResult(data);
    })
    .catch(error => {
      handleFormError(error, 'prediction-form');
    });
  });

  // Handle BMI calculation
  const heightInput = document.getElementById('height');
  const weightInput = document.getElementById('weight');
  const bmiInput = document.getElementById('BMI');

  if (heightInput && weightInput && bmiInput) {
    const calculateBMI = () => {
      const height = parseFloat(heightInput.value);
      const weight = parseFloat(weightInput.value);

      if (height && weight) {
        // Calculate BMI (weight in kg / height in meters squared)
        const heightInMeters = height / 100;
        const bmi = weight / (heightInMeters * heightInMeters);
        bmiInput.value = bmi.toFixed(1);
      }
    };

    heightInput.addEventListener('input', calculateBMI);
    weightInput.addEventListener('input', calculateBMI);
  }
}

/**
 * Initialize batch prediction form for file upload
 */
function initBatchUploadForm() {
  const form = document.getElementById('batch-upload-form');
  const fileInput = document.getElementById('csv-file');
  const fileNameDisplay = document.getElementById('file-name');
  
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
    showLoading('batch-upload-form', true);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('algorithm', document.getElementById('batch-algorithm').value);
    
    // Send data to server
    fetch('/api/batch-predict', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.error || 'Failed to process batch prediction. Please try again.');
        });
      }
      
      // For CSV responses, trigger download
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('text/csv')) {
        return response.blob().then(blob => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = 'prediction_results.csv';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          return { success: true };
        });
      }
      
      return response.json();
    })
    .then(data => {
      // Hide loading state
      showLoading('batch-upload-form', false);
      
      if (data.success || data.message) {
        showNotification('Batch prediction completed successfully. The results have been downloaded.', 'success');
        
        // Reset form
        form.reset();
        if (fileNameDisplay) {
          fileNameDisplay.textContent = 'No file selected';
        }
      }
    })
    .catch(error => {
      handleFormError(error, 'batch-upload-form');
    });
  });
}

/**
 * Display prediction result in the result box
 * @param {Object} result - The prediction result from the API
 */
function displayPredictionResult(result) {
  const resultBox = document.getElementById('result-box');
  if (!resultBox) return;
  
  // Determine risk level and color
  let riskLevel = '';
  let riskColor = '';
  
  if (result.prediction) {
    riskLevel = 'High Risk';
    riskColor = 'danger';
  } else {
    riskLevel = 'Low Risk';
    riskColor = 'success';
  }
  
  // Format result HTML
  const resultHTML = `
    <div class="alert alert-${riskColor} mb-4" role="alert">
      <h4 class="alert-heading">
        <i class="fas fa-${result.prediction ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
        ${riskLevel} of Heart Disease
      </h4>
      <p class="mb-0">Based on your health information, you have a ${riskLevel.toLowerCase()} of developing heart disease in the next 10 years according to the Framingham Heart Study model.</p>
    </div>
    
    <div class="card mb-4">
      <div class="card-header bg-${riskColor} text-white">
        <h5 class="mb-0">Risk Assessment - ${result.algorithm}</h5>
      </div>
      <div class="card-body">
        <p>Your risk of developing coronary heart disease (CHD) within the next 10 years was calculated using the ${result.algorithm} algorithm.</p>
        
        <div class="d-flex justify-content-between align-items-center mb-4">
          <h5 class="mb-0">Prediction Result:</h5>
          <span class="badge bg-${riskColor} fs-5 py-2 px-3">${result.prediction ? 'Elevated Risk' : 'Low Risk'}</span>
        </div>
        
        ${result.prediction ? `
        <div class="alert alert-warning">
          <i class="fas fa-info-circle me-2"></i>
          <strong>Important:</strong> This is a screening result, not a diagnosis. Please consult a healthcare professional for a complete evaluation.
        </div>
        ` : ''}
        
        <div class="mt-4">
          <h6>Next Steps:</h6>
          <ul class="list-group list-group-flush">
            ${result.prediction ? `
            <li class="list-group-item"><i class="fas fa-user-md text-primary me-2"></i> Schedule an appointment with a cardiologist</li>
            <li class="list-group-item"><i class="fas fa-heartbeat text-danger me-2"></i> Monitor your blood pressure regularly</li>
            <li class="list-group-item"><i class="fas fa-utensils text-success me-2"></i> Follow a heart-healthy diet (reduced sodium, low saturated fat)</li>
            <li class="list-group-item"><i class="fas fa-walking text-info me-2"></i> Aim for 150 minutes of moderate exercise weekly</li>
            ` : `
            <li class="list-group-item"><i class="fas fa-check text-success me-2"></i> Continue maintaining healthy lifestyle habits</li>
            <li class="list-group-item"><i class="fas fa-calendar-check text-primary me-2"></i> Schedule regular check-ups with your healthcare provider</li>
            <li class="list-group-item"><i class="fas fa-heartbeat text-info me-2"></i> Monitor your blood pressure periodically</li>
            `}
          </ul>
        </div>
      </div>
    </div>
    
    ${result.prediction ? `
    <div class="card mb-4">
      <div class="card-header bg-info text-white">
        <h5 class="mb-0"><i class="fas fa-map-marker-alt me-2"></i>Find Nearby Heart Specialists</h5>
      </div>
      <div class="card-body">
        <p>Enter your address to find cardiologists and heart clinics near you:</p>
        <div class="input-group mb-3">
          <input type="text" id="address-input" class="form-control" placeholder="Enter your address">
          <button class="btn btn-primary" type="button" id="find-clinics-btn">
            <i class="fas fa-search me-2"></i>Find Clinics
          </button>
        </div>
      </div>
    </div>
    ` : ''}
  `;
  
  // Set the HTML content
  resultBox.innerHTML = resultHTML;
  
  // Show the result box
  resultBox.style.display = 'block';
  
  // Scroll to the result box
  resultBox.scrollIntoView({ behavior: 'smooth' });
  
  // If high risk, initialize the clinic finder button
  if (result.prediction) {
    const findClinicsBtn = document.getElementById('find-clinics-btn');
    const addressInput = document.getElementById('address-input');
    
    if (findClinicsBtn && addressInput) {
      findClinicsBtn.addEventListener('click', function() {
        const address = addressInput.value.trim();
        
        if (address) {
          // Find nearby heart clinics using Gemini API
          findNearbyHeartClinics(address);
        } else {
          showNotification('Please enter your address to find nearby clinics.', 'warning');
        }
      });
    }
  }
}

/**
 * Initialize tab switching functionality
 */
function initTabSwitching() {
  const tabLinks = document.querySelectorAll('.nav-link[data-bs-toggle="tab"]');
  
  tabLinks.forEach(tabLink => {
    tabLink.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Remove active class from all tabs
      tabLinks.forEach(link => {
        link.classList.remove('active');
        const tabContent = document.querySelector(link.getAttribute('href'));
        if (tabContent) {
          tabContent.classList.remove('show', 'active');
        }
      });
      
      // Add active class to current tab
      this.classList.add('active');
      const targetTab = document.querySelector(this.getAttribute('href'));
      if (targetTab) {
        targetTab.classList.add('show', 'active');
      }
    });
  });
}

// This code has been replaced by the Gemini API functionality in clinics.js
