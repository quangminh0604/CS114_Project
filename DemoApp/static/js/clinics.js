/**
 * JavaScript file for the clinic finder functionality using Gemini API
 */

let clinicsList = [];

/**
 * Initialize the clinic finder functionality
 */
function initClinicFinder() {
  const clinicsContainer = document.getElementById('clinics-container');
  if (!clinicsContainer) return;
  
  // Hide initially
  clinicsContainer.style.display = 'none';
}

/**
 * Find nearby heart clinics based on user address
 * @param {string} address - The user's address to search around
 */
function findNearbyHeartClinics(address) {
  if (!address) {
    showNotification('Please enter an address to find nearby clinics.', 'warning');
    return;
  }
  
  const clinicsContainer = document.getElementById('clinics-container');
  if (!clinicsContainer) return;
  
  // Show loading state
  clinicsContainer.innerHTML = `
    <div class="text-center my-5">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
      <p class="mt-3">Finding heart clinics near your location...</p>
    </div>
  `;
  clinicsContainer.style.display = 'block';
  
  // Scroll to clinics section
  clinicsContainer.scrollIntoView({ behavior: 'smooth' });
  
  // Call the API to find clinics
  fetch('/api/find-clinics', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ address: address })
  })
  .then(response => {
    if (!response.ok) {
      return response.json().then(data => {
        throw new Error(data.error || 'Failed to find clinics. Please try again.');
      });
    }
    return response.json();
  })
  .then(data => {
    // Save clinics data
    clinicsList = data.clinics;
    
    // Display clinics
    displayClinics(clinicsList);
    
    // Show notification
    showNotification(`Found ${clinicsList.length} heart clinics near your location.`, 'info');
  })
  .catch(error => {
    console.error('Error finding clinics:', error);
    clinicsContainer.innerHTML = `
      <div class="alert alert-danger" role="alert">
        <i class="fas fa-exclamation-circle me-2"></i>
        Failed to find clinics: ${error.message}
      </div>
    `;
  });
}

/**
 * Display clinics in the UI
 * @param {Array} clinics - List of clinic objects
 */
function displayClinics(clinics) {
  const clinicsContainer = document.getElementById('clinics-container');
  if (!clinicsContainer) return;
  
  // If no clinics found
  if (!clinics || clinics.length === 0) {
    clinicsContainer.innerHTML = `
      <div class="alert alert-warning" role="alert">
        <i class="fas fa-exclamation-triangle me-2"></i>
        No heart clinics found near your location.
      </div>
    `;
    return;
  }
  
  // Create HTML for clinics list
  let html = `
    <h3 class="mb-4">
      <i class="fas fa-heartbeat text-danger me-2"></i>
      Heart Clinics Near You
    </h3>
    <div class="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
  `;
  
  // Add each clinic card
  clinics.forEach(clinic => {
    html += `
      <div class="col">
        <div class="card h-100 clinic-card">
          <div class="card-body">
            <h5 class="card-title">${clinic.name}</h5>
            <p class="card-text mb-1">
              <i class="fas fa-map-marker-alt text-primary me-2"></i> 
              ${clinic.address}
            </p>
            <p class="card-text mb-1">
              <i class="fas fa-phone text-primary me-2"></i> 
              ${clinic.phone}
            </p>
            <p class="card-text mb-3">
              <i class="fas fa-star text-warning me-2"></i> 
              ${clinic.rating} / 5
            </p>
            <p class="card-text">
              <small class="text-muted">${clinic.description}</small>
            </p>
          </div>
          <div class="card-footer bg-transparent">
            <a href="https://www.google.com/maps/search/${encodeURIComponent(clinic.name + ' ' + clinic.address)}" 
              target="_blank" class="btn btn-outline-primary btn-sm w-100">
              <i class="fas fa-directions me-2"></i> Get Directions
            </a>
          </div>
        </div>
      </div>
    `;
  });
  
  html += `
    </div>
    <div class="mt-4 text-center">
      <small class="text-muted">Note: Please call to confirm availability before visiting</small>
    </div>
  `;
  
  // Update container
  clinicsContainer.innerHTML = html;
}