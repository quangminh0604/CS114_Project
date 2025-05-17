/**
 * Main JavaScript file for the Framingham Heart Disease Risk Prediction Website
 */

document.addEventListener('DOMContentLoaded', function() {
  // Initialize tooltips
  initializeTooltips();
  
  // Initialize navigation highlighting
  highlightCurrentNavItem();
  
  // Add event listener for mobile menu toggle
  const mobileMenuToggle = document.querySelector('.navbar-toggler');
  if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', function() {
      const navbarCollapse = document.querySelector('.navbar-collapse');
      navbarCollapse.classList.toggle('show');
    });
  }
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
  const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

/**
 * Highlight the current navigation item based on the URL path
 */
function highlightCurrentNavItem() {
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || 
        (href !== '/' && currentPath.startsWith(href))) {
      link.classList.add('active');
    }
  });
}

/**
 * Show a notification message
 * @param {string} message - The message to display
 * @param {string} type - The type of notification ('success', 'error', 'warning', 'info')
 * @param {number} duration - The duration in milliseconds (defaults to 5000)
 */
function showNotification(message, type = 'info', duration = 5000) {
  // Check if notification container exists, create if not
  let notificationContainer = document.getElementById('notification-container');
  if (!notificationContainer) {
    notificationContainer = document.createElement('div');
    notificationContainer.id = 'notification-container';
    notificationContainer.style.position = 'fixed';
    notificationContainer.style.top = '20px';
    notificationContainer.style.right = '20px';
    notificationContainer.style.zIndex = '9999';
    document.body.appendChild(notificationContainer);
  }
  
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `alert alert-${type} alert-dismissible fade show`;
  notification.role = 'alert';
  notification.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
  `;
  
  // Add notification to container
  notificationContainer.appendChild(notification);
  
  // Auto dismiss after duration
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => {
      notificationContainer.removeChild(notification);
    }, 300);
  }, duration);
}

/**
 * Format a number with commas as thousands separators
 * @param {number} num - The number to format
 * @returns {string} The formatted number
 */
function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Convert boolean values to Yes/No strings
 * @param {boolean} value - The boolean value
 * @returns {string} 'Yes' for true, 'No' for false
 */
function booleanToYesNo(value) {
  return value ? 'Yes' : 'No';
}

/**
 * Handle form submission errors
 * @param {Object} error - The error object
 * @param {string} formId - The ID of the form that triggered the error
 */
function handleFormError(error, formId) {
  console.error(`Error in form ${formId}:`, error);
  
  let errorMessage = 'An unexpected error occurred. Please try again.';
  
  if (error.response && error.response.data && error.response.data.error) {
    errorMessage = error.response.data.error;
  } else if (typeof error === 'string') {
    errorMessage = error;
  } else if (error.message) {
    errorMessage = error.message;
  }
  
  showNotification(errorMessage, 'danger');
  
  // Hide any loading indicators
  const loader = document.querySelector(`#${formId} .loader`);
  if (loader) {
    loader.style.display = 'none';
  }
  
  // Re-enable submit buttons
  const submitButton = document.querySelector(`#${formId} button[type="submit"]`);
  if (submitButton) {
    submitButton.disabled = false;
  }
}

/**
 * Show loading indicator in a form
 * @param {string} formId - The ID of the form
 * @param {boolean} isLoading - Whether to show or hide the loader
 */
function showLoading(formId, isLoading = true) {
  const loader = document.querySelector(`#${formId} .loader`);
  const submitButton = document.querySelector(`#${formId} button[type="submit"]`);
  
  if (loader) {
    loader.style.display = isLoading ? 'block' : 'none';
  }
  
  if (submitButton) {
    submitButton.disabled = isLoading;
  }
}

/**
 * Validate a form's inputs
 * @param {HTMLFormElement} form - The form to validate
 * @returns {Object} An object with validation result and error message
 */
function validateForm(form) {
  const inputs = form.querySelectorAll('input, select, textarea');
  let isValid = true;
  let errorMessage = '';
  
  inputs.forEach(input => {
    // Skip inputs that don't need validation
    if (input.type === 'submit' || input.type === 'button' || input.type === 'hidden') {
      return;
    }
    
    // Reset validation state
    input.classList.remove('is-invalid');
    const feedbackElement = input.nextElementSibling;
    if (feedbackElement && feedbackElement.classList.contains('invalid-feedback')) {
      feedbackElement.textContent = '';
    }
    
    // Check for required fields
    if (input.hasAttribute('required') && !input.value.trim()) {
      isValid = false;
      input.classList.add('is-invalid');
      
      if (feedbackElement) {
        feedbackElement.textContent = 'This field is required';
      } else {
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = 'This field is required';
        input.parentNode.insertBefore(feedback, input.nextSibling);
      }
      
      if (!errorMessage) {
        errorMessage = 'Please fill in all required fields';
      }
    }
    
    // Check numeric inputs
    if (input.type === 'number' && input.value) {
      const min = parseFloat(input.getAttribute('min'));
      const max = parseFloat(input.getAttribute('max'));
      const value = parseFloat(input.value);
      
      if (!isNaN(min) && value < min) {
        isValid = false;
        input.classList.add('is-invalid');
        
        if (feedbackElement) {
          feedbackElement.textContent = `Value must be at least ${min}`;
        }
        
        if (!errorMessage) {
          errorMessage = 'Some values are outside the allowed range';
        }
      }
      
      if (!isNaN(max) && value > max) {
        isValid = false;
        input.classList.add('is-invalid');
        
        if (feedbackElement) {
          feedbackElement.textContent = `Value must be at most ${max}`;
        }
        
        if (!errorMessage) {
          errorMessage = 'Some values are outside the allowed range';
        }
      }
    }
  });
  
  return { isValid, errorMessage };
}
