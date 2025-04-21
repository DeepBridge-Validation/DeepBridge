// index.js - placeholder
/**
 * Controllers index module for the uncertainty report
 * 
 * This file serves as the central hub for initializing and managing
 * all controllers for the uncertainty visualization report.
 */

// Import controllers
import OverviewController from './overview/OverviewController.js';
import DetailsController from './details/DetailsController.js';
import CalibrationController from './calibration/CalibrationController.js';

// Define controller registry
const controllers = {
  overview: null,
  details: null,
  calibration: null
};

/**
 * Initialize all controllers for the uncertainty report
 * @param {Object} reportData - The full uncertainty report data
 * @param {DataExtractor} dataExtractor - Data extraction utility
 * @param {ChartFactory} chartFactory - Chart creation factory
 */
function initControllers(reportData, dataExtractor, chartFactory) {
  // Initialize individual controllers
  controllers.overview = new OverviewController(reportData, dataExtractor, chartFactory);
  controllers.details = new DetailsController(reportData, dataExtractor, chartFactory);
  controllers.calibration = new CalibrationController(reportData, dataExtractor, chartFactory);
  
  // Return controllers for external access
  return controllers;
}

/**
 * Get a specific controller instance
 * @param {string} controllerName - Name of the controller to retrieve
 * @return {Object} The requested controller instance
 */
function getController(controllerName) {
  return controllers[controllerName] || null;
}

/**
 * Handle tab navigation events
 * @param {string} tabId - ID of the selected tab
 */
function handleTabChange(tabId) {
  // Notify the appropriate controller about tab activation
  switch (tabId) {
    case 'overview':
      if (controllers.overview) controllers.overview.render();
      break;
    case 'details':
      if (controllers.details) controllers.details.render();
      break;
    case 'calibration':
      if (controllers.calibration) controllers.calibration.render();
      break;
    default:
      console.warn(`Unknown tab ID: ${tabId}`);
  }
}

/**
 * Set up main tab navigation
 */
function setupTabNavigation() {
  const mainTabButtons = document.querySelectorAll('.main-tabs .tab-btn');
  const tabContainers = document.querySelectorAll('.tab-content');
  
  mainTabButtons.forEach(button => {
    button.addEventListener('click', function() {
      const tabId = this.getAttribute('data-tab');
      
      // Update active tab button
      mainTabButtons.forEach(btn => btn.classList.remove('active'));
      this.classList.add('active');
      
      // Update visible tab content
      tabContainers.forEach(container => {
        container.classList.toggle('active', container.id === tabId);
      });
      
      // Notify controller about tab change
      handleTabChange(tabId);
    });
  });
}

/**
 * Initialize the active tab on page load
 */
function initActiveTab() {
  const activeTab = document.querySelector('.main-tabs .tab-btn.active');
  if (activeTab) {
    handleTabChange(activeTab.getAttribute('data-tab'));
  }
}

// Export public API
export {
  initControllers,
  getController,
  handleTabChange,
  setupTabNavigation,
  initActiveTab
};