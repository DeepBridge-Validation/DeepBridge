// main.js - placeholder
/**
 * Main entry point for the resilience report application
 * Initializes data services and controllers, and sets up the main navigation
 */
import DataExtractor from './modules/data/DataExtractor.js';
import DataModel from './modules/data/DataModel.js';
import ChartFactory from './modules/charts/ChartFactory.js';
import OverviewController from './controllers/overview/OverviewController.js';
import DetailsController from './controllers/details/DetailsController.js';
import DistributionController from './controllers/distribution/DistributionController.js';

document.addEventListener('DOMContentLoaded', function() {
  // Initialize data services
  const dataExtractor = new DataExtractor();
  const dataModel = new DataModel();
  const chartFactory = new ChartFactory();
  
  // Initialize controllers
  const overviewController = new OverviewController(window.reportData, dataExtractor, chartFactory);
  const detailsController = new DetailsController(window.reportData, dataExtractor, chartFactory);
  const distributionController = new DistributionController(window.reportData, dataExtractor, chartFactory);
  
  // Set up main tab navigation
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
      
      // Trigger tab changed event for lazy loading
      document.dispatchEvent(new CustomEvent('tab-changed', {
        detail: { tabId: tabId }
      }));
    });
  });
  
  // Handle theme toggle if present
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', function() {
      const currentTheme = document.body.classList.contains('dark-theme') ? 'dark' : 'light';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      // Update body class
      document.body.classList.toggle('dark-theme', newTheme === 'dark');
      
      // Update stylesheet
      const themeStylesheet = document.getElementById('theme-stylesheet');
      if (themeStylesheet) {
        themeStylesheet.href = `/assets/css/themes/${newTheme}.css`;
      }
      
      // Store preference in local storage
      localStorage.setItem('preferred-theme', newTheme);
      
      // Trigger event to update charts
      window.dispatchEvent(new CustomEvent('theme-changed', {
        detail: { theme: newTheme }
      }));
    });
  }
  
  // Load user's preferred theme from local storage
  const loadUserTheme = () => {
    const preferredTheme = localStorage.getItem('preferred-theme') || 'light';
    document.body.classList.toggle('dark-theme', preferredTheme === 'dark');
    
    const themeStylesheet = document.getElementById('theme-stylesheet');
    if (themeStylesheet) {
      themeStylesheet.href = `/assets/css/themes/${preferredTheme}.css`;
    }
  };
  
  // Call to load theme
  loadUserTheme();
  
  // Initialize the active tab
  const activeTab = document.querySelector('.main-tabs .tab-btn.active');
  if (activeTab) {
    document.dispatchEvent(new CustomEvent('tab-changed', {
      detail: { tabId: activeTab.getAttribute('data-tab') }
    }));
  }
  
  // Expose controllers for debugging in development
  if (window.reportConfig && window.reportConfig.debug) {
    window.controllers = {
      overview: overviewController,
      details: detailsController,
      distribution: distributionController
    };
  }
  
  // Add window resize handler to refresh charts
  let resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(function() {
      // Trigger chart resize events
      const event = new CustomEvent('chart-resize');
      document.dispatchEvent(event);
    }, 250);
  });
  
  // Add export report functionality if export button exists
  const exportButton = document.getElementById('export-report');
  if (exportButton) {
    exportButton.addEventListener('click', exportReport);
  }
  
  /**
   * Export the report as PDF or HTML
   */
  function exportReport() {
    // Implementation would depend on specific export library being used
    alert('Export functionality would be implemented here');
    
    // Example implementation: convert to PDF using a library
    // if (window.html2pdf) {
    //   const element = document.querySelector('.resilience-report');
    //   const opts = {
    //     filename: 'resilience-report.pdf',
    //     image: { type: 'jpeg', quality: 0.98 },
    //     html2canvas: { scale: 2 },
    //     jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    //   };
    //   html2pdf().set(opts).from(element).save();
    // }
  }
});