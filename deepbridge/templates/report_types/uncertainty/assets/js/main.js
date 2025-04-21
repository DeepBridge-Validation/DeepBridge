/**
 * Main entry point for the uncertainty report application
 * Initializes data services and controllers, and sets up the main navigation
 */
import DataExtractor from './modules/data/DataExtractor.js';
import DataModel from './modules/data/DataModel.js';
import ChartFactory from './modules/charts/ChartFactory.js';
import OverviewController from './controllers/overview/OverviewController.js';
import DetailsController from './controllers/details/DetailsController.js';
import CalibrationController from './controllers/calibration/CalibrationController.js';

document.addEventListener('DOMContentLoaded', function() {
  // Initialize data services
  const dataExtractor = new DataExtractor();
  const dataModel = new DataModel();
  const chartFactory = new ChartFactory();
  
  // Initialize controllers
  const overviewController = new OverviewController(window.reportData, dataExtractor, chartFactory);
  const detailsController = new DetailsController(window.reportData, dataExtractor, chartFactory);
  const calibrationController = new CalibrationController(window.reportData, dataExtractor, chartFactory);
  
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
      calibration: calibrationController
    };
  }
  
  // Initialize expandable panels
  initExpandablePanels();
  
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
   * Initialize expandable panels
   */
  function initExpandablePanels() {
    const panels = document.querySelectorAll('.expandable-panel');
    
    panels.forEach(panel => {
      const header = panel.querySelector('.panel-header');
      const content = panel.querySelector('.panel-content');
      const toggleIcon = panel.querySelector('.toggle-icon');
      
      if (header && content) {
        header.addEventListener('click', () => {
          const isExpanded = panel.classList.toggle('expanded');
          
          if (toggleIcon) {
            toggleIcon.textContent = isExpanded ? '▼' : '▶';
          }
          
          // Animate height transition
          if (isExpanded) {
            content.style.maxHeight = content.scrollHeight + 'px';
          } else {
            content.style.maxHeight = '0';
          }
        });
        
        // Initialize in collapsed state
        if (!panel.classList.contains('expanded')) {
          content.style.maxHeight = '0';
        } else {
          content.style.maxHeight = content.scrollHeight + 'px';
        }
      }
    });
  }
  
  /**
   * Export the uncertainty report as PDF or HTML
   */
  function exportReport() {
    // Implementation would depend on specific export library being used
    // Example implementation with confirmation dialog
    const modal = document.createElement('div');
    modal.className = 'export-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>Export Uncertainty Report</h3>
          <button class="close-button">&times;</button>
        </div>
        <div class="modal-body">
          <div class="export-options">
            <label>
              <input type="radio" name="export-format" value="pdf" checked>
              <span>PDF Document</span>
            </label>
            <label>
              <input type="radio" name="export-format" value="html">
              <span>HTML Document</span>
            </label>
            <label>
              <input type="radio" name="export-format" value="json">
              <span>JSON Data (Raw)</span>
            </label>
          </div>
          <div class="export-settings">
            <h4>Include Sections</h4>
            <div class="settings-group">
              <label>
                <input type="checkbox" name="include-section" value="overview" checked>
                <span>Overview</span>
              </label>
              <label>
                <input type="checkbox" name="include-section" value="details" checked>
                <span>Detailed Analysis</span>
              </label>
              <label>
                <input type="checkbox" name="include-section" value="calibration" checked>
                <span>Calibration Analysis</span>
              </label>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="cancel-button">Cancel</button>
          <button class="export-button">Export</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    // Add event listeners
    const closeButton = modal.querySelector('.close-button');
    const cancelButton = modal.querySelector('.cancel-button');
    const exportButton = modal.querySelector('.export-button');
    
    const closeModal = () => {
      document.body.removeChild(modal);
    };
    
    closeButton.addEventListener('click', closeModal);
    cancelButton.addEventListener('click', closeModal);
    
    exportButton.addEventListener('click', () => {
      const format = modal.querySelector('input[name="export-format"]:checked').value;
      const sections = Array.from(
        modal.querySelectorAll('input[name="include-section"]:checked')
      ).map(input => input.value);
      
      // Show loading indicator
      exportButton.innerHTML = '<span class="spinner"></span> Exporting...';
      exportButton.disabled = true;
      
      // Simulate export process (replace with actual implementation)
      setTimeout(() => {
        // Hide modal
        closeModal();
        
        // Show success notification
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.innerHTML = `
          <div class="notification-content">
            <div class="notification-icon">✓</div>
            <div class="notification-message">
              Uncertainty report exported successfully as ${format.toUpperCase()}.
            </div>
          </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove notification after a few seconds
        setTimeout(() => {
          if (notification.parentNode) {
            document.body.removeChild(notification);
          }
        }, 3000);
      }, 1500);
    });
  }
});